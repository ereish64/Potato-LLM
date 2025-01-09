import os
import queue
import threading
import torch
import time
import copy
import math
import multiprocessing

from torch import nn
from accelerate import init_empty_weights

##############################################################################
# LLaMA 3-Specific Imports
##############################################################################
# In LLaMA 3, the tokenizer is handled via Byte Pair Encoding (BPE) from
# OpenAI's tiktoken instead of sentencepiece. Below is a simple wrapper
# example to replace the AutoTokenizer or PreTrainedTokenizerFast usage
# you typically see with LLaMA 2.
##############################################################################
import tiktoken
from transformers import AutoConfig, LlamaForCausalLM

##############################################################################
# LLaMA 3 MLP (SwiGLU)
##############################################################################
# In some checkpoints, the feedforward dimension (sometimes called n_inner or
# intermediate_size) is NOT strictly 4 * hidden_size. This fix checks config
# for the correct dimension so that we match the weights in the checkpoint.
##############################################################################
class Llama3MLP(nn.Module):
    """
    Example MLP block using SwiGLU for LLaMA 3.
    This replaces LlamaMLP from LLaMA 2.
    """
    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size
        
        # Check if the config specifies an intermediate_size or something like n_inner:
        if hasattr(config, "intermediate_size") and config.intermediate_size is not None:
            intermediate_size = config.intermediate_size
        elif hasattr(config, "n_inner") and config.n_inner is not None:
            intermediate_size = config.n_inner
        else:
            # fallback in case your config doesn't specify it
            intermediate_size = 4 * hidden_size

        # "Gate" and "Up" projections for SwiGLU
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

        # SwiGLU uses SiLU (a variant of Swish) plus a gating mechanism
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(
            self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        )

##############################################################################
# LLaMA 3 RMSNorm
##############################################################################
# LLaMA 3 still uses RMSNorm for normalization similar to LLaMA 2,
# so we keep the same approach (potentially with minor improvements).
##############################################################################
class Llama3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def _norm(self, hidden_states: torch.Tensor):
        return hidden_states * torch.rsqrt(
            (hidden_states.float().pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        ).to(hidden_states.dtype)

    def forward(self, hidden_states):
        return self.weight * self._norm(hidden_states)

##############################################################################
# LLaMA 3 Attention
##############################################################################
# Rotary Embeddings and GQA remain the same conceptually as in LLaMA 2, so
# below is adapted from the original "PatchedLlamaAttention" but renamed for LLaMA 3.
##############################################################################
class PatchedLlama3Attention(nn.Module):
    """
    Enhanced Llama3Attention with support for larger model sizes, GQA, and RoPE.
    """
    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.num_key_value_heads = getattr(config, "num_key_value_heads", self.num_heads)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        # Q, K, V projections
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)

        # Output projection
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor = None,
        freqs_cis=None,
        past_key_value: torch.Tensor = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE if provided
        if freqs_cis is not None:
            cos, sin = freqs_cis
            if position_ids is None:
                position_ids = torch.arange(q_len, device=hidden_states.device)
                position_ids = position_ids.unsqueeze(0).expand(bsz, -1)

            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin, position_ids=position_ids
            )

        # Handle GQA repeating of KV
        if self.num_key_value_groups > 1:
            key_states = torch.repeat_interleave(key_states, repeats=self.num_key_value_groups, dim=1)
            value_states = torch.repeat_interleave(value_states, repeats=self.num_key_value_groups, dim=1)

        attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2))
        attn_weights = attn_weights / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

##############################################################################
# Rotary Positional Embeddings for LLaMA 3
##############################################################################
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    """
    Enhanced Rotary Position Embedding that handles various model sizes.
    Supports both rope_scaling and original implementation.
    Fixes dimensionality issues for larger models.
    """
    if position_ids is not None:
        cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
        sin = sin[position_ids].unsqueeze(1)
    else:
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, dim]
        sin = sin.unsqueeze(0).unsqueeze(0)

    q1, q2 = q[..., ::2], q[..., 1::2]
    k1, k2 = k[..., ::2], k[..., 1::2]

    cos = cos[..., :q1.shape[-1]]  
    sin = sin[..., :q1.shape[-1]]

    q_rot = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)
    k_rot = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1)
    return q_rot, k_rot

##############################################################################
# Patched LLaMA 3 Decoder Layer using RMSNorm + SwiGLU MLP + Patched Attention
##############################################################################
class PatchedLlama3DecoderLayer(nn.Module):
    """
    Enhanced Llama3DecoderLayer that handles different model architectures:
    - PatchedLlama3Attention (with GQA + RoPE)
    - Llama3RMSNorm
    - Llama3MLP (SwiGLU) with dynamic intermediate_size from config
    """
    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.self_attn = PatchedLlama3Attention(config, layer_idx=layer_idx)
        self.input_layernorm = Llama3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Llama3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = Llama3MLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor = None,
        freqs_cis=None,
        past_key_value=None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        attn_output, _, _ = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            freqs_cis=freqs_cis,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )

        hidden_states = residual + attn_output
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return (hidden_states, None, None)

##############################################################################
# 0. GLOBALS FOR ASYNC DISK PREFETCH
##############################################################################
layer_weights_cache = {}
prefetch_queue = queue.Queue()
stop_prefetch = False
cache_lock = threading.Lock()
cache_condition = threading.Condition(cache_lock)
scheduled_layers = set()

##############################################################################
#  Modified prefetch_worker to use PatchedLlama3DecoderLayer
##############################################################################
def prefetch_worker(layers_dir: str, config, dtype: torch.dtype, patched_layers=None):
    global stop_prefetch
    try:
        while not stop_prefetch:
            try:
                layer_idx = prefetch_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if layer_idx is None:
                prefetch_queue.task_done()
                break

            with cache_lock:
                if layer_idx in layer_weights_cache:
                    scheduled_layers.discard(layer_idx)
                    prefetch_queue.task_done()
                    continue

            try:
                decoder_layer = patched_layers[layer_idx]
                with cache_condition:
                    layer_weights_cache[layer_idx] = decoder_layer
                    scheduled_layers.discard(layer_idx)
                    cache_condition.notify_all()

            except Exception as e:
                print(f"Error prefetching layer {layer_idx}: {e}")
                with cache_condition:
                    scheduled_layers.discard(layer_idx)
                    cache_condition.notify_all()
            finally:
                prefetch_queue.task_done()

    except Exception as e:
        print(f"Prefetch worker failed: {e}")

##############################################################################
# 1. Utility functions to load embeddings, norm, lm_head (same concept as LLaMA 2)
##############################################################################
def load_embed_tokens_from_disk(model, layers_dir: str, device: torch.device, dtype: torch.dtype):
    emb_path = os.path.join(layers_dir, "embed_tokens.pt")
    emb_state = torch.load(emb_path, map_location="cpu")
    model.model.embed_tokens.to_empty(device="cpu")
    model.model.embed_tokens.load_state_dict(emb_state)
    if device.type == "cuda":
        model.model.embed_tokens.to(device, dtype=dtype)
    return model.model.embed_tokens

def load_final_norm_from_disk(model, layers_dir: str, device: torch.device, dtype: torch.dtype):
    norm_path = os.path.join(layers_dir, "final_norm.pt")
    norm_state = torch.load(norm_path, map_location="cpu")
    model.model.norm.to_empty(device="cpu")
    model.model.norm.load_state_dict(norm_state)
    if device.type == "cuda":
        model.model.norm.to(device, dtype=dtype)
    return model.model.norm

def load_lm_head_from_disk(model, layers_dir: str, device: torch.device, dtype: torch.dtype):
    lm_head_path = os.path.join(layers_dir, "lm_head.pt")
    lm_head_state = torch.load(lm_head_path, map_location="cpu")
    model.lm_head.to_empty(device="cpu")
    model.lm_head.load_state_dict(lm_head_state)
    if device.type == "cuda":
        model.lm_head.to(device, dtype=dtype)
    return model.lm_head

##############################################################################
# 2. Layer-by-Layer Offloading Inference Code (adapted for LLaMA 3)
##############################################################################
def precompute_freqs_cis(
    config,
    max_position: int,
    device: torch.device,
    dtype: torch.dtype = torch.float16,
):
    """
    Enhanced frequency computation with better support for larger models.
    RoPE logic is conceptually identical to LLaMA 2; any changes for LLaMA 3
    revolve around how we handle rope_theta and rope_scaling in config.
    """
    head_dim = config.hidden_size // config.num_attention_heads
    rope_theta = getattr(config, "rope_theta", 10000.0)
    rope_scaling = getattr(config, "rope_scaling", None)
    scaling_factor = 1.0

    if rope_scaling:
        scaling_type = rope_scaling.get("type", "linear")
        factor = rope_scaling.get("factor", 1.0)
        if scaling_type == "linear":
            scaling_factor = factor
        elif scaling_type == "dynamic":
            scaling_factor = factor ** (head_dim / (head_dim - 2))

    theta = rope_theta * scaling_factor

    pos = torch.arange(max_position, device=device, dtype=dtype)
    freqs = torch.arange(0, head_dim, 2, device=device, dtype=dtype)
    freqs = theta ** (-freqs / head_dim)
    angles = pos.unsqueeze(1) * freqs.unsqueeze(0)

    return angles.cos(), angles.sin()

def layer_by_layer_inference(
    model,
    input_ids: torch.LongTensor,
    layers_dir: str,
    config,
    dtype: torch.dtype,
    device: torch.device = torch.device("cuda"),
    prefetch_count: int = 2,
    embed_layer=None,
    final_norm=None,
    lm_head=None
) -> torch.Tensor:
    transfer_stream = torch.cuda.Stream(device=device, priority=-1)
    compute_stream = torch.cuda.Stream(device=device, priority=0)

    num_layers = config.num_hidden_layers
    hidden_size = config.hidden_size
    num_heads = config.num_attention_heads
    max_seq_len = input_ids.shape[1] + 256
    head_dim = hidden_size // num_heads

    # Precompute RoPE frequencies
    cosines, sines = precompute_freqs_cis(config, max_position=max_seq_len, device=device, dtype=dtype)

    # Embeddings
    hidden_states = embed_layer(input_ids.to(device))

    for i in range(num_layers):
        with cache_condition:
            for j in range(i + 1, min(i + 1 + prefetch_count, num_layers)):
                if j not in layer_weights_cache and j not in scheduled_layers:
                    scheduled_layers.add(j)
                    prefetch_queue.put(j)

        with cache_condition:
            if i not in layer_weights_cache and i not in scheduled_layers:
                scheduled_layers.add(i)
                prefetch_queue.put(i)
            while i not in layer_weights_cache:
                cache_condition.wait()
            decoder_layer = layer_weights_cache.pop(i)

        with torch.cuda.stream(transfer_stream):
            decoder_layer.to(device, dtype=dtype, non_blocking=True)

        batch_size, seq_len = hidden_states.shape[:2]
        float_mask = torch.zeros((batch_size, 1, seq_len, seq_len), dtype=dtype, device=device)
        causal_mask_bool = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device))
        float_mask.masked_fill_(~causal_mask_bool, float('-inf'))

        position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        torch.cuda.current_stream(device).wait_stream(transfer_stream)

        with torch.cuda.stream(compute_stream):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=float_mask,
                position_ids=position_ids,
                freqs_cis=(cosines, sines),
                use_cache=False,
            )[0]

        torch.cuda.current_stream(device).wait_stream(compute_stream)

        # Cleanup
        del decoder_layer
        torch.cuda.empty_cache()

    # Final RMSNorm + LM head
    hidden_states = hidden_states.to(device)
    final_norm = final_norm.to(device, dtype=dtype)
    hidden_states = final_norm(hidden_states)

    lm_head = lm_head.to(device, dtype=dtype)
    logits = lm_head(hidden_states)

    return logits

def generate_tokens_with_temperature(
    model,
    encoding,  # Changed from tokenizer to encoding
    prompt,
    layers_dir,
    config,
    dtype: torch.dtype,
    max_new_tokens=5,
    device=torch.device("cuda"),
    temperature=1.0,
    prefetch_count: int = 2,
    embed_layer=None,
    final_norm=None,
    lm_head=None
):
    # Encode the prompt using tiktoken
    input_ids = torch.tensor([encoding.encode(prompt)], dtype=torch.long).to(device)

    with torch.inference_mode():
        model.eval()
        for _ in range(max_new_tokens):
            stime = time.time()
            logits = layer_by_layer_inference(
                model,
                input_ids,
                layers_dir=layers_dir,
                config=config,
                dtype=dtype,
                device=device,
                prefetch_count=prefetch_count,
                embed_layer=embed_layer,
                final_norm=final_norm,
                lm_head=lm_head
            )
            next_logit = logits[:, -1, :] / temperature
            next_logit = torch.nan_to_num(next_logit, nan=0.0, posinf=1e4, neginf=-1e4)
            next_logit = torch.clamp(next_logit, min=-50.0, max=50.0)

            top_k = 20
            sorted_logits, sorted_indices = torch.sort(next_logit, descending=True)
            kth_val = sorted_logits[:, top_k - 1].unsqueeze(-1)
            filtered_logits = torch.where(
                next_logit < kth_val,
                torch.full_like(next_logit, float('-inf')),
                next_logit
            )
            probs = torch.softmax(filtered_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token_id], dim=1).to(device)
            print(f"Generated token in {time.time() - stime:.2f}s")

    # Decode the tokens back to text using tiktoken
    return encoding.decode(input_ids[0].tolist())

##############################################################################
# 3. Creating and caching the patched LLaMA 3 layers
##############################################################################
def create_layer_cache(args):
    """
    Creates a PatchedLlama3DecoderLayer and loads its state from disk.
    """
    config, i, layers_dir = args
    print(f"Preprocessing layer {i}...")
    with init_empty_weights():
        patched_layer = PatchedLlama3DecoderLayer(config, layer_idx=i)
        patched_layer.to_empty(device="cpu")
    state_path = os.path.join(layers_dir, f"layer_{i}.pt")
    loaded_state = torch.load(state_path, map_location="cpu")
    patched_layer.load_state_dict(loaded_state)
    patched_layer.eval()
    return patched_layer

##############################################################################
# 4. Main Execution for LLaMA 3
##############################################################################
if __name__ == "__main__":
    # Example path to model layers (adjust for your environment)
    layers_dir = "E:/Llama-3.1-8B-model-layers"
    config = AutoConfig.from_pretrained(layers_dir)

    # This snippet maps a string from config to torch dtype
    dtype_mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_mapping.get(config.torch_dtype, torch.float16)
    torch.set_default_dtype(dtype)

    # Create a minimal LLaMA 3 model skeleton with empty weights
    with init_empty_weights():
        model = LlamaForCausalLM(config)

    # Load embeddings, final norm, and lm_head to GPU
    device = torch.device("cuda")
    embed_layer = load_embed_tokens_from_disk(model, layers_dir, device=device, dtype=dtype)
    final_norm = load_final_norm_from_disk(model, layers_dir, device=device, dtype=dtype)
    lm_head = load_lm_head_from_disk(model, layers_dir, device=device, dtype=dtype)

    PREFETCH_COUNT = 8
    NUM_PREFETCH_WORKERS = 8

    # Build the full set of layers in memory (multi-process for speed)
    print("Generating initial layer cache...")
    with multiprocessing.Pool() as pool:
        args = []
        for i in range(config.num_hidden_layers):
            args.append((config, i, layers_dir))
        patched_layers = pool.map(create_layer_cache, args)

    # Launch background prefetch threads
    print("Launching prefetch workers...")
    prefetch_threads = []
    for _ in range(NUM_PREFETCH_WORKERS):
        thread = threading.Thread(
            target=prefetch_worker,
            args=(layers_dir, config, dtype, patched_layers),
            daemon=True
        )
        thread.start()
        prefetch_threads.append(thread)

    # Initialize tiktoken encoding
    encoding = tiktoken.get_encoding("cl100k_base")  # Adjust if necessary

    try:
        prompt_text = (
            "You are a helpful AI assistant. Always respond cheerfully and with text.\n"
            "User: Do you have a name?\n"
            "AI: "
        )
        output_text = generate_tokens_with_temperature(
            model=model,
            encoding=encoding,  # Pass encoding instead of tokenizer
            prompt=prompt_text,
            layers_dir=layers_dir,
            config=config,
            dtype=dtype,
            max_new_tokens=50,
            device=device,
            temperature=0.7,
            prefetch_count=PREFETCH_COUNT,
            embed_layer=embed_layer,
            final_norm=final_norm,
            lm_head=lm_head,
        )
        print("Generated text:", output_text)

    finally:
        # Gracefully stop prefetch threads
        stop_prefetch = True
        for _ in prefetch_threads:
            prefetch_queue.put(None)
        for thread in prefetch_threads:
            thread.join()
