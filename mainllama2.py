import os
import queue
import threading
import torch
import time
import copy

# Pull in Llama classes
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizerFast,
    LlamaConfig,
    AutoTokenizer,
    PreTrainedTokenizerFast,
    LlamaTokenizer,
    AutoConfig,
)
# We no longer need to import LlamaDecoderLayer directly from HF,
# because we'll use our patched version below.

# Additional import
from accelerate import init_empty_weights

##############################################################################
#  PATCH ADDED HERE: Custom classes
##############################################################################
import math
import torch
from torch import nn
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaMLP,
    LlamaRMSNorm,
)

def apply_rotary_pos_emb(q, k, cos, sin):
    """
    Same helper you had before. 
    Expects q, k of shape [batch, n_heads, seq_len, head_dim].
    Expects cos, sin of shape [seq_len, n_heads, head_dim/2].
    """
    # q, k shape: [b, h, seq, d]
    q1, q2 = q[..., ::2], q[..., 1::2]
    k1, k2 = k[..., ::2], k[..., 1::2]
    q_rot = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
    k_rot = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)
    return q_rot, k_rot

class PatchedLlamaAttention(LlamaAttention):
    """
    Subclass of HF's LlamaAttention that checks for freqs_cis=(cos, sin) in forward()
    and applies apply_rotary_pos_emb to Q/K if present.
    """
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor = None,
        freqs_cis=None,  # <-- new!
        past_key_value: torch.Tensor = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ):
        bsz, q_len, _ = hidden_states.size()

        # 1) Project Q/K/V
        query_states = self.q_proj(hidden_states)
        key_states   = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # 2) Reshape to [bsz, num_heads, seq_len, head_dim]
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states   = key_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 3) Apply RoPE if freqs_cis is not None
        if freqs_cis is not None:
            cos, sin = freqs_cis
            # slice to the current seq_len if needed
            cos = cos[:q_len]
            sin = sines = sin[:q_len]
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # 4) Scaled dot-product attention
        attn_weights = torch.matmul(
            query_states, key_states.transpose(-1, -2)
        ) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_probs = nn.functional.softmax(attn_weights, dim=-1, dtype=query_states.dtype)
        attn_output = torch.matmul(attn_probs, value_states)

        # [bsz, num_heads, seq_len, head_dim] -> [bsz, seq_len, hidden_size]
        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output, None, None


class PatchedLlamaDecoderLayer(LlamaDecoderLayer):
    """
    Same as HF's LlamaDecoderLayer, but we override forward(...) 
    to add a freqs_cis argument and pass it to self.self_attn.
    """
    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx=layer_idx)
        # Replace self_attn with our patched attention
        self.self_attn = PatchedLlamaAttention(config, layer_idx=layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor = None,
        freqs_cis=None,       # <--- new param
        past_key_value=None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Pass freqs_cis to our patched self_attn
        attn_output, _, _ = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            freqs_cis=freqs_cis,  # <---
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
#  Modified prefetch_worker to use PatchedLlamaDecoderLayer
##############################################################################
def prefetch_worker(layers_dir: str, config: LlamaConfig):
    """
    Same logic as your original, but we instantiate PatchedLlamaDecoderLayer 
    instead of the stock LlamaDecoderLayer.
    """
    global stop_prefetch
    try:
        config.torch_dtype = torch.bfloat16

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
                with init_empty_weights():
                    # Instantiate patched decoder layer
                    decoder_layer = PatchedLlamaDecoderLayer(config, layer_idx=layer_idx)

                decoder_layer.to_empty(device="cpu")

                # Load from disk
                state_path = layers_dir + f"/layer_{layer_idx}.pt"
                loaded_state = torch.load(state_path, map_location="cpu")
                decoder_layer.load_state_dict(loaded_state)
                decoder_layer.eval()

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
# 1. Utility functions to load embeddings, norm, lm_head
##############################################################################
def load_embed_tokens_from_disk(model, layers_dir: str, device: torch.device):
    emb_path = layers_dir+"/embed_tokens.pt"
    emb_state = torch.load(emb_path, map_location="cpu")
    model.model.embed_tokens.to_empty(device="cpu")
    model.model.embed_tokens.load_state_dict(emb_state)
    if device.type == "cuda":
        model.model.embed_tokens.to(device)
    return model.model.embed_tokens

def load_final_norm_from_disk(model, layers_dir: str, device: torch.device):
    norm_path = layers_dir+"/final_norm.pt"
    norm_state = torch.load(norm_path, map_location="cpu")
    model.model.norm.to_empty(device="cpu")
    model.model.norm.load_state_dict(norm_state)
    if device.type == "cuda":
        model.model.norm.to(device)
    return model.model.norm

def load_lm_head_from_disk(model, layers_dir: str, device: torch.device):
    lm_head_path = layers_dir+"/lm_head.pt"
    lm_head_state = torch.load(lm_head_path, map_location="cpu")
    model.lm_head.to_empty(device="cpu")
    model.lm_head.load_state_dict(lm_head_state)
    if device.type == "cuda":
        model.lm_head.to(device)
    return model.lm_head

##############################################################################
# 2. Layer-by-Layer Offloading Inference Code
##############################################################################
def precompute_freqs_cis(
    hidden_size: int,
    num_attention_heads: int,
    max_position: int,
    base=10000.0,
    dtype=torch.bfloat16,
    device="cuda",
):
    head_dim = config.hidden_size // config.num_attention_heads
    position_ids = torch.arange(0, max_position, dtype=dtype, device=device)
    dims = torch.arange(0, head_dim, 2, dtype=dtype, device=device)
    frequencies = (1.0 / (base ** (dims / head_dim)))
    angles = position_ids[:, None] * frequencies[None, :]
    cosines = angles.cos().unsqueeze(1).expand(-1, num_attention_heads, -1)
    sines = angles.sin().unsqueeze(1).expand(-1, num_attention_heads, -1)
    return cosines, sines

def layer_by_layer_inference(
    model: LlamaForCausalLM,
    input_ids: torch.LongTensor,
    layers_dir: str,
    device: torch.device = torch.device("cuda"),
    prefetch_count: int = 2,
    embed_layer=None,
    final_norm=None,
    lm_head=None
) -> torch.Tensor:
    transfer_stream = torch.cuda.Stream(device=device, priority=-1)
    compute_stream = torch.cuda.Stream(device=device, priority=0)

    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    num_heads = model.config.num_attention_heads
    max_seq_len = input_ids.shape[1] + 256

    # Calculate head_dim correctly
    head_dim = hidden_size // num_heads
    assert head_dim % 2 == 0, "head_dim must be even for RoPE."

    # Precompute RoPE
    cosines, sines = precompute_freqs_cis(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        max_position=max_seq_len,
        device=device,
        dtype=torch.bfloat16,
    )

    # Word embeddings
    hidden_states = model.model.embed_tokens(input_ids.to(device))

    for i in range(num_layers):
        print(f"Processing layer {i}...")
        stime = time.time()

        # Prefetch future layers
        with cache_condition:
            for j in range(i + 1, min(i + 1 + prefetch_count, num_layers)):
                if j not in layer_weights_cache and j not in scheduled_layers:
                    scheduled_layers.add(j)
                    prefetch_queue.put(j)

        # Wait for layer i to appear in cache
        with cache_condition:
            if i not in layer_weights_cache and i not in scheduled_layers:
                scheduled_layers.add(i)
                prefetch_queue.put(i)
            while i not in layer_weights_cache:
                cache_condition.wait()
            decoder_layer = layer_weights_cache.pop(i)

        # Move layer i onto GPU
        with torch.cuda.stream(transfer_stream):
            decoder_layer.to(device, dtype=torch.bfloat16, non_blocking=False)

        batch_size, seq_len = hidden_states.shape[:2]

        # Build a *float* causal mask with -inf for invalid positions
        float_mask = torch.zeros(
            (batch_size, 1, seq_len, seq_len),
            dtype=torch.bfloat16,
            device=device
        )
        causal_mask_bool = torch.tril(
            torch.ones((seq_len, seq_len), dtype=torch.bool, device=device)
        )
        float_mask.masked_fill_(~causal_mask_bool, float('-inf'))

        position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # Slice and permute RoPE for current seq_len
        cos = cosines[:seq_len].permute(1, 0, 2).unsqueeze(0)  # Shape: [1, num_heads, seq_len, head_dim//2]
        sin = sines[:seq_len].permute(1, 0, 2).unsqueeze(0)    # Shape: [1, num_heads, seq_len, head_dim//2]

        torch.cuda.synchronize(transfer_stream)

        with torch.cuda.stream(compute_stream):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=float_mask,   # pass float mask
                position_ids=position_ids,
                freqs_cis=(cos, sin),         # Correctly shaped tensors
                use_cache=False,
            )[0]

        torch.cuda.synchronize()

        # Offload layer
        decoder_layer.to("cpu")
        del decoder_layer
        torch.cuda.empty_cache()
        print(f"Layer {i} took {time.time() - stime:.2f}s")

    # Final norm & head
    hidden_states = hidden_states.to(device)
    final_norm = final_norm.to(device)
    hidden_states = final_norm(hidden_states)

    lm_head = lm_head.to(device)
    logits = lm_head(hidden_states)

    final_norm.to("cpu")
    lm_head.to("cpu")

    return logits

def generate_tokens_with_temperature(
    model,
    tokenizer,
    prompt,
    layers_dir,
    max_new_tokens=5,
    device=torch.device("cuda"),
    temperature=1.0,
    prefetch_count: int = 2,
    embed_layer=None,
    final_norm=None,
    lm_head=None
):
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

    with torch.inference_mode():
        model.eval()
        for _ in range(max_new_tokens):
            logits = layer_by_layer_inference(
                model,
                input_ids,
                layers_dir=layers_dir,
                device=device,
                prefetch_count=prefetch_count,
                embed_layer=embed_layer,
                final_norm=final_norm,
                lm_head=lm_head
            )
            with torch.device("cpu"):
                logits.to("cpu")
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

    return tokenizer.decode(input_ids[0], skip_special_tokens=False)

if __name__ == "__main__":
    device = torch.device("cuda")
    torch.set_default_dtype(torch.bfloat16)

    layers_dir = "F:/7b_model_layers"
    print(f"Loading config/tokenizer from: {layers_dir}")

    # Immediately after loading the config:
    config = AutoConfig.from_pretrained(layers_dir)

    # If needed, also set intermediate_size, etc.
    # config.intermediate_size = 4096  # for example

    with init_empty_weights():
        model = LlamaForCausalLM(config)
    
    print(config)

    tokenizer = PreTrainedTokenizerFast.from_pretrained(layers_dir)
    print("Special tokens:", tokenizer.all_special_tokens)
    print("Special tokens count:", len(tokenizer.all_special_tokens))

    with init_empty_weights():
        model = LlamaForCausalLM(config)

    # Load embeddings, final norm, lm_head
    embed_layer = load_embed_tokens_from_disk(model, layers_dir, device=device)
    final_norm = load_final_norm_from_disk(model, layers_dir, device=device)
    lm_head = load_lm_head_from_disk(model, layers_dir, device=device)

    PREFETCH_COUNT = 12
    NUM_PREFETCH_WORKERS = 12

    prefetch_threads = []
    for _ in range(NUM_PREFETCH_WORKERS):
        thread = threading.Thread(
            target=prefetch_worker,
            args=(layers_dir, config),
            daemon=True
        )
        thread.start()
        prefetch_threads.append(thread)

    try:
        prompt_text = """You are a helpful AI assistant. Always respond cheerfully and with text.

User: Do you have a name?

"""
        output_text = generate_tokens_with_temperature(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt_text,
            layers_dir=layers_dir,
            max_new_tokens=5,
            device=device, 
            temperature=0.6,
            prefetch_count=PREFETCH_COUNT,
            embed_layer=embed_layer,
            final_norm=final_norm,
            lm_head=lm_head,
        )
        print("Generated text:", output_text)

    finally:
        stop_prefetch = True
        for _ in prefetch_threads:
            prefetch_queue.put(None)
        for thread in prefetch_threads:
            thread.join()
