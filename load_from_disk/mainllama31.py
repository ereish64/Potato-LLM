import os
import queue
import threading
import torch
import time
import math
from torch import nn
from transformers import (
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
    AutoConfig,
)
from accelerate import init_empty_weights
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaMLP,
    LlamaRMSNorm,
)

##############################################################################
#  PATCH ADDED HERE: Custom classes
##############################################################################

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    """
    Enhanced Rotary Position Embedding for Llama3.
    """
    if position_ids is not None:
        cos = cos[position_ids].unsqueeze(1)
        sin = sin[position_ids].unsqueeze(1)
    else:
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
    
    q1, q2 = q[..., ::2], q[..., 1::2]
    k1, k2 = k[..., ::2], k[..., 1::2]
    cos = cos[..., :q1.shape[-1]]
    sin = sin[..., :q1.shape[-1]]
    q_rot = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)
    k_rot = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1)
    return q_rot, k_rot

class PatchedLlamaAttention(LlamaAttention):
    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx)
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_key_value_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        self.num_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

    def forward(self, hidden_states, attention_mask, position_ids=None, freqs_cis=None, past_key_value=None,
                output_attentions=False, use_cache=False):
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if freqs_cis is not None:
            cos, sin = freqs_cis
            if position_ids is None:
                position_ids = torch.arange(q_len, device=hidden_states.device).unsqueeze(0).expand(bsz, -1)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if self.num_key_value_groups > 1:
            key_states = torch.repeat_interleave(key_states, self.num_key_value_groups, dim=1)
            value_states = torch.repeat_interleave(value_states, self.num_key_value_groups, dim=1)

        attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn_weights += attention_mask
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, self.hidden_size)
        return self.o_proj(attn_output), None, None

class PatchedLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx)
        self.self_attn = PatchedLlamaAttention(config, layer_idx)

    def forward(self, hidden_states, attention_mask, position_ids=None, freqs_cis=None, past_key_value=None,
                use_cache=False, output_attentions=False):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_output, _, _ = self.self_attn(hidden_states, attention_mask, position_ids, freqs_cis, past_key_value,
                                           output_attentions, use_cache)
        hidden_states = residual + attn_output
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states, None, None
##############################################################################

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
def prefetch_worker(layers_dir: str, config: AutoConfig, dtype: torch.dtype):
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
                # Create layer in empty state
                with init_empty_weights():
                    decoder_layer = PatchedLlamaDecoderLayer(config, layer_idx=layer_idx)

                decoder_layer.to_empty(device="cpu")
                state_path = os.path.join(layers_dir, f"layer_{layer_idx}.pt")
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
# 2. Layer-by-Layer Offloading Inference Code
##############################################################################
def precompute_freqs_cis(
    config: AutoConfig,
    max_position: int,
    device: torch.device,
    dtype: torch.dtype = torch.float16,
):
    """
    Enhanced frequency computation with better support for larger models.
    """
    head_dim = config.hidden_size // config.num_attention_heads
    
    # Check if rope_scaling or rope_theta is set
    rope_theta = getattr(config, "rope_theta", 10000.0)
    rope_scaling = getattr(config, "rope_scaling", None)
    scaling_factor = 1.0
    
    # If rope_scaling is present, adjust the rope frequency
    if rope_scaling:
        scaling_type = rope_scaling.get("type", "linear")
        factor = rope_scaling.get("factor", 1.0)
        if scaling_type == "linear":
            scaling_factor = factor
        elif scaling_type == "dynamic":
            # Simple approach; some variants scale differently
            scaling_factor = factor ** (head_dim / (head_dim - 2))
    
    theta = rope_theta * scaling_factor
    
    pos = torch.arange(max_position, device=device, dtype=dtype)
    freqs = torch.arange(0, head_dim, 2, device=device, dtype=dtype)
    
    # Compute frequencies
    freqs = theta ** (-freqs / head_dim)
    angles = pos.unsqueeze(1) * freqs.unsqueeze(0)
    
    return angles.cos(), angles.sin()

def layer_by_layer_inference(
    model: LlamaForCausalLM,
    input_ids: torch.LongTensor,
    layers_dir: str,
    config: AutoConfig,
    dtype: torch.dtype,
    device: torch.device = torch.device("cuda"),
    prefetch_count: int = 2,
    embed_layer=None,
    final_norm=None,
    lm_head=None
) -> torch.Tensor:
    """
    Executes a forward pass layer-by-layer, offloading each layer to GPU
    just-in-time and freeing memory. Prefetches subsequent layers on CPU.
    """
    transfer_stream = torch.cuda.Stream(device=device, priority=-1)
    compute_stream = torch.cuda.Stream(device=device, priority=0)

    # Model-specific parameters
    num_layers = config.num_hidden_layers
    hidden_size = config.hidden_size
    num_heads = config.num_attention_heads
    max_seq_len = input_ids.shape[1] + 256

    # Precompute RoPE frequencies with model-specific settings
    cosines, sines = precompute_freqs_cis(
        config=config,
        max_position=max_seq_len,
        device=device,
        dtype=dtype,
    )

    # Process embeddings
    hidden_states = embed_layer(input_ids.to(device))

    # Iterate over each layer
    for i in range(num_layers):
        print(f"Processing layer {i}/{num_layers}...")
        stime = time.time()

        # Prefetch the next few layers
        with cache_condition:
            for j in range(i + 1, min(i + 1 + prefetch_count, num_layers)):
                if j not in layer_weights_cache and j not in scheduled_layers:
                    scheduled_layers.add(j)
                    prefetch_queue.put(j)

        # Ensure the layer we need is prefetched
        with cache_condition:
            if i not in layer_weights_cache and i not in scheduled_layers:
                scheduled_layers.add(i)
                prefetch_queue.put(i)
            while i not in layer_weights_cache:
                cache_condition.wait()
            decoder_layer = layer_weights_cache.pop(i)

        # Move to GPU
        with torch.cuda.stream(transfer_stream):
            decoder_layer.to(device, dtype=dtype, non_blocking=True)

        batch_size, seq_len = hidden_states.shape[:2]

        # Create the causal attention mask
        float_mask = torch.zeros(
            (batch_size, 1, seq_len, seq_len),
            dtype=dtype,
            device=device
        )
        causal_mask_bool = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device))
        float_mask.masked_fill_(~causal_mask_bool, float('-inf'))

        position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # Ensure any transfers are done before compute
        torch.cuda.current_stream(device).wait_stream(transfer_stream)

        # Compute with the compute stream
        with torch.cuda.stream(compute_stream):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=float_mask,
                position_ids=position_ids,
                freqs_cis=(cosines, sines),
                use_cache=False,
            )[0]

        # Wait for the compute stream to finish
        torch.cuda.current_stream(device).wait_stream(compute_stream)

        # Cleanup
        del decoder_layer
        torch.cuda.empty_cache()
        print(f"Layer {i} took {time.time() - stime:.2f}s")

    # Final LN and head
    hidden_states = hidden_states.to(device)
    final_norm = final_norm.to(device, dtype=dtype)
    hidden_states = final_norm(hidden_states)

    lm_head = lm_head.to(device, dtype=dtype)
    logits = lm_head(hidden_states)

    # Offload final layers to CPU after compute
    final_norm.to("cpu")
    lm_head.to("cpu")

    return logits

def generate_tokens_with_temperature(
    model,
    tokenizer,
    prompt,
    layers_dir,
    config,
    dtype: torch.dtype,
    max_new_tokens=5,
    device=torch.device("cuda"),
    temperature=1.0,
    top_p=0.9,  # Adding nucleus sampling parameter
    prefetch_count: int = 2,
    embed_layer=None,
    final_norm=None,
    lm_head=None
):
    """
    Example text generation that uses our layer-by-layer approach
    with temperature sampling, top-k, and top-p (nucleus sampling).
    """
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

    with torch.inference_mode():
        model.eval()
        for _ in range(max_new_tokens):
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
            # Temperature scaling and clamp
            next_logit = logits[:, -1, :] / temperature
            next_logit = torch.nan_to_num(next_logit, nan=0.0, posinf=1e4, neginf=-1e4)
            next_logit = torch.clamp(next_logit, min=-50.0, max=50.0)

            # Sample on CPU to reduce GPU overhead
            with torch.device("cpu"):
                sorted_logits, sorted_indices = torch.sort(next_logit, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                # Apply top-p filtering
                cumulative_mask = cumulative_probs > top_p
                cumulative_mask[..., 1:] = cumulative_mask[..., :-1].clone()  # Shift mask
                cumulative_mask[..., 0] = False  # Ensure at least one token remains

                sorted_logits[cumulative_mask] = float('-inf')
                probs = torch.softmax(sorted_logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)

                next_token_id = torch.gather(sorted_indices, 1, next_token_id)

            input_ids = torch.cat([input_ids, next_token_id.to(device)], dim=1)

    return tokenizer.decode(input_ids[0], skip_special_tokens=False)

if __name__ == "__main__":
    layers_dir = "E:/Llama-3.1-8B-model-layers"

    print(f"Loading config/tokenizer from: {layers_dir}")

    # Load configuration
    config = AutoConfig.from_pretrained(layers_dir)

    # Map string dtype to torch dtype
    dtype_mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_mapping.get(config.torch_dtype, torch.bfloat16)

    # Set default dtype based on config
    torch.set_default_dtype(dtype)

    # Initialize an empty model
    with init_empty_weights():
        model = LlamaForCausalLM(config)

    tokenizer = PreTrainedTokenizerFast.from_pretrained(layers_dir)
    print("Special tokens:", tokenizer.all_special_tokens)
    print("Special tokens count:", len(tokenizer.all_special_tokens))

    # Load embeddings, final norm, lm_head from disk
    embed_layer = load_embed_tokens_from_disk(model, layers_dir, device=torch.device("cuda"), dtype=dtype)
    final_norm = load_final_norm_from_disk(model, layers_dir, device=torch.device("cuda"), dtype=dtype)
    lm_head = load_lm_head_from_disk(model, layers_dir, device=torch.device("cuda"), dtype=dtype)

    # Externalize prefetch settings if desired
    PREFETCH_COUNT = 10
    NUM_PREFETCH_WORKERS = PREFETCH_COUNT

    device = torch.device("cuda")
    
    # Spin up prefetch threads
    prefetch_threads = []
    for _ in range(NUM_PREFETCH_WORKERS):
        thread = threading.Thread(
            target=prefetch_worker,
            args=(layers_dir, config, dtype),
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
            config=config,
            dtype=dtype,
            max_new_tokens=7,
            device=device,
            temperature=0.5,
            prefetch_count=PREFETCH_COUNT,
            embed_layer=embed_layer,
            final_norm=final_norm,
            lm_head=lm_head,
            top_p=0.9
        )
        print("Generated text:", output_text)

    finally:
        # Clean up threads
        stop_prefetch = True
        for _ in prefetch_threads:
            prefetch_queue.put(None)
        for thread in prefetch_threads:
            thread.join()
