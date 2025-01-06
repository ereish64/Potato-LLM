import os
import queue
import threading
import torch
import time
import math
from torch import nn
from accelerate import init_empty_weights

# Import necessary components from Hugging Face Transformers
from transformers import (
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
    AutoConfig,
    LlamaConfig,
)

# Attempt to import LlamaAttention and related classes from Llama3.1
# Adjust the import path based on the actual structure of Llama3.1
try:
    from transformers.models.llama3_1.modeling_llama3_1 import (
        LlamaAttention,
        LlamaDecoderLayer,
        LlamaMLP,
        LlamaRMSNorm,
    )
except ImportError:
    # Fallback to previous import if Llama3.1 uses the same module structure
    from transformers.models.llama.modeling_llama import (
        LlamaAttention,
        LlamaDecoderLayer,
        LlamaMLP,
        LlamaRMSNorm,
    )

##############################################################################
#  PATCH ADDED HERE: Custom Classes
##############################################################################

def apply_rotary_pos_emb(q, cos, sin):
    """
    Applies rotary positional embeddings to the query or key tensors.
    
    Args:
        q (torch.Tensor): Query or Key tensor of shape [batch, num_heads, seq_len, head_dim].
        cos (torch.Tensor): Cosine tensor of shape [1, num_heads, seq_len, head_dim//2].
        sin (torch.Tensor): Sine tensor of shape [1, num_heads, seq_len, head_dim//2].
    
    Returns:
        torch.Tensor: Rotated tensor of the same shape as input `q`.
    """
    # Debug shapes
    print(f"Applying RoPE:")
    print(f"  q shape: {q.shape}")
    print(f"  cos shape: {cos.shape}")
    print(f"  sin shape: {sin.shape}")
    
    # Split the last dimension into even and odd parts
    q1, q2 = q[..., ::2], q[..., 1::2]
    print(f"  q1 shape: {q1.shape}")
    print(f"  q2 shape: {q2.shape}")
    
    # Apply rotary transformation
    q_rot = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
    print(f"  q_rot shape: {q_rot.shape}")
    return q_rot

class PatchedLlamaAttention(LlamaAttention):
    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx=layer_idx)
        
        # Configuration for Query
        self.num_heads_q = config.num_attention_heads
        self.head_dim_q = config.hidden_size // self.num_heads_q
        assert self.head_dim_q * self.num_heads_q == config.hidden_size, "head_dim_q * num_heads_q must equal hidden_size"
        
        # Configuration for Key and Value
        self.kv_hidden_size = self.k_proj.weight.shape[0]  # e.g., 4096
        self.num_heads_kv = self.kv_hidden_size // self.head_dim_q  # Should align with config
        self.head_dim_kv = self.head_dim_q  # Ensuring head_dim_kv equals head_dim_q
        assert self.head_dim_kv * self.num_heads_kv == self.kv_hidden_size, "head_dim_kv * num_heads_kv must equal kv_hidden_size"
        
        print(f"Initialized PatchedLlamaAttention with:")
        print(f"  Query: num_heads={self.num_heads_q}, head_dim={self.head_dim_q}")
        print(f"  Key/Value: num_heads={self.num_heads_kv}, head_dim={self.head_dim_kv}")
        print(f"  Hidden Size: {config.hidden_size}")

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

        # Project Q/K/V
        query_states = self.q_proj(hidden_states)  # [bsz, q_len, hidden_size_q]
        key_states   = self.k_proj(hidden_states)  # [bsz, q_len, hidden_size_kv]
        value_states = self.v_proj(hidden_states)  # [bsz, q_len, hidden_size_kv]

        # Debug shapes
        print(f"Q_proj Output Shape: {query_states.shape}")
        print(f"K_proj Output Shape: {key_states.shape}")
        print(f"V_proj Output Shape: {value_states.shape}")

        # Reshape Q, K, V
        query_states = query_states.view(bsz, q_len, self.num_heads_q, self.head_dim_q).transpose(1, 2)  # [bsz, num_heads_q, q_len, head_dim_q]
        key_states   = key_states.view(bsz, q_len, self.num_heads_kv, self.head_dim_kv).transpose(1, 2)      # [bsz, num_heads_kv, q_len, head_dim_kv]
        value_states = value_states.view(bsz, q_len, self.num_heads_kv, self.head_dim_kv).transpose(1, 2)  # [bsz, num_heads_kv, q_len, head_dim_kv]

        print(f"Reshaped Query States Shape: {query_states.shape}")
        print(f"Reshaped Key States Shape: {key_states.shape}")
        print(f"Reshaped Value States Shape: {value_states.shape}")

        # Apply RoPE if freqs_cis is not None
        if freqs_cis is not None:
            cos_q, sin_q, cos_k, sin_k = freqs_cis  # Unpack separate cos and sin for Q and K

            # Slice to the current seq_len
            cos_q = cos_q[:, :, :q_len, :]  # [1, num_heads_q, q_len, head_dim//2]
            sin_q = sin_q[:, :, :q_len, :]  # [1, num_heads_q, q_len, head_dim//2]
            cos_k = cos_k[:, :, :q_len, :]  # [1, num_heads_kv, q_len, head_dim//2]
            sin_k = sin_k[:, :, :q_len, :]  # [1, num_heads_kv, q_len, head_dim//2]

            print(f"cos_q sliced shape: {cos_q.shape}")
            print(f"sin_q sliced shape: {sin_q.shape}")
            print(f"cos_k sliced shape: {cos_k.shape}")
            print(f"sin_k sliced shape: {sin_k.shape}")

            # Apply RoPE to Q
            query_states = apply_rotary_pos_emb(query_states, cos_q, sin_q)

            # Apply RoPE to K
            key_states = apply_rotary_pos_emb(key_states, cos_k, sin_k)

        # Ensure head_dim_q == head_dim_kv
        if self.head_dim_q != self.head_dim_kv:
            raise ValueError("head_dim_q and head_dim_kv must be equal for scaled dot-product attention.")

        # Scaled dot-product attention
        # To handle different num_heads for Q and K/V, we need to broadcast K and V
        # Assuming num_heads_q is divisible by num_heads_kv
        if self.num_heads_q % self.num_heads_kv != 0:
            raise ValueError("num_heads_q must be divisible by num_heads_kv for broadcasting.")
        
        heads_per_kv_head = self.num_heads_q // self.num_heads_kv
        key_states = key_states.repeat(1, heads_per_kv_head, 1, 1)      # [bsz, num_heads_q, q_len, head_dim_kv]
        value_states = value_states.repeat(1, heads_per_kv_head, 1, 1)  # [bsz, num_heads_q, q_len, head_dim_kv]

        print(f"After broadcasting, Key States Shape: {key_states.shape}")
        print(f"After broadcasting, Value States Shape: {value_states.shape}")

        attn_weights = torch.matmul(
            query_states, key_states.transpose(-1, -2)
        ) / math.sqrt(self.head_dim_q)

        print(f"Attention Weights Shape: {attn_weights.shape}")

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_probs = nn.functional.softmax(attn_weights, dim=-1, dtype=query_states.dtype)
        attn_output = torch.matmul(attn_probs, value_states)

        print(f"Attention Output Shape: {attn_output.shape}")

        # [bsz, num_heads_q, q_len, head_dim_q] -> [bsz, q_len, hidden_size_q]
        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, self.num_heads_q * self.head_dim_q)
        attn_output = self.o_proj(attn_output)

        print(f"Final Attention Output Shape: {attn_output.shape}")

        return attn_output, None, None

class PatchedLlamaDecoderLayer(LlamaDecoderLayer):
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
                state_path = os.path.join(layers_dir, f"layer_{layer_idx}.pt")
                loaded_state = torch.load(state_path, map_location="cpu")
                decoder_layer.load_state_dict(loaded_state)
                decoder_layer.eval()

                # Debugging: Print layer details
                print(f"Prefetched Layer {layer_idx}")
                print(f"  Q_proj shape: {decoder_layer.self_attn.q_proj.weight.shape}")
                print(f"  K_proj shape: {decoder_layer.self_attn.k_proj.weight.shape}")
                print(f"  V_proj shape: {decoder_layer.self_attn.v_proj.weight.shape}")

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
    emb_path = os.path.join(layers_dir, "embed_tokens.pt")
    emb_state = torch.load(emb_path, map_location="cpu")
    model.model.embed_tokens.to_empty(device="cpu")
    model.model.embed_tokens.load_state_dict(emb_state)
    if device.type == "cuda":
        model.model.embed_tokens.to(device)
    print(f"Loaded embed_tokens from {emb_path} with shape {model.model.embed_tokens.weight.shape}")
    return model.model.embed_tokens

def load_final_norm_from_disk(model, layers_dir: str, device: torch.device):
    norm_path = os.path.join(layers_dir, "final_norm.pt")
    norm_state = torch.load(norm_path, map_location="cpu")
    model.model.norm.to_empty(device="cpu")
    model.model.norm.load_state_dict(norm_state)
    if device.type == "cuda":
        model.model.norm.to(device)
    print(f"Loaded final_norm from {norm_path} with shape {model.model.norm.weight.shape}")
    return model.model.norm

def load_lm_head_from_disk(model, layers_dir: str, device: torch.device):
    lm_head_path = os.path.join(layers_dir, "lm_head.pt")
    lm_head_state = torch.load(lm_head_path, map_location="cpu")
    
    # Verify the loaded state matches the expected dimensions
    expected_shape = (model.config.vocab_size, model.config.hidden_size)
    loaded_shape = lm_head_state['weight'].shape if 'weight' in lm_head_state else lm_head_state['lm_head.weight'].shape
    print(f"Loaded lm_head weight shape: {loaded_shape}, Expected: {expected_shape}")
    
    if loaded_shape != expected_shape:
        raise ValueError(f"LM head weight shape mismatch: got {loaded_shape}, expected {expected_shape}")
    
    model.lm_head.to_empty(device="cpu")
    model.lm_head.load_state_dict(lm_head_state)
    if device.type == "cuda":
        model.lm_head.to(device)
    print(f"Loaded lm_head from {lm_head_path} with shape {model.lm_head.weight.shape}")
    return model.lm_head

##############################################################################
# 2. RoPE Precomputation Function
##############################################################################
def precompute_rotary_embeddings(
    head_dim: int,
    num_heads: int,
    max_position: int,
    base=10000.0,
    dtype=torch.bfloat16,
    device="cuda",
):
    """
    Precomputes the cosine and sine tensors for Rotary Positional Embeddings.

    Args:
        head_dim (int): Dimension per attention head.
        num_heads (int): Number of attention heads.
        max_position (int): Maximum sequence length.
        base (float, optional): Base for frequency computation. Defaults to 10000.0.
        dtype (torch.dtype, optional): Data type. Defaults to torch.bfloat16.
        device (str or torch.device, optional): Device. Defaults to "cuda".

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (cos, sin) tensors.
    """
    position_ids = torch.arange(0, max_position, dtype=dtype, device=device)
    dims = torch.arange(0, head_dim, 2, dtype=dtype, device=device)
    frequencies = 1.0 / (base ** (dims / head_dim))
    angles = position_ids[:, None] * frequencies[None, :]  # [max_position, head_dim//2]

    # Compute cos and sin
    cos = torch.cos(angles).repeat(1, num_heads)  # [max_position, num_heads * head_dim//2]
    sin = torch.sin(angles).repeat(1, num_heads)  # [max_position, num_heads * head_dim//2]

    # Reshape to [max_position, num_heads, head_dim//2]
    cos = cos.view(max_position, num_heads, head_dim // 2)
    sin = sin.view(max_position, num_heads, head_dim // 2)

    # Add batch dimension
    cos = cos.unsqueeze(0)  # [1, max_position, num_heads, head_dim//2]
    sin = sin.unsqueeze(0)  # [1, max_position, num_heads, head_dim//2]

    # Permute to [1, num_heads, max_position, head_dim//2]
    cos = cos.permute(0, 2, 1, 3)
    sin = sin.permute(0, 2, 1, 3)

    print(f"Precomputed RoPE: cos shape {cos.shape}, sin shape {sin.shape}")
    return cos, sin

##############################################################################
# 3. Layer-by-Layer Offloading Inference Code
##############################################################################
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
    num_heads_q = model.config.num_attention_heads
    num_heads_kv = num_heads_q  # Corrected: Align K/V heads with Q heads
    max_seq_len = input_ids.shape[1] + 256  # Adjust as needed

    # Calculate head_dim correctly
    head_dim_q = hidden_size // num_heads_q
    head_dim_kv = head_dim_q  # As per the patched attention class
    assert head_dim_q % 2 == 0, "head_dim_q must be even for RoPE."

    print(f"Precomputing RoPE for Q and K")
    # Precompute RoPE for Q and K
    cos_q, sin_q = precompute_rotary_embeddings(
        head_dim=head_dim_q,
        num_heads=num_heads_q,
        max_position=max_seq_len,
        device=device,
        dtype=torch.bfloat16,
    )
    cos_k, sin_k = precompute_rotary_embeddings(
        head_dim=head_dim_kv,
        num_heads=num_heads_kv,
        max_position=max_seq_len,
        device=device,
        dtype=torch.bfloat16,
    )

    # Word embeddings
    hidden_states = model.model.embed_tokens(input_ids.to(device))
    print(f"Word embeddings shape: {hidden_states.shape}")

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
            print(f"Layer {i} loaded into cache.")

        # Move layer i onto GPU
        with torch.cuda.stream(transfer_stream):
            decoder_layer.to(device, dtype=torch.bfloat16, non_blocking=False)

        batch_size, seq_len = hidden_states.shape[:2]
        print(f"Current hidden_states shape: {hidden_states.shape}")

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
        print(f"Causal mask shape: {float_mask.shape}")

        position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        print(f"Position IDs shape: {position_ids.shape}")

        torch.cuda.synchronize(transfer_stream)

        with torch.cuda.stream(compute_stream):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=float_mask,   # pass float mask
                position_ids=position_ids,
                freqs_cis=(cos_q, sin_q, cos_k, sin_k),   # Separate cos and sin for Q and K
                use_cache=False,
            )[0]

        torch.cuda.synchronize()
        print(f"Layer {i} completed in {time.time() - stime:.2f}s")

        # Offload layer
        decoder_layer.to("cpu")
        del decoder_layer
        torch.cuda.empty_cache()

    # Final norm & head
    print(f"Applying final normalization and LM head.")
    hidden_states = hidden_states.to(device)
    final_norm = final_norm.to(device)
    hidden_states = final_norm(hidden_states)
    print(f"After final norm, hidden_states shape: {hidden_states.shape}")

    lm_head = lm_head.to(device)
    logits = lm_head(hidden_states)
    print(f"Logits shape: {logits.shape}")

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
    print(f"Input IDs shape: {input_ids.shape}")

    with torch.inference_mode():
        model.eval()
        for step in range(max_new_tokens):
            print(f"Generating token {step + 1}/{max_new_tokens}...")
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
                logits = logits.to("cpu")
                input_ids = input_ids.to("cpu")
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

                print(f"Next token ID: {next_token_id.item()}")

                input_ids = torch.cat([input_ids, next_token_id], dim=1).to(device)
                print(f"Updated input_ids shape: {input_ids.shape}")

    return tokenizer.decode(input_ids[0], skip_special_tokens=False)

##############################################################################
# 4. Complete Prefetch Worker and Utility Functions
##############################################################################
def prefetch_worker(layers_dir: str, config: LlamaConfig):
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
                state_path = os.path.join(layers_dir, f"layer_{layer_idx}.pt")
                loaded_state = torch.load(state_path, map_location="cpu")
                decoder_layer.load_state_dict(loaded_state)
                decoder_layer.eval()

                # Debugging: Print layer details
                print(f"Prefetched Layer {layer_idx}")
                print(f"  Q_proj shape: {decoder_layer.self_attn.q_proj.weight.shape}")
                print(f"  K_proj shape: {decoder_layer.self_attn.k_proj.weight.shape}")
                print(f"  V_proj shape: {decoder_layer.self_attn.v_proj.weight.shape}")

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
# 5. Main Execution Block
##############################################################################
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.bfloat16)

    layers_dir = "F:/7b_model_layers"  # Update this path as necessary
    print(f"Loading config/tokenizer from: {layers_dir}")

    # Load configuration
    config = AutoConfig.from_pretrained(layers_dir)
    # If Llama3.1 introduces new configuration parameters, set them here
    # Example:
    # config.new_parameter = value

    print("Model Configuration:")
    print(config)

    # Load tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained(layers_dir)
    print("Special tokens:", tokenizer.all_special_tokens)
    print("Special tokens count:", len(tokenizer.all_special_tokens))
    print(f"Tokenizer vocab size: {len(tokenizer)}")

    # Verify and correct vocab_size if necessary
    print(f"Model config vocab_size before correction: {config.vocab_size}")
    if config.vocab_size != len(tokenizer):
        print("Correcting config.vocab_size to match tokenizer.")
        config.vocab_size = len(tokenizer)
    print(f"Model config vocab_size after correction: {config.vocab_size}")

    with init_empty_weights():
        model = LlamaForCausalLM(config)

    # Load embeddings, final norm, lm_head
    embed_layer = load_embed_tokens_from_disk(model, layers_dir, device=device)
    final_norm = load_final_norm_from_disk(model, layers_dir, device=device)
    lm_head = load_lm_head_from_disk(model, layers_dir, device=device)

    PREFETCH_COUNT = 2  # Adjust based on your GPU memory and bandwidth
    NUM_PREFETCH_WORKERS = 4  # Adjust based on your CPU and I/O capabilities

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

Assistant:"""
        output_text = generate_tokens_with_temperature(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt_text,
            layers_dir=layers_dir,
            max_new_tokens=5,  # Increased for better response
            device=device, 
            temperature=0.7,
            prefetch_count=PREFETCH_COUNT,
            embed_layer=embed_layer,
            final_norm=final_norm,
            lm_head=lm_head,
        )
        print("Generated text:", output_text)

    finally:
        # Signal prefetch threads to stop
        stop_prefetch = True
        for _ in prefetch_threads:
            prefetch_queue.put(None)
        for thread in prefetch_threads:
            thread.join()
