import os
import queue
import threading
import torch
import time
import copy
import math
from torch import nn
from transformers import (
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
    AutoConfig,
    AutoTokenizer,
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
    Enhanced Rotary Position Embedding that handles various model sizes.
    Supports both rope_scaling and original implementation.
    Fixes dimensionality issues for larger models.
    """
    # Handle position IDs for variable sequence lengths
    if position_ids is not None:
        cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
        sin = sin[position_ids].unsqueeze(1)
    else:
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, dim]
        sin = sin.unsqueeze(0).unsqueeze(0)
    
    # Split heads for rotation
    q1, q2 = q[..., ::2], q[..., 1::2]
    k1, k2 = k[..., ::2], k[..., 1::2]
    
    # Ensure dimensions match for larger models
    cos = cos[..., :q1.shape[-1]]  # Truncate to match head dimension
    sin = sin[..., :q1.shape[-1]]
    
    # Apply rotary embeddings
    q_rot = torch.cat([
        q1 * cos - q2 * sin,
        q2 * cos + q1 * sin,
    ], dim=-1)
    
    k_rot = torch.cat([
        k1 * cos - k2 * sin,
        k2 * cos + k1 * sin,
    ], dim=-1)
    
    return q_rot, k_rot

class PatchedLlamaAttention(LlamaAttention):
    """
    Enhanced LlamaAttention with support for larger model sizes and GQA.
    """
    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx)
        self.head_dim = config.hidden_size // config.num_attention_heads
        # Support for different KV head configurations
        self.num_key_value_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        self.num_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        
        # Compute repeats needed for GQA
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        
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
        
        # Project with correct dimensions
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape considering grouped-query attention
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        # Transpose for attention
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # Apply RoPE with proper dimension handling
        if freqs_cis is not None:
            cos, sin = freqs_cis
            # Ensure proper position handling
            if position_ids is None:
                position_ids = torch.arange(q_len, device=hidden_states.device)
                position_ids = position_ids.unsqueeze(0).expand(bsz, -1)
            
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin, position_ids=position_ids
            )

        # Handle GQA by repeating KV heads
        if self.num_key_value_groups > 1:
            key_states = torch.repeat_interleave(
                key_states, 
                repeats=self.num_key_value_groups, 
                dim=1
            )
            value_states = torch.repeat_interleave(
                value_states,
                repeats=self.num_key_value_groups,
                dim=1
            )

        # Compute scaled dot-product attention
        attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2))
        attn_weights = attn_weights / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Compute attention probabilities
        attn_weights = torch.nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)

        # Compute attention output
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
       
class PatchedLlamaDecoderLayer(LlamaDecoderLayer):
    """
    Enhanced LlamaDecoderLayer that handles different model architectures.
    """
    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx=layer_idx)
        self.self_attn = PatchedLlamaAttention(config, layer_idx=layer_idx)
        
        # Initialize MLP with correct dimensions
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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
    """
    Enhanced LlamaDecoderLayer that handles different model architectures.
    """
    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx=layer_idx)
        self.self_attn = PatchedLlamaAttention(config, layer_idx=layer_idx)
        
        # Initialize MLP with correct dimensions
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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
    """
    Subclass of HF's LlamaDecoderLayer to incorporate patched attention with RoPE.
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
def prefetch_worker(layers_dir: str, config: AutoConfig, dtype: torch.dtype):
    """
    Prefetch worker to load layers from disk and cache them into CPU RAM.

    Parameters:
        layers_dir: str - Directory containing layer weights
        config: AutoConfig - Model configuration
        dtype: torch.dtype - Data type for weights
    """
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
# 1. Utility functions to load embeddings, norm, lm_head from the disk to CPU RAM
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
    
    # Get rope settings from config
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
    
    # Apply scaling for different model sizes
    theta = rope_theta * scaling_factor
    
    pos = torch.arange(max_position, device=device, dtype=dtype)
    freqs = torch.arange(0, head_dim, 2, device=device, dtype=dtype)
    
    # Compute frequencies with scaling
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
    Perform layer-by-layer inference on a model with offloaded weights. Loads each layer from CPU ram to GPU then runs inference, rinse and repeat for each layer.

    Parameters:
        model: LlamaForCausalLM - The model to run inference on
        input_ids: torch.LongTensor - The input token IDs
        layers_dir: str - The directory containing the model layers
        config: AutoConfig - The model configuration
        dtype: torch.dtype - The data type to use for inference
        device: torch.device - The device to run inference on
        prefetch_count: int - The number of layers to prefetch
        embed_layer: nn.Embedding - The embedding layer to use for inference
        final_norm: nn.LayerNorm - The final normalization layer
        lm_head: nn.Linear - The LM head layer

    Returns:
        torch.Tensor - The logits for the next token
    """

    transfer_stream = torch.cuda.Stream(device=device, priority=-1)
    compute_stream = torch.cuda.Stream(device=device, priority=0)

    # Get model-specific parameters
    num_layers = config.num_hidden_layers
    hidden_size = config.hidden_size
    num_heads = config.num_attention_heads
    max_seq_len = input_ids.shape[1] + 256

    # Calculate head dimensions for the specific model size
    head_dim = hidden_size // num_heads

    # Precompute RoPE frequencies with model-specific settings
    cosines, sines = precompute_freqs_cis(
        config=config,
        max_position=max_seq_len,
        device=device,
        dtype=dtype,
    )

    # Process embeddings
    hidden_states = embed_layer(input_ids.to(device))

    # Process layers with appropriate dimensions
    for i in range(num_layers):
        print(f"Processing layer {i}/{num_layers}...")
        stime = time.time()

        # Prefetch handling remains the same
        with cache_condition:
            for j in range(i + 1, min(i + 1 + prefetch_count, num_layers)):
                if j not in layer_weights_cache and j not in scheduled_layers:
                    scheduled_layers.add(j)
                    prefetch_queue.put(j)

        # Wait for layer and process
        with cache_condition:
            if i not in layer_weights_cache and i not in scheduled_layers:
                scheduled_layers.add(i)
                prefetch_queue.put(i)
            while i not in layer_weights_cache:
                cache_condition.wait()
            decoder_layer = layer_weights_cache.pop(i)

        # Move layer to device
        with torch.cuda.stream(transfer_stream):
            decoder_layer.to(device, dtype=dtype, non_blocking=True)

        batch_size, seq_len = hidden_states.shape[:2]

        # Create attention mask with appropriate dimensions
        float_mask = torch.zeros(
            (batch_size, 1, seq_len, seq_len),
            dtype=dtype,
            device=device
        )
        causal_mask_bool = torch.tril(
            torch.ones((seq_len, seq_len), dtype=torch.bool, device=device)
        )
        float_mask.masked_fill_(~causal_mask_bool, float('-inf'))

        position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # Process with streams
        torch.cuda.current_stream(device).wait_stream(transfer_stream)

        with torch.cuda.stream(compute_stream):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=float_mask,
                position_ids=position_ids,
                freqs_cis=(cosines, sines),
                use_cache=False,
            )[0]

        # wait for compute to finish. Maybe we don't need this.
        torch.cuda.current_stream(device).wait_stream(compute_stream)

        # Cleanup
        del decoder_layer
        torch.cuda.empty_cache()
        print(f"Layer {i} took {time.time() - stime:.2f}s")

    # Final processing
    hidden_states = hidden_states.to(device)
    final_norm = final_norm.to(device, dtype=dtype)
    hidden_states = final_norm(hidden_states)

    lm_head = lm_head.to(device, dtype=dtype)
    logits = lm_head(hidden_states)

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
    prefetch_count: int = 2,
    embed_layer=None,
    final_norm=None,
    lm_head=None
):
    """
    Generate new tokens from a prompt using the model with layer-by-layer inference to generate each token
    
    Parameters:
        model: LlamaForCausalLM - The model to generate tokens with
        tokenizer: PreTrainedTokenizerFast - The tokenizer to use for encoding/decoding
        prompt: str - The input prompt to generate tokens from
        layers_dir: str - The directory containing the model layers
        config: AutoConfig - The model configuration
        dtype: torch.dtype - The data type to use for inference
        max_new_tokens: int - The maximum number of tokens to generate
        device: torch.device - The device to run inference on
        temperature: float - The temperature to use for sampling
        prefetch_count: int - The number of layers to prefetch
        embed_layer: nn.Embedding - The embedding layer to use for inference
        final_norm: nn.LayerNorm - The final normalization layer
        lm_head: nn.Linear - The LM head layer
    
    Returns:
        str - The generated tokens as a string
    """
    # tokenize our input prompt
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

    # switch model to eval mode
    with torch.inference_mode():
        model.eval()
        # loop until we have generated the max number of new tokens
        for _ in range(max_new_tokens):
            # run layer-by-layer inference for one token
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
            # get the logits for the next token
            next_logit = logits[:, -1, :] / temperature
            next_logit = torch.nan_to_num(next_logit, nan=0.0, posinf=1e4, neginf=-1e4)
            next_logit = torch.clamp(next_logit, min=-50.0, max=50.0)

            # calculate top-k logits. This needs to be moved to the function parameters at some point.
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
    # layers_dir = "E:/Llama-3.1-8B/"  # Update this path as needed
    layers_dir = "E:/Llama-2-7B-model-layers"

    print(f"Loading config/tokenizer from: {layers_dir}")

    # Load configuration
    config = AutoConfig.from_pretrained(layers_dir)

    # Map string dtype to the model's torch dtype
    dtype_mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_mapping.get(config.torch_dtype, torch.float16)

    # Set default dtype based on config
    torch.set_default_dtype(dtype)

    print(config)

    # Load tokenizer to CPU RAM
    tokenizer = PreTrainedTokenizerFast.from_pretrained(layers_dir)
    print("Special tokens:", tokenizer.all_special_tokens)
    print("Special tokens count:", len(tokenizer.all_special_tokens))

    # Initialize model with empty weights
    with init_empty_weights():
        model = LlamaForCausalLM(config)

    # Load embeddings, final norm, lm_head from the disk to CPU RAM
    embed_layer = load_embed_tokens_from_disk(model, layers_dir, device=torch.device("cuda"), dtype=dtype)
    final_norm = load_final_norm_from_disk(model, layers_dir, device=torch.device("cuda"), dtype=dtype)
    lm_head = load_lm_head_from_disk(model, layers_dir, device=torch.device("cuda"), dtype=dtype)

    # set the number of prefetch workers
    NUM_PREFETCH_WORKERS = 8
    PREFETCH_COUNT = NUM_PREFETCH_WORKERS

    # set the target device which in this case is a CUDA GPU.
    device = torch.device("cuda")
    
    # Start prefetch threads
    prefetch_threads = []
    for _ in range(NUM_PREFETCH_WORKERS):
        thread = threading.Thread(
            target=prefetch_worker,
            args=(layers_dir, config, dtype),
            daemon=True
        )
        thread.start()
        prefetch_threads.append(thread)

    try:
        # example prompt
        prompt_text = """You are a helpful AI assistant. Always respond cheerfully and with text.
User: Do you have a name?
</s>
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
