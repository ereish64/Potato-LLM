import os
import queue
import threading
import torch
import time
import copy
import math
import multiprocessing

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
    Apply Rotary Position Embeddings (RoPE) to the query and key tensors.

    Args:
        q (torch.Tensor): Query tensor.
        k (torch.Tensor): Key tensor.
        cos (torch.Tensor): Cosine frequencies.
        sin (torch.Tensor): Sine frequencies.
        position_ids (torch.Tensor, optional): Position indices.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Rotated query and key tensors.
    """
    # Handle position IDs for variable sequence lengths
    if position_ids is not None:
        cos = cos[position_ids].unsqueeze(1)  # Shape: [batch_size, 1, seq_len, dim]
        sin = sin[position_ids].unsqueeze(1)
    else:
        cos = cos.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, seq_len, dim]
        sin = sin.unsqueeze(0).unsqueeze(0)
    
    # Split heads for rotation (even and odd indices)
    q1, q2 = q[..., ::2], q[..., 1::2]
    k1, k2 = k[..., ::2], k[..., 1::2]
    
    # Ensure dimensions match for larger models by truncating if necessary
    cos = cos[..., :q1.shape[-1]]
    sin = sin[..., :q1.shape[-1]]
    
    # Apply rotary embeddings to queries
    q_rot = torch.cat([
        q1 * cos - q2 * sin,
        q2 * cos + q1 * sin,
    ], dim=-1)
    
    # Apply rotary embeddings to keys
    k_rot = torch.cat([
        k1 * cos - k2 * sin,
        k2 * cos + k1 * sin,
    ], dim=-1)
    
    return q_rot, k_rot

class PatchedLlamaAttention(LlamaAttention):
    """
    Custom LlamaAttention class with enhancements for larger model sizes and Grouped-Query Attention (GQA).
    """
    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx)
        self.head_dim = config.hidden_size // config.num_attention_heads
        # Support for different numbers of key-value heads
        self.num_key_value_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        self.num_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        
        # Calculate how many times to repeat key-value heads for GQA
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
        
        # Project hidden states to query, key, and value
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape projections to separate heads
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        # Transpose for attention computation (batch, heads, seq_len, head_dim)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # Apply Rotary Position Embeddings if frequencies are provided
        if freqs_cis is not None:
            cos, sin = freqs_cis
            # Generate default position_ids if not provided
            if position_ids is None:
                position_ids = torch.arange(q_len, device=hidden_states.device)
                position_ids = position_ids.unsqueeze(0).expand(bsz, -1)
            
            # Apply RoPE to queries and keys
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin, position_ids=position_ids
            )

        # Handle Grouped-Query Attention by repeating key-value heads
        if self.num_key_value_groups > 1:
            key_states = torch.repeat_interleave(key_states, repeats=self.num_key_value_groups, dim=1)
            value_states = torch.repeat_interleave(value_states, repeats=self.num_key_value_groups, dim=1)

        # Compute scaled dot-product attention weights
        attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2))
        attn_weights = attn_weights / math.sqrt(self.head_dim)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Compute attention probabilities using softmax
        attn_weights = torch.nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)

        # Compute attention output by weighted sum of value states
        attn_output = torch.matmul(attn_weights, value_states)
        # Reshape back to (batch_size, seq_len, hidden_size)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, self.hidden_size)
        # Project back to hidden size
        attn_output = self.o_proj(attn_output)

        # Optionally return attention weights
        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
           
class PatchedLlamaDecoderLayer(LlamaDecoderLayer):
    """
    Custom LlamaDecoderLayer that incorporates the patched attention mechanism with RoPE.
    """
    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx=layer_idx)
        # Replace the original self-attention with the patched version
        self.self_attn = PatchedLlamaAttention(config, layer_idx=layer_idx)
        
        # Initialize the MLP and normalization layers
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
        # Save residual for the skip connection
        residual = hidden_states
        # Apply input layer normalization
        hidden_states = self.input_layernorm(hidden_states)

        # Apply self-attention with patched attention
        attn_output, _, _ = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            freqs_cis=freqs_cis,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )

        # Add the attention output to the residual
        hidden_states = residual + attn_output
        residual = hidden_states
        # Apply post-attention layer normalization
        hidden_states = self.post_attention_layernorm(hidden_states)
        # Apply the MLP
        hidden_states = self.mlp(hidden_states)
        # Add the MLP output to the residual
        hidden_states = residual + hidden_states

        # Return the updated hidden states and placeholders for attention outputs
        return (hidden_states, None, None)

##############################################################################
# 0. GLOBALS FOR ASYNC DISK PREFETCH
##############################################################################

# Dictionary to cache layer weights after loading
layer_weights_cache = {}
# Queue to manage layers to prefetch
prefetch_queue = queue.Queue()
# Flag to signal prefetch workers to stop
stop_prefetch = False
# Lock to ensure thread-safe access to the cache
cache_lock = threading.Lock()
# Condition variable to notify threads when cache is updated
cache_condition = threading.Condition(cache_lock)
# Set to keep track of layers that are scheduled for prefetching
scheduled_layers = set()

##############################################################################
#  Modified prefetch_worker to use PatchedLlamaDecoderLayer
##############################################################################
def prefetch_worker(layers_dir: str, config: AutoConfig, dtype: torch.dtype, patched_layers=None):
    """
    Worker function to prefetch layers from disk into the cache asynchronously.

    Args:
        layers_dir (str): Directory where layer files are stored.
        config (AutoConfig): Model configuration.
        dtype (torch.dtype): Data type for tensors.
        patched_layers (list): List of preloaded patched layers.
    """
    global stop_prefetch
    try:
        while not stop_prefetch:
            try:
                # Attempt to get the next layer index to prefetch
                layer_idx = prefetch_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if layer_idx is None:
                # None is a signal to terminate the worker
                prefetch_queue.task_done()
                break

            with cache_lock:
                # If layer is already cached, skip it
                if layer_idx in layer_weights_cache:
                    scheduled_layers.discard(layer_idx)
                    prefetch_queue.task_done()
                    continue

            try:
                # Retrieve the preloaded patched layer
                decoder_layer = patched_layers[layer_idx]

                with cache_condition:
                    # Cache the layer and notify any waiting threads
                    layer_weights_cache[layer_idx] = decoder_layer
                    scheduled_layers.discard(layer_idx)
                    cache_condition.notify_all()

            except Exception as e:
                print(f"Error prefetching layer {layer_idx}: {e}")
                with cache_condition:
                    # On error, remove the layer from scheduled and notify
                    scheduled_layers.discard(layer_idx)
                    cache_condition.notify_all()
            finally:
                # Mark the task as done
                prefetch_queue.task_done()

    except Exception as e:
        print(f"Prefetch worker failed: {e}")

##############################################################################
# 1. Utility functions to load embeddings, norm, lm_head
##############################################################################
def load_embed_tokens_from_disk(model, layers_dir: str, device: torch.device, dtype: torch.dtype):
    """
    Load the embedding tokens from disk and move them to the specified device.

    Args:
        model: The language model.
        layers_dir (str): Directory containing the layer files.
        device (torch.device): Target device.
        dtype (torch.dtype): Data type for tensors.

    Returns:
        torch.nn.Module: The loaded embedding tokens.
    """
    emb_path = os.path.join(layers_dir, "embed_tokens.pt")
    # Load embedding state dict from disk
    emb_state = torch.load(emb_path, map_location="cpu")
    # Clear existing embeddings and load new state
    model.model.embed_tokens.to_empty(device="cpu")
    model.model.embed_tokens.load_state_dict(emb_state)
    # Move embeddings to the target device if CUDA
    if device.type == "cuda":
        model.model.embed_tokens.to(device, dtype=dtype)
    return model.model.embed_tokens

def load_final_norm_from_disk(model, layers_dir: str, device: torch.device, dtype: torch.dtype):
    """
    Load the final normalization layer from disk and move it to the specified device.

    Args:
        model: The language model.
        layers_dir (str): Directory containing the layer files.
        device (torch.device): Target device.
        dtype (torch.dtype): Data type for tensors.

    Returns:
        torch.nn.Module: The loaded final normalization layer.
    """
    norm_path = os.path.join(layers_dir, "final_norm.pt")
    # Load normalization state dict from disk
    norm_state = torch.load(norm_path, map_location="cpu")
    # Clear existing norm and load new state
    model.model.norm.to_empty(device="cpu")
    model.model.norm.load_state_dict(norm_state)
    # Move norm to the target device if CUDA
    if device.type == "cuda":
        model.model.norm.to(device, dtype=dtype)
    return model.model.norm

def load_lm_head_from_disk(model, layers_dir: str, device: torch.device, dtype: torch.dtype):
    """
    Load the language model head from disk and move it to the specified device.

    Args:
        model: The language model.
        layers_dir (str): Directory containing the layer files.
        device (torch.device): Target device.
        dtype (torch.dtype): Data type for tensors.

    Returns:
        torch.nn.Module: The loaded LM head.
    """
    lm_head_path = os.path.join(layers_dir, "lm_head.pt")
    # Load LM head state dict from disk
    lm_head_state = torch.load(lm_head_path, map_location="cpu")
    # Clear existing LM head and load new state
    model.lm_head.to_empty(device="cpu")
    model.lm_head.load_state_dict(lm_head_state)
    # Move LM head to the target device if CUDA
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
    Precompute the cosine and sine frequencies for Rotary Position Embeddings.

    Args:
        config (AutoConfig): Model configuration.
        max_position (int): Maximum sequence length.
        device (torch.device): Target device.
        dtype (torch.dtype, optional): Data type for tensors.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Cosine and sine frequency tensors.
    """
    head_dim = config.hidden_size // config.num_attention_heads
    
    # Retrieve RoPE settings from config
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
    
    # Apply scaling factor to theta
    theta = rope_theta * scaling_factor
    
    # Generate position indices
    pos = torch.arange(max_position, device=device, dtype=dtype)
    freqs = torch.arange(0, head_dim, 2, device=device, dtype=dtype)
    
    # Compute frequencies with scaling
    freqs = theta ** (-freqs / head_dim)
    angles = pos.unsqueeze(1) * freqs.unsqueeze(0)
    
    # Return cosine and sine of the angles
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
    Perform inference by processing the model layer by layer, offloading each layer to GPU as needed.

    Args:
        model (LlamaForCausalLM): The language model.
        input_ids (torch.LongTensor): Input token IDs.
        layers_dir (str): Directory containing the layer files.
        config (AutoConfig): Model configuration.
        dtype (torch.dtype): Data type for tensors.
        device (torch.device, optional): Target device.
        prefetch_count (int, optional): Number of layers to prefetch.
        embed_layer (torch.nn.Module, optional): Embedding layer.
        final_norm (torch.nn.Module, optional): Final normalization layer.
        lm_head (torch.nn.Module, optional): Language model head.

    Returns:
        torch.Tensor: Logits output from the LM head.
    """
    # Create CUDA streams for data transfer and computation
    transfer_stream = torch.cuda.Stream(device=device, priority=-1)
    compute_stream = torch.cuda.Stream(device=device, priority=0)

    # Extract model parameters from config
    num_layers = config.num_hidden_layers
    hidden_size = config.hidden_size
    num_heads = config.num_attention_heads
    max_seq_len = input_ids.shape[1] + 256  # Additional buffer for new tokens

    # Calculate head dimension
    head_dim = hidden_size // num_heads

    # Precompute RoPE frequencies
    cosines, sines = precompute_freqs_cis(
        config=config,
        max_position=max_seq_len,
        device=device,
        dtype=dtype,
    )

    # Obtain initial hidden states from embedding layer
    hidden_states = embed_layer(input_ids.to(device))

    # Iterate through each layer for inference
    for i in range(num_layers):
        # Prefetch upcoming layers based on prefetch_count
        with cache_condition:
            for j in range(i + 1, min(i + 1 + prefetch_count, num_layers)):
                if j not in layer_weights_cache and j not in scheduled_layers:
                    scheduled_layers.add(j)
                    prefetch_queue.put(j)

        # Ensure the current layer is loaded into the cache
        with cache_condition:
            if i not in layer_weights_cache and i not in scheduled_layers:
                scheduled_layers.add(i)
                prefetch_queue.put(i)
            while i not in layer_weights_cache:
                # Wait until the layer is available in the cache
                cache_condition.wait()
            # Retrieve and remove the layer from the cache
            decoder_layer = layer_weights_cache.pop(i)

        # Move the decoder layer to the device using the transfer stream
        with torch.cuda.stream(transfer_stream):
            decoder_layer.to(device, dtype=dtype, non_blocking=True)

        batch_size, seq_len = hidden_states.shape[:2]

        # Create a causal attention mask to prevent attending to future tokens
        float_mask = torch.zeros(
            (batch_size, 1, seq_len, seq_len),
            dtype=dtype,
            device=device
        )
        causal_mask_bool = torch.tril(
            torch.ones((seq_len, seq_len), dtype=torch.bool, device=device)
        )
        float_mask.masked_fill_(~causal_mask_bool, float('-inf'))

        # Generate position IDs for the current sequence
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # Synchronize streams: wait for data transfer to complete before computation
        torch.cuda.current_stream(device).wait_stream(transfer_stream)

        # Perform computation on the compute stream
        with torch.cuda.stream(compute_stream):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=float_mask,
                position_ids=position_ids,
                freqs_cis=(cosines, sines),
                use_cache=False,
            )[0]

        # Ensure the compute stream completes before proceeding
        torch.cuda.current_stream(device).wait_stream(compute_stream)

        # Clean up by deleting the decoder layer and freeing GPU memory
        del decoder_layer
        torch.cuda.empty_cache()

    # Apply the final normalization layer
    hidden_states = hidden_states.to(device)
    final_norm = final_norm.to(device, dtype=dtype)
    hidden_states = final_norm(hidden_states)

    # Compute logits using the LM head
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
    Generate tokens for a given prompt using temperature sampling.

    Args:
        model: The language model.
        tokenizer: The tokenizer for encoding and decoding.
        prompt (str): The input prompt.
        layers_dir (str): Directory containing the layer files.
        config (AutoConfig): Model configuration.
        dtype (torch.dtype): Data type for tensors.
        max_new_tokens (int, optional): Number of tokens to generate.
        device (torch.device, optional): Target device.
        temperature (float, optional): Temperature for sampling.
        prefetch_count (int, optional): Number of layers to prefetch.
        embed_layer (torch.nn.Module, optional): Embedding layer.
        final_norm (torch.nn.Module, optional): Final normalization layer.
        lm_head (torch.nn.Module, optional): Language model head.

    Returns:
        str: The generated text.
    """
    # Encode the prompt into input IDs
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

    with torch.inference_mode():
        model.eval()
        # Generate tokens iteratively
        for _ in range(max_new_tokens):
            stime = time.time()
            # Perform layer-by-layer inference to get logits
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
            # Select the logits for the last token
            next_logit = logits[:, -1, :] / temperature
            # Replace NaNs and clamp extreme values to stabilize softmax
            next_logit = torch.nan_to_num(next_logit, nan=0.0, posinf=1e4, neginf=-1e4)
            next_logit = torch.clamp(next_logit, min=-50.0, max=50.0)

            # Perform top-k filtering to limit the next token choices
            top_k = 20
            sorted_logits, sorted_indices = torch.sort(next_logit, descending=True)
            kth_val = sorted_logits[:, top_k - 1].unsqueeze(-1)
            filtered_logits = torch.where(
                next_logit < kth_val,
                torch.full_like(next_logit, float('-inf')),
                next_logit
            )
            # Compute probabilities from filtered logits
            probs = torch.softmax(filtered_logits, dim=-1)
            # Sample the next token from the probability distribution
            next_token_id = torch.multinomial(probs, num_samples=1)

            # Append the sampled token to the input IDs
            input_ids = torch.cat([input_ids, next_token_id], dim=1).to(device)
            print(f"Single token generation took {time.time() - stime:.2f}s")

    # Decode the input IDs to generate the final text
    return tokenizer.decode(input_ids[0], skip_special_tokens=False)

def create_layer_cache(args):
    """
    Preprocess and load a specific decoder layer into memory.

    Args:
        args (tuple): Tuple containing config, layer index, and layers directory.

    Returns:
        PatchedLlamaDecoderLayer: The preloaded decoder layer.
    """
    config, i, layers_dir = args
    print(f"Preprocessing layer {i}...")
    with init_empty_weights():
        # Initialize an empty patched decoder layer
        patched_layer = PatchedLlamaDecoderLayer(config, layer_idx=i)
        patched_layer.to_empty(device="cpu")
    # Path to the specific layer's state dict
    state_path = os.path.join(layers_dir, f"layer_{i}.pt")
    # Load the layer's state dict from disk
    loaded_state = torch.load(state_path, map_location="cpu")
    # Load the state dict into the patched layer
    patched_layer.load_state_dict(loaded_state)
    # Set the layer to evaluation mode
    patched_layer.eval()
    return patched_layer

if __name__ == "__main__":
    # Directory containing the model layers (update this path as needed)
    layers_dir = "E:/Llama-3.1-8B-model-layers"

    print(f"Loading config/tokenizer from: {layers_dir}")

    # Load the model configuration from the layers directory
    config = AutoConfig.from_pretrained(layers_dir)

    # Map string dtype from config to torch dtype
    dtype_mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_mapping.get(config.torch_dtype, torch.float16)

    # Set the default data type for torch tensors
    torch.set_default_dtype(dtype)

    with init_empty_weights():
        # Initialize the model with empty weights
        model = LlamaForCausalLM(config)

    print(config)

    # Load the tokenizer with support for custom code
    tokenizer = AutoTokenizer.from_pretrained(layers_dir, trust_remote_code=True)

    # Display special tokens for verification
    print("Special tokens:", tokenizer.all_special_tokens)
    print("Special tokens count:", len(tokenizer.all_special_tokens))

    with init_empty_weights():
        # Re-initialize the model with empty weights
        model = LlamaForCausalLM(config)

    # Load the embedding layer, final normalization, and LM head from disk
    embed_layer = load_embed_tokens_from_disk(model, layers_dir, device=torch.device("cuda"), dtype=dtype)
    final_norm = load_final_norm_from_disk(model, layers_dir, device=torch.device("cuda"), dtype=dtype)
    lm_head = load_lm_head_from_disk(model, layers_dir, device=torch.device("cuda"), dtype=dtype)

    # Externalize prefetch settings or define them as variables
    PREFETCH_COUNT = 8
    NUM_PREFETCH_WORKERS = 8

    # Initialize the target device (CUDA)
    device = torch.device("cuda")

    print("Generating initial layer cache...")

    # Preload all patched layers using multiprocessing for efficiency
    with multiprocessing.Pool() as pool:
        args = []
        for i in range(config.num_hidden_layers):
            args.append((config, i, layers_dir))
        patched_layers = pool.map(create_layer_cache, args)
    
    # Start prefetch worker threads for asynchronous layer loading
    print("launching workers...")
    prefetch_threads = []
    for _ in range(NUM_PREFETCH_WORKERS):
        thread = threading.Thread(
            target=prefetch_worker,
            args=(layers_dir, config, dtype, patched_layers),
            daemon=True
        )
        thread.start()
        prefetch_threads.append(thread)

    try:
        # Define the input prompt for the model
        prompt_text = """You are a helpful AI assistant. Always respond cheerfully and with text.
User: Write a story about a silly dog who wears a bucket on their head.

"""
        # Generate text based on the prompt
        output_text = generate_tokens_with_temperature(
            model=model,
            tokenizer=tokenizer,
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
        # Print the generated text
        print("Generated text:", output_text)

    finally:
        # Gracefully stop all prefetch worker threads
        stop_prefetch = True
        for _ in prefetch_threads:
            prefetch_queue.put(None)  # Signal each worker to stop
        for thread in prefetch_threads:
            thread.join()  # Wait for all workers to finish
