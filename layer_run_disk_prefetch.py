"""
offload_inference.py

Demonstration of layer-by-layer offloading inference with a Llama model whose
weights have been saved as separate layer files (via save_model_layers.py).

Now modified to add a global cache and multiple background threads for asynchronous
disk→CPU prefetching of layers.
"""

import os
import queue
import threading
import torch
import torch.nn as nn

# Pull in Llama classes
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    LlamaConfig,
    AutoTokenizer,
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

# Additional import
from accelerate import init_empty_weights

##############################################################################
# 0. GLOBALS FOR ASYNC DISK PREFETCH
##############################################################################
layer_weights_cache = {}          # Holds prefetched layer state_dicts in CPU memory
prefetch_queue = queue.Queue()    # Tasks for the background threads
stop_prefetch = False             # Signal for stopping the threads
cache_lock = threading.Lock()     # Lock to manage access to the cache
scheduled_layers = set()          # Tracks layers that are scheduled or being loaded

def prefetch_worker(layers_dir: str):
    """
    Background thread worker that continuously waits for layer indices
    on prefetch_queue. When a layer index arrives, it loads that layer
    from disk into CPU memory and stores it in layer_weights_cache.
    """
    global stop_prefetch

    while not stop_prefetch:
        try:
            layer_idx = prefetch_queue.get(timeout=0.2)  # Wait briefly for a task
        except queue.Empty:
            continue
        if layer_idx is None:
            # Means we're shutting down
            prefetch_queue.task_done()
            break
        with cache_lock:
            if layer_idx in layer_weights_cache:
                # Already prefetched by another worker
                scheduled_layers.discard(layer_idx)
                prefetch_queue.task_done()
                continue
        try:
            # Load from disk into CPU
            state_path = os.path.join(layers_dir, f"layer_{layer_idx}.pt")
            loaded_state = torch.load(state_path, map_location="cpu")
            with cache_lock:
                layer_weights_cache[layer_idx] = loaded_state
                scheduled_layers.discard(layer_idx)
        except Exception as e:
            print(f"Error prefetching layer {layer_idx}: {e}")
            with cache_lock:
                scheduled_layers.discard(layer_idx)
        finally:
            # Mark this task complete so that join() can proceed
            prefetch_queue.task_done()


##############################################################################
# 1. Utility functions to load layers on-demand
##############################################################################
def load_layer_weights_from_disk(layer: nn.Module, layer_idx: int, layers_dir: str):
    """
    Load state_dict for a single LlamaDecoderLayer from disk into 'layer'.
    The file must be named: 'layer_{layer_idx}.pt'
    """
    state_path = os.path.join(layers_dir, f"layer_{layer_idx}.pt")
    loaded_state = torch.load(state_path, map_location="cpu")
    layer.load_state_dict(loaded_state)


def load_embed_tokens_from_disk(model, layers_dir: str, device: torch.device):
    emb_path = os.path.join(layers_dir, "embed_tokens.pt")
    emb_state = torch.load(emb_path, map_location="cpu")

    # Step 1) Allocate real storage on CPU
    model.model.embed_tokens.to_empty(device="cpu")

    # Step 2) Load real weights from disk into CPU
    model.model.embed_tokens.load_state_dict(emb_state)

    # Step 3) If you really do want them on GPU, move them now
    if device.type == "cuda":
        model.model.embed_tokens.to(device)


def load_final_norm_from_disk(model, layers_dir: str, device: torch.device):
    norm_path = os.path.join(layers_dir, "final_norm.pt")
    norm_state = torch.load(norm_path, map_location="cpu")

    model.model.norm.to_empty(device="cpu")
    model.model.norm.load_state_dict(norm_state)
    if device.type == "cuda":
        model.model.norm.to(device)


def load_lm_head_from_disk(model, layers_dir: str, device: torch.device):
    lm_head_path = os.path.join(layers_dir, "lm_head.pt")
    lm_head_state = torch.load(lm_head_path, map_location="cpu")

    model.lm_head.to_empty(device="cpu")
    model.lm_head.load_state_dict(lm_head_state)
    if device.type == "cuda":
        model.lm_head.to(device)


##############################################################################
# 2. Layer-by-Layer Offloading Inference Code
##############################################################################
def async_load_module(module: nn.Module, device: torch.device, stream: torch.cuda.Stream):
    """
    Asynchronously load a module's parameters onto the specified device 
    using the given CUDA stream.
    """
    with torch.cuda.stream(stream):
        module.to(device, non_blocking=True)


def forward_module(hidden_states: torch.Tensor,
                   module: nn.Module,
                   stream: torch.cuda.Stream):
    """
    Run a forward pass on the given module using the specified CUDA stream.
    For LlamaDecoderLayer, the output is typically a tuple (hidden_states, presents, ...). 
    We return only the hidden_states.
    """
    with torch.cuda.stream(stream):
        output = module(hidden_states)[0]  # (hidden_states, ...)
    return output


def layer_by_layer_inference(
    model: LlamaForCausalLM,
    input_ids: torch.LongTensor,
    layers_dir: str,
    device: torch.device = torch.device("cuda"),
    prefetch_count: int = 2
) -> torch.Tensor:
    """
    Executes layer-by-layer inference, but now we will also:
      1) Prefetch the next few layers (disk->CPU) in the background threads
      2) Check the global cache first for layers

    Parameters:
        model: The LlamaForCausalLM model.
        input_ids: Tokenized input IDs.
        layers_dir: Directory where layer files are stored.
        device: Torch device to run the model on.
        prefetch_count: Number of layers to prefetch ahead.
    
    Returns:
        logits: The output logits from the model.
    """

    transfer_stream = torch.cuda.Stream(device=device, priority=-1)
    compute_stream = torch.cuda.Stream(device=device, priority=0)

    num_layers = model.config.num_hidden_layers
    layer_loaded_on_gpu = [False] * num_layers

    # 1) Word embeddings
    hidden_states = model.model.embed_tokens(input_ids.to(device))

    final_norm = model.model.norm
    lm_head = model.lm_head

    for i in range(num_layers):
        print(f"Processing layer {i}...")

        # ---------------------------------------------------------------------
        # 2a) Schedule background CPU prefetch for future layers
        #     (makes sure we have them in CPU memory soon).
        # ---------------------------------------------------------------------
        for j in range(i+1, min(i+1+prefetch_count, num_layers)):
            with cache_lock:
                if j not in layer_weights_cache and j not in scheduled_layers:
                    prefetch_queue.put(j)  # schedule for background load
                    scheduled_layers.add(j)

        # ---------------------------------------------------------------------
        # 2b) Get layer i from the global cache if available; otherwise load
        #     from disk synchronously as a fallback.
        # ---------------------------------------------------------------------
        with cache_lock:
            if i in layer_weights_cache:
                # Already in CPU memory from background thread
                loaded_state = layer_weights_cache.pop(i)
            else:
                # If not prefetched yet, we must load it ourselves (blocking)
                print(f"Layer {i} not in cache; loading from disk...")
                state_path = os.path.join(layers_dir, f"layer_{i}.pt")
                loaded_state = torch.load(state_path, map_location="cpu")

        # Create a fresh layer
        decoder_layer = LlamaDecoderLayer(model.config, layer_idx=i)
        decoder_layer.load_state_dict(loaded_state)

        # Asynchronously move the layer's weights onto the GPU
        with torch.cuda.stream(transfer_stream):
            decoder_layer.to(device, non_blocking=True)
        layer_loaded_on_gpu[i] = True

        # Wait for GPU to finish loading
        torch.cuda.current_stream(device).wait_stream(transfer_stream)

        # The forward pass
        batch_size, seq_len = hidden_states.shape[:2]
        position_ids = torch.arange(
            seq_len, dtype=torch.long, device=hidden_states.device
        ).unsqueeze(0).expand(batch_size, -1)

        with torch.cuda.stream(compute_stream):
            hidden_states = decoder_layer(
                hidden_states,
                position_ids=position_ids
            )[0]

        # Wait for the compute stream to finish
        torch.cuda.current_stream(device).wait_stream(compute_stream)

        # Offload from GPU to free memory
        decoder_layer.to("cpu")
        layer_loaded_on_gpu[i] = False
        del decoder_layer

    # Final norm & head
    hidden_states = hidden_states.to(device)  # Ensure hidden_states is on GPU
    hidden_states = final_norm(hidden_states)
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
    prefetch_count: int = 3
):
    """
    Generate tokens from the model, layer by layer, while applying a temperature.

    Parameters:
        model: The LlamaForCausalLM model.
        tokenizer: The tokenizer associated with the model.
        prompt: The input prompt string.
        layers_dir: Directory where layer files are stored.
        max_new_tokens: Number of tokens to generate.
        device: Torch device to run the model on.
        temperature: Sampling temperature.
        prefetch_count: Number of layers to prefetch ahead.
    
    Returns:
        The generated text string.
    """
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            logits = layer_by_layer_inference(
                model, 
                input_ids, 
                layers_dir=layers_dir,
                device=device, 
                prefetch_count=prefetch_count
            )
            next_logit = logits[:, -1, :] / temperature
            probs = torch.softmax(next_logit, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token_id.cpu()], dim=1)

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("Warning: CUDA device not available. This example is intended for a GPU.")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    layers_dir = "/mnt/e/model_layers"
    print(f"Loading config/tokenizer from: {layers_dir}")

    config = LlamaConfig.from_pretrained(layers_dir)
    tokenizer = AutoTokenizer.from_pretrained(layers_dir)

    # Create a truly empty (meta) model skeleton
    with init_empty_weights():
        model = LlamaForCausalLM(config)

    # Now load the small pieces (embed tokens, final norm, lm head) onto GPU or CPU
    load_embed_tokens_from_disk(model, layers_dir, device=device)
    load_final_norm_from_disk(model, layers_dir, device=device)
    load_lm_head_from_disk(model, layers_dir, device=device)

    # Define the number of layers to prefetch ahead and number of workers
    PREFETCH_COUNT = 3
    NUM_PREFETCH_WORKERS = 4  # Adjust based on your system's capabilities

    # Start background prefetch threads
    prefetch_threads = []
    for _ in range(NUM_PREFETCH_WORKERS):
        thread = threading.Thread(target=prefetch_worker, args=(layers_dir,), daemon=True)
        thread.start()
        prefetch_threads.append(thread)

    try:
        prompt_text = "Hello, how are you?"
        output_text = generate_tokens_with_temperature(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt_text,
            layers_dir=layers_dir,
            max_new_tokens=5, 
            device=device, 
            temperature=1.0,
            prefetch_count=PREFETCH_COUNT
        )
        print("Generated text:", output_text)
    finally:
        # Clean up: signal the prefetch threads to stop and wait for them
        stop_prefetch = True
        # Put None for each worker to ensure all threads can exit
        for _ in prefetch_threads:
            prefetch_queue.put(None)
        # Wait for all threads to finish
        for thread in prefetch_threads:
            thread.join()
