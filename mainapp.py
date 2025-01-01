import os
import queue
import threading
import torch
import torch.nn as nn
import time

# Pull in Llama classes
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    LlamaConfig,
    AutoTokenizer,
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from accelerate import init_empty_weights

##############################################################################
# 0. GLOBALS FOR ASYNC DISK PREFETCH
##############################################################################
layer_weights_cache = {}          # Holds prefetched LlamaDecoderLayer modules in CPU memory
prefetch_queue = queue.Queue()    # Tasks for the background threads
stop_prefetch = False             # Signal for stopping the threads
cache_lock = threading.Lock()     # Lock to manage access to the cache
# A condition variable associated with cache_lock.
# Used to signal "layer i is now in cache" to any thread that is waiting.
cache_condition = threading.Condition(cache_lock)

scheduled_layers = set()          # Tracks layers that are scheduled or being loaded


def prefetch_worker(layers_dir: str, config: LlamaConfig):
    """
    Background thread worker that continuously waits for layer indices
    on prefetch_queue. When a layer index arrives, it constructs a LlamaDecoderLayer,
    loads weights from disk, and stores the entire module in CPU memory in layer_weights_cache.
    """
    global stop_prefetch

    # Force the worker’s layers to be created at the same dtype as the config’s
    desired_dtype = getattr(config, "torch_dtype", torch.float32)
    if desired_dtype is None:
        desired_dtype = torch.float32

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
            # -----------------------------------------------------------------
            # Create the layer in "meta" mode, then switch to CPU + correct dtype
            # -----------------------------------------------------------------
            with init_empty_weights():
                decoder_layer = LlamaDecoderLayer(config, layer_idx=layer_idx)
            
            # Switch to CPU with the appropriate dtype
            decoder_layer.to_empty(device="cpu")

            # Load from disk into CPU
            state_path = os.path.join(layers_dir, f"layer_{layer_idx}.pt")
            loaded_state = torch.load(state_path, map_location="cpu")
            decoder_layer.load_state_dict(loaded_state)

            # Store in cache (thread-safe) and notify any waiters
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


##############################################################################
# 1. Utility functions to load embeddings, norm, lm_head
##############################################################################
def load_embed_tokens_from_disk(model, layers_dir: str, device: torch.device):
    emb_path = os.path.join(layers_dir, "embed_tokens.pt")
    emb_state = torch.load(emb_path, map_location="cpu")

    # Step 1) Allocate real storage on CPU
    model.model.embed_tokens.to_empty(device="cpu")

    # Step 2) Load real weights from disk into CPU
    model.model.embed_tokens.load_state_dict(emb_state)

    # Step 3) Optionally move them onto GPU
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
def forward_module(hidden_states: torch.Tensor,
                   module: nn.Module,
                   stream: torch.cuda.Stream):
    """
    Run a forward pass on the given module using the specified CUDA stream.
    For LlamaDecoderLayer, the output is typically a tuple (hidden_states, ...). 
    We return only the hidden_states.
    """
    with torch.cuda.stream(stream):
        output = module(hidden_states)[0]  # (hidden_states, ...)
    return output

def build_causal_attention_mask(seq_len: int, 
                                batch_size: int = 1,
                                dtype: torch.dtype = torch.float16, 
                                device: torch.device = torch.device("cuda")):
    """
    Build a [batch_size, 1, seq_len, seq_len] causal mask
    with 0.0 where attention is allowed, and -inf where it is blocked.
    """
    # Start with an all-zero mask, then fill upper-right region (above diagonal) with -inf
    mask = torch.zeros((batch_size, 1, seq_len, seq_len), dtype=dtype, device=device)
    mask[:, :, torch.arange(seq_len)[:, None], torch.arange(seq_len)] = 0.0  # Redundant, but for clarity
    # Fill the strictly upper-triangular part with -inf
    mask = mask.fill_(0.0).float()  # ensure float to hold -inf
    mask.triu_(1)  # set everything above diagonal to 1.0
    mask = mask.masked_fill(mask == 1.0, float("-inf"))
    return mask


def layer_by_layer_inference(
    model: LlamaForCausalLM,
    input_ids: torch.LongTensor,
    layers_dir: str,
    device: torch.device = torch.device("cuda"),
    prefetch_count: int = 2,
):
    """
    Executes layer-by-layer inference, re-running the entire sequence from scratch,
    but now we build a 4D causal mask that LLaMA expects.
    """
    transfer_stream = torch.cuda.Stream(device=device, priority=-1)
    compute_stream = torch.cuda.Stream(device=device, priority=0)

    num_layers = model.config.num_hidden_layers
    batch_size, seq_len = input_ids.shape

    # --------------------------------------------------------------------------
    # Build the 4D causal mask that matches LLaMA's indexing
    # shape: [batch_size, 1, seq_len, seq_len]
    # --------------------------------------------------------------------------
    causal_mask = build_causal_attention_mask(
        seq_len=seq_len, 
        batch_size=batch_size,
        dtype=torch.float32,  # or model's dtype
        device=device
    )

    # Generate position_ids: shape [batch_size, seq_len]
    position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)
    position_ids = position_ids.expand(batch_size, -1)

    # 1) Word embeddings
    input_ids = input_ids.to(device)
    hidden_states = model.model.embed_tokens(input_ids)

    final_norm = model.model.norm
    lm_head = model.lm_head

    for i in range(num_layers):
        stime = time.time()
        print(f"Processing layer {i}...")

        # ---------------------------
        # 2a) Prefetch next layers
        # ---------------------------
        with cache_condition:
            for j in range(i + 1, min(i + 1 + prefetch_count, num_layers)):
                if j not in layer_weights_cache and j not in scheduled_layers:
                    scheduled_layers.add(j)
                    prefetch_queue.put(j)

        # ---------------------------
        # 2b) Wait for this layer
        # ---------------------------
        with cache_condition:
            if i not in layer_weights_cache and i not in scheduled_layers:
                scheduled_layers.add(i)
                prefetch_queue.put(i)

            while i not in layer_weights_cache:
                cache_condition.wait()

            decoder_layer = layer_weights_cache.pop(i)

        # Move the layer's weights to GPU
        with torch.cuda.stream(transfer_stream):
            decoder_layer.to(device, non_blocking=True)

        # ---------------------------
        # 2c) Forward pass
        # ---------------------------
        with torch.cuda.stream(compute_stream):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids
            )[0]

        # Wait for the compute to finish
        torch.cuda.current_stream(device).wait_stream(compute_stream)

        # Offload from GPU to CPU
        decoder_layer.to("cpu")
        del decoder_layer
        torch.cuda.empty_cache()
        print(f"Layer {i} took {time.time() - stime:.2f} seconds")

    # Final norm & head
    hidden_states = hidden_states.to(device)
    final_norm = final_norm.to(device)
    hidden_states = final_norm(hidden_states)

    lm_head = lm_head.to(device)
    logits = lm_head(hidden_states)

    # (Optional) offload final layers
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
    prefetch_count: int = 2
):
    """
    Generate tokens from the model, layer by layer, while applying a temperature.
    Now uses corrected position IDs and an attention mask for the entire sequence
    each time. This still re-runs from scratch on every new token (no KV caching).
    """
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            # Re-run the entire sequence from scratch, but with correct position info:
            logits = layer_by_layer_inference(
                model, 
                input_ids, 
                layers_dir=layers_dir,
                device=device, 
                prefetch_count=prefetch_count
            )
            next_logit = logits[:, -1, :] / temperature
            # Nan-safe clamp
            next_logit = torch.nan_to_num(next_logit, nan=0.0, posinf=1e4, neginf=-1e4)
            next_logit = torch.clamp(next_logit, min=-50.0, max=50.0)
            probs = torch.softmax(next_logit, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)  # shape (B,1)

            # Append new token to input_ids
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

    # If you want to fix a desired dtype in config (e.g. float16), do:
    # config.torch_dtype = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(layers_dir)

    # Create a truly empty (meta) model skeleton
    with init_empty_weights():
        model = LlamaForCausalLM(config)

    # Now load the small pieces (embed tokens, final norm, lm head) 
    # onto GPU or CPU
    load_embed_tokens_from_disk(model, layers_dir, device=device)
    load_final_norm_from_disk(model, layers_dir, device=device)
    load_lm_head_from_disk(model, layers_dir, device=device)

    # Define the number of layers to prefetch and number of workers
    PREFETCH_COUNT = 10
    NUM_PREFETCH_WORKERS = 10

    # Start background prefetch threads
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
        prompt_text = "Hello, how are you?\n\n"
        output_text = generate_tokens_with_temperature(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt_text,
            layers_dir=layers_dir,
            max_new_tokens=2, 
            device=device, 
            temperature=0.6,
            prefetch_count=PREFETCH_COUNT
        )
        print("Generated text:", output_text)
    finally:
        # Clean up: signal the prefetch threads to stop and wait for them
        stop_prefetch = True
        for _ in prefetch_threads:
            prefetch_queue.put(None)
        for thread in prefetch_threads:
            thread.join()
