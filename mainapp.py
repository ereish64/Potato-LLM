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
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

# Additional import
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
    # If you prefer bfloat16, set config.torch_dtype = torch.bfloat16, etc.
    # desired_dtype = getattr(config, "torch_dtype", torch.bfloat16)
    # if desired_dtype is None:
    #     desired_dtype = torch.bfloat16
    try:
        config.torch_dtype = torch.bfloat16

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
                    # If your local LlamaDecoderLayer does not accept layer_idx,
                    # remove layer_idx=layer_idx below
                    decoder_layer = LlamaDecoderLayer(config, layer_idx=layer_idx)
                
                # Switch to CPU with the appropriate dtype
                decoder_layer.to_empty(device="cpu")

                # Load from disk into CPU
                state_path = layers_dir+f"/layer_{layer_idx}.pt"
                loaded_state = torch.load(state_path, map_location="cpu")
                decoder_layer.load_state_dict(loaded_state)
                decoder_layer.eval()
                # After loading state_dict
                for key, value in loaded_state.items():
                    if key not in decoder_layer.state_dict():
                        print(f"Unexpected key {key} in state_dict")
                    elif not torch.equal(decoder_layer.state_dict()[key], value):
                        print(f"Mismatch in key {key}")

                # Store in cache (thread-safe) and notify any waiters
                with cache_condition:
                    layer_weights_cache[layer_idx] = decoder_layer
                    scheduled_layers.discard(layer_idx)
                    # Signal the main thread that layer_idx is now available
                    cache_condition.notify_all()

            except Exception as e:
                print(f"Error prefetching layer {layer_idx}: {e}")
                with cache_condition:
                    scheduled_layers.discard(layer_idx)
                    cache_condition.notify_all()
            finally:
                # Mark this task complete so that join() can proceed
                prefetch_queue.task_done()
    except Exception as e:
        print(f"Prefetch worker failed: {e}")

##############################################################################
# 1. Utility functions to load embeddings, norm, lm_head
##############################################################################
def load_embed_tokens_from_disk(model, layers_dir: str, device: torch.device):
    # 1) Load the saved state dict from disk
    emb_path = layers_dir+"/embed_tokens.pt"
    emb_state = torch.load(emb_path, map_location="cpu")

    # 2) Allocate "real" storage on CPU (turns the meta-allocated layer into CPU-allocated)
    model.model.embed_tokens.to_empty(device="cpu")

    # 3) Load the state dict
    model.model.embed_tokens.load_state_dict(emb_state)

    # 4) Optionally move it onto GPU if desired
    if device.type == "cuda":
        model.model.embed_tokens.to(device)


def load_final_norm_from_disk(model, layers_dir: str, device: torch.device):
    norm_path = layers_dir+"/final_norm.pt"
    norm_state = torch.load(norm_path, map_location="cpu")

    model.model.norm.to_empty(device="cpu")
    model.model.norm.load_state_dict(norm_state)
    if device.type == "cuda":
        model.model.norm.to(device)


def load_lm_head_from_disk(model, layers_dir: str, device: torch.device):
    lm_head_path = layers_dir+"/lm_head.pt"
    lm_head_state = torch.load(lm_head_path, map_location="cpu")

    model.lm_head.to_empty(device="cpu")
    model.lm_head.load_state_dict(lm_head_state)
    if device.type == "cuda":
        model.lm_head.to(device)


##############################################################################
# 2. Layer-by-Layer Offloading Inference Code
##############################################################################
def layer_by_layer_inference(
    model: LlamaForCausalLM,
    input_ids: torch.LongTensor,
    layers_dir: str,
    device: torch.device = torch.device("cuda"),
    prefetch_count: int = 2
) -> torch.Tensor:
    """
    Executes layer-by-layer inference, reliant on the background 
    thread prefetch. The main thread will schedule needed layers and 
    *wait* for them to appear in layer_weights_cache.
    """
    transfer_stream = torch.cuda.Stream(device=device, priority=-1)
    compute_stream = torch.cuda.Stream(device=device, priority=0)

    num_layers = model.config.num_hidden_layers

    # 1) Word embeddings
    hidden_states = model.model.embed_tokens(input_ids.to(device))

    final_norm = model.model.norm
    lm_head = model.lm_head

    for i in range(num_layers):
        stime = time.time()
        print(f"Processing layer {i}...")

        # ---------------------------------------------------------------------
        # 2a) Schedule background CPU prefetch for future layers
        # ---------------------------------------------------------------------
        with cache_condition:
            for j in range(i + 1, min(i + 1 + prefetch_count, num_layers)):
                if j not in layer_weights_cache and j not in scheduled_layers:
                    scheduled_layers.add(j)
                    prefetch_queue.put(j)

        # ---------------------------------------------------------------------
        # 2b) Wait for layer i to appear in the cache (no fallback!)
        # ---------------------------------------------------------------------
        with cache_condition:
            # If not already scheduled, schedule it now
            if i not in layer_weights_cache and i not in scheduled_layers:
                scheduled_layers.add(i)
                prefetch_queue.put(i)

            # Now wait until the worker signals it is loaded
            while i not in layer_weights_cache:
                cache_condition.wait()

            # We can pop it out of the cache now
            decoder_layer = layer_weights_cache.pop(i)
            # decoder_layer = copy.deepcopy(layer_weights_cache.pop(i))

        # ---------------------------------------------------------------------
        # 2c) Move the layer's weights onto the GPU (async transfer)
        # ---------------------------------------------------------------------
        with torch.cuda.stream(transfer_stream):
            decoder_layer.to(device, dtype=torch.bfloat16, non_blocking=False)

        # ---------------------------------------------------------------------
        # 2d) Forward pass with a causal mask
        #     (Without a causal mask, you'll get nonsense!)
        # ---------------------------------------------------------------------
        batch_size, seq_len = hidden_states.shape[:2]
    
        # Build a causal (attention) mask (1,1,seq_len,seq_len) 
        causal_mask = torch.ones((1, 1, seq_len, seq_len), dtype=torch.bool, device=hidden_states.device)
        causal_mask = torch.tril(causal_mask)

        assert causal_mask.shape == (1, 1, seq_len, seq_len), "Causal mask dimensions mismatch!"

        # Build position_ids of shape (batch_size, seq_len)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=hidden_states.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # wait for the decode layer to arrive on GPU
        torch.cuda.synchronize()
        with torch.cuda.stream(compute_stream):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                use_cache=False
            )[0]


        torch.cuda.synchronize()
        # Offload from GPU to free memory
        decoder_layer.to("cpu")
        torch.cuda.synchronize()
        del decoder_layer
        torch.cuda.empty_cache()
        print(f"Layer {i} took {time.time() - stime:.2f} seconds")

    # Final norm & head
    hidden_states = hidden_states.to(device)  # Ensure hidden_states is on GPU
    final_norm = final_norm.to(device)
    hidden_states = final_norm(hidden_states)

    lm_head = lm_head.to(device)
    logits = lm_head(hidden_states)

    # Offload final layers from GPU if desired
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
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

    with torch.inference_mode():
        model.eval()
        for _ in range(max_new_tokens):
            logits = layer_by_layer_inference(
                model,
                input_ids,
                layers_dir=layers_dir,
                device=device,
                prefetch_count=prefetch_count
            )
            with torch.device("cpu"):
                # { changed code }
                next_logit = logits[:, -1, :] / temperature
                next_logit = torch.nan_to_num(next_logit, nan=0.0, posinf=1e4, neginf=-1e4)
                next_logit = torch.clamp(next_logit, min=-50.0, max=50.0)

                top_k = 15
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

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)

if __name__ == "__main__":
    # if not torch.cuda.is_available():
    #     print("Warning: CUDA device not available. This example is intended for a GPU.")
    #     device = torch.device("cpu")
    # else:
    device = torch.device("cuda")

    torch.set_default_dtype(torch.bfloat16)

    # layers_dir = "/mnt/f/7b_model_layers"
    layers_dir = "F:/7b_model_layers"
    print(f"Loading config/tokenizer from: {layers_dir}")

    config = AutoConfig.from_pretrained(layers_dir)
    print(config)
    # If you want a different dtype, do e.g.:
    # config.torch_dtype = torch.bfloat16

    # tokenizer = LlamaTokenizer.from_pretrained("/mnt/e/Llama-3.1-7B/", use_fast=True)
    # tokenizer = LlamaTokenizerFast.from_pretrained(layers_dir)
    tokenizer = AutoTokenizer.from_pretrained(layers_dir)
    # Load tokens from a local JSON file
    # with open("special_tokens.json", "r") as f:
    #     tokens_data = json.load(f)
    
    # tokenizer = AutoTokenizer.from_pretrained(layers_dir, add_special_tokens=True)
    print("Special tokens:", tokenizer.all_special_tokens)
    print("Special tokens count:", len(tokenizer.all_special_tokens))

    # print(tokenizer.vocab_size)
    # # tokenizer.vocab_size = len(tokenizer)
    # print(config.vocab_size)
    # print(len(tokenizer))
    # assert tokenizer.vocab_size == config.vocab_size, "Tokenizer vocabulary size mismatch!"


    # Create a truly empty (meta) model skeleton
    with init_empty_weights():
        # If your local LlamaDecoderLayer doesn’t accept layer_idx,
        # remove the layer_idx references in prefetch_worker().
        model = LlamaForCausalLM(config)

    # Now load the small pieces (embed tokens, final norm, lm head) onto GPU or CPU
    load_embed_tokens_from_disk(model, layers_dir, device=device)
    load_final_norm_from_disk(model, layers_dir, device=device)
    load_lm_head_from_disk(model, layers_dir, device=device)

    # Define the number of layers to prefetch ahead and number of workers
    PREFETCH_COUNT = 12
    NUM_PREFETCH_WORKERS = 12  # Adjust based on your system's capabilities

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
        prompt_text = """You are a helpful AI assistant. Always respond cheerfully.

User: Hello, how are you today?

AI: """
        output_text = generate_tokens_with_temperature(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt_text,
            layers_dir=layers_dir,
            max_new_tokens=5,   # generate more than 1 token to see clearer text
            device=device, 
            temperature=0.6,
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
