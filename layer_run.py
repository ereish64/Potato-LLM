"""
offload_inference.py

Demonstration of layer-by-layer offloading inference with a Llama model whose
weights have been saved as separate layer files (via save_model_layers.py).
"""

import os
import torch
import torch.nn as nn

# Pull in Llama classes
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer


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
    """
    Load the embedding weights from 'embed_tokens.pt' into model.model.embed_tokens.
    """
    emb_path = os.path.join(layers_dir, "embed_tokens.pt")
    emb_state = torch.load(emb_path, map_location="cpu")
    model.model.embed_tokens.load_state_dict(emb_state)
    model.model.embed_tokens.to(device)


def load_final_norm_from_disk(model, layers_dir: str, device: torch.device):
    """
    Load the final norm weights from 'final_norm.pt' into model.model.norm.
    """
    norm_path = os.path.join(layers_dir, "final_norm.pt")
    norm_state = torch.load(norm_path, map_location="cpu")
    model.model.norm.load_state_dict(norm_state)
    model.model.norm.to(device)


def load_lm_head_from_disk(model, layers_dir: str, device: torch.device):
    """
    Load the LM head (output projection) weights from 'lm_head.pt' into model.lm_head.
    """
    lm_head_path = os.path.join(layers_dir, "lm_head.pt")
    lm_head_state = torch.load(lm_head_path, map_location="cpu")
    model.lm_head.load_state_dict(lm_head_state)
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


def layer_by_layer_inference(model: LlamaForCausalLM,
                             input_ids: torch.LongTensor,
                             layers_dir: str,
                             device: torch.device = torch.device("cuda"),
                             prefetch_count: int = 3) -> torch.Tensor:
    """
    Perform a forward pass on the Llama model one block at a time, offloading blocks
    to/from GPU so that at most a handful of blocks occupy GPU memory concurrently.
    
    :param model: A "skeleton" LlamaForCausalLM model (no real weights in the layers).
    :param input_ids: Token IDs of the current input prompt
    :param layers_dir: Path to the directory with layer_{i}.pt files
    :param device: 'cuda' by default
    :param prefetch_count: Number of layers to prefetch (load ahead)
    :return: The logits from the final output layer
    """

    transfer_stream = torch.cuda.Stream(device=device, priority=-1)  # lower priority for loading
    compute_stream = torch.cuda.Stream(device=device, priority=0)    # main compute stream

    # Number of decoder layers from the config
    num_layers = model.config.num_hidden_layers

    # Keep track of whether each layer is already loaded on GPU
    layer_loaded = [False] * num_layers

    # -----------------------------
    # 1) Compute initial word embeddings on GPU
    # -----------------------------
    # We already loaded embed_tokens onto GPU
    hidden_states = model.model.embed_tokens(input_ids.to(device))

    # We'll also load final_norm and lm_head at the end, so get references:
    final_norm = model.model.norm
    lm_head = model.lm_head

    # -----------------------------
    # 2) Main loop over blocks
    # -----------------------------
    for i in range(num_layers):
        # (a) Build an empty LlamaDecoderLayer for layer i, then load its weights from disk
        decoder_layer = LlamaDecoderLayer(model.config)
        load_layer_weights_from_disk(decoder_layer, i, layers_dir)
        
        # (b) Prefetch up to 'prefetch_count' future layers
        #     so we don't wait on I/O when we get there.
        #     (In practice, you might do partial loads in parallel.)
        for j in range(i, min(i + prefetch_count + 1, num_layers)):
            if not layer_loaded[j] and j == i:
                async_load_module(decoder_layer, device, transfer_stream)
                layer_loaded[j] = True
        
        # (c) Wait for layer i to finish loading before compute
        torch.cuda.current_stream(device).wait_stream(transfer_stream)

        # (d) Forward pass on layer i
        hidden_states = forward_module(hidden_states, decoder_layer, compute_stream)

        # (e) Wait for forward pass to complete
        torch.cuda.current_stream(device).wait_stream(compute_stream)

        # (f) Offload the i-th block to free GPU memory
        #     (In practice, you might call `.to("cpu")` or `del decoder_layer`.)
        decoder_layer.to("cpu")
        layer_loaded[i] = False
        del decoder_layer  # remove references, let Python GC reclaim memory

    # -----------------------------
    # 3) Final layer norm & LM head
    # -----------------------------
    # We already loaded final_norm and lm_head onto GPU if needed
    hidden_states = final_norm(hidden_states)
    logits = lm_head(hidden_states)

    # Move final_norm & lm_head back to CPU if you want
    final_norm.to("cpu")
    lm_head.to("cpu")

    return logits


##############################################################################
# 3. Multi-Token Generation Using Offloading
##############################################################################
def generate_tokens_with_temperature(
    model, 
    tokenizer, 
    prompt, 
    layers_dir,
    max_new_tokens=5, 
    device=torch.device("cuda"), 
    temperature=1.0
):
    """
    Generate multiple tokens from the model using temperature-based sampling.
    This version loads each layer from disk only when needed.
    
    :param model: A "skeleton" LlamaForCausalLM (layers not pre-loaded)
    :param tokenizer: LlamaTokenizer
    :param prompt: Initial text prompt
    :param layers_dir: Directory that contains layer_*.pt, embed_tokens.pt, etc.
    :param max_new_tokens: Number of tokens to generate
    :param device: 'cuda' or 'cpu'
    :param temperature: Higher = more random, lower = more deterministic
    :return: The generated text (prompt + newly generated tokens)
    """
    # Prepare prompt tokens
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            # 1) Run inference (returns logits on GPU)
            logits = layer_by_layer_inference(
                model, 
                input_ids, 
                layers_dir=layers_dir,
                device=device, 
                prefetch_count=3
            )

            # 2) Focus on the last token's logits, then apply temperature
            next_logit = logits[:, -1, :] / temperature

            # 3) Convert logits to probabilities
            probs = torch.softmax(next_logit, dim=-1)

            # 4) Sample the next token
            next_token_id = torch.multinomial(probs, num_samples=1)

            # 5) Append the new token
            input_ids = torch.cat([input_ids, next_token_id.cpu()], dim=1)

    # Decode final output
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)


##############################################################################
# 4. Main Example
##############################################################################
if __name__ == "__main__":
    # You can still run on CPU if no CUDA is available
    if not torch.cuda.is_available():
        print("Warning: CUDA device not available. This example is intended for a GPU.")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    # Suppose you used save_model_layers.py to chunk a model into E:/llama_chunked
    layers_dir = "E:/llama_chunked"

    # -----------------------------------------------
    # 1) Load the config & tokenizer
    #    (The model layers we will load on-demand)
    # -----------------------------------------------
    print(f"Loading config/tokenizer from: {layers_dir}")
    config = LlamaConfig.from_pretrained(layers_dir)
    tokenizer = LlamaTokenizer.from_pretrained(layers_dir)

    # Create a "skeleton" LlamaForCausalLM using the config
    # (This has the correct architecture, but no layer weights yet.)
    model = LlamaForCausalLM(config)

    # -----------------------------------------------
    # 2) Load embedding, final norm, LM head
    #    (only once; keep them on GPU or CPU)
    # -----------------------------------------------
    load_embed_tokens_from_disk(model, layers_dir, device=device)
    load_final_norm_from_disk(model, layers_dir, device=device)
    load_lm_head_from_disk(model, layers_dir, device=device)

    # -----------------------------------------------
    # 3) Generate
    # -----------------------------------------------
    prompt_text = "Hello, how are you?"
    output_text = generate_tokens_with_temperature(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt_text,
        layers_dir=layers_dir,
        max_new_tokens=5, 
        device=device, 
        temperature=1.0
    )
    print("Generated text:", output_text)
