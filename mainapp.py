"""
offload_inference_all_in_ram.py

Demonstration of layer-by-layer offloading inference with a Llama model whose
weights have been saved as separate layer files (via save_model_layers.py).

In this variation, we first load ALL layer weights from disk into CPU RAM.
Then, for each forward pass, we move layers from CPU RAM -> GPU, run forward,
and then move them back to CPU.
"""

import os
import torch
import torch.nn as nn
import os
import torch
from concurrent.futures import ThreadPoolExecutor

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
# 1. Utility functions for embedding, norm, head
##############################################################################
def load_embed_tokens_from_disk(model, layers_dir: str, device: torch.device):
    emb_path = os.path.join(layers_dir, "embed_tokens.pt")
    emb_state = torch.load(emb_path, map_location="cpu")

    # Step 1) Allocate real storage on CPU
    model.model.embed_tokens.to_empty(device="cpu")

    # Step 2) Load real weights from disk into CPU
    model.model.embed_tokens.load_state_dict(emb_state)

    # Step 3) Optionally move them to GPU
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
# 2. Load all layer weights into CPU RAM
##############################################################################
def load_all_layer_weights_into_ram(layers_dir: str, num_layers: int):
    def load_layer(layer_idx):
        print(f"Loading layer {layer_idx}...")
        state_path = os.path.join(layers_dir, f"layer_{layer_idx}.pt")
        return torch.load(state_path, map_location="cpu")

    with ThreadPoolExecutor(max_workers=12) as executor:
        all_layer_states = list(executor.map(load_layer, range(num_layers)))

    return all_layer_states


##############################################################################
# 3. Layer-by-Layer Offloading Inference, using CPU RAM as intermediate
##############################################################################
def layer_by_layer_inference(
    model: LlamaForCausalLM,
    input_ids: torch.LongTensor,
    all_layer_states,  # List of state_dicts (CPU RAM)
    device: torch.device = torch.device("cuda"),
    prefetch_count: int = 2
) -> torch.Tensor:
    """
    Perform inference by loading layer weights from CPU RAM into GPU, 
    running a forward pass, then offloading back to CPU RAM.
    """

    transfer_stream = torch.cuda.Stream(device=device, priority=-1)
    compute_stream = torch.cuda.Stream(device=device, priority=0)

    num_layers = model.config.num_hidden_layers
    layer_loaded = [False] * num_layers

    # 1) Word embeddings
    hidden_states = model.model.embed_tokens(input_ids.to(device))

    final_norm = model.model.norm
    lm_head = model.lm_head

    for i in range(num_layers):
        print(f"Processing layer {i}...")

        # Create an empty LlamaDecoderLayer for this index
        decoder_layer = LlamaDecoderLayer(model.config, layer_idx=i)

        # Load the layer state_dict (which is already in CPU RAM)
        decoder_layer.load_state_dict(all_layer_states[i], strict=True)

        # Prefetch logic
        for j in range(i, min(i + prefetch_count + 1, num_layers)):
            if not layer_loaded[j] and j == i:
                # Asynchronously load this layer onto GPU
                with torch.cuda.stream(transfer_stream):
                    decoder_layer.to(device, non_blocking=True)
                layer_loaded[j] = True

        # Wait for load to finish
        torch.cuda.current_stream(device).wait_stream(transfer_stream)

        batch_size, seq_len = hidden_states.shape[:2]
        position_ids = torch.arange(
            seq_len, dtype=torch.long, device=hidden_states.device
        ).unsqueeze(0).expand(batch_size, -1)

        # Forward pass
        with torch.cuda.stream(compute_stream):
            hidden_states = decoder_layer(
                hidden_states,
                position_ids=position_ids
            )[0]

        # Wait for compute to finish
        torch.cuda.current_stream(device).wait_stream(compute_stream)

        # Move this layer back to CPU to free GPU memory
        decoder_layer.to("cpu")
        layer_loaded[i] = False
        del decoder_layer  # Cleanup reference

    # Final norm & head
    hidden_states = final_norm(hidden_states)
    logits = lm_head(hidden_states)

    # (Optional) move final norm & head off GPU to free memory
    final_norm.to("cpu")
    lm_head.to("cpu")

    return logits


def generate_tokens_with_temperature(
    model,
    tokenizer,
    prompt,
    all_layer_states,
    max_new_tokens=5,
    device=torch.device("cuda"),
    temperature=1.0
):
    """
    Generate tokens using the layer-by-layer offloading approach.
    Here, we pass in `all_layer_states`, which is a list of state_dicts
    for each layer, already loaded into CPU RAM.
    """
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            logits = layer_by_layer_inference(
                model,
                input_ids,
                all_layer_states=all_layer_states,
                device=device,
                prefetch_count=3
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

    # 1) Load model config and tokenizer
    config = LlamaConfig.from_pretrained(layers_dir)
    tokenizer = AutoTokenizer.from_pretrained(layers_dir)

    # 2) Create an empty (meta) model skeleton
    with init_empty_weights():
        model = LlamaForCausalLM(config)

    # 3) Load embed tokens, final norm, lm head into CPU or GPU
    #    (depending on what you prefer)
    load_embed_tokens_from_disk(model, layers_dir, device=device)
    load_final_norm_from_disk(model, layers_dir, device=device)
    load_lm_head_from_disk(model, layers_dir, device=device)

    # 4) Load all layers into CPU RAM
    all_layer_states = load_all_layer_weights_into_ram(
        layers_dir=layers_dir,
        num_layers=model.config.num_hidden_layers
    )

    # 5) Inference
    prompt_text = "Hello, how are you?"
    output_text = generate_tokens_with_temperature(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt_text,
        all_layer_states=all_layer_states,
        max_new_tokens=5,
        device=device,
        temperature=1.0
    )
    print("Generated text:", output_text)
