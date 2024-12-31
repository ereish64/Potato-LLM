"""
offload_inference.py

Demonstration of layer-by-layer offloading inference with a Llama model whose
weights have been saved as separate layer files (via save_model_layers.py).
"""

import os
import torch
import torch.nn as nn

# Pull in Llama classes
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig, AutoTokenizer

# Additional import
from accelerate import init_empty_weights

# Pull in Llama classes
from transformers import (
    LlamaForCausalLM, 
    LlamaTokenizer, 
    LlamaConfig, 
    AutoTokenizer,
)
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
    prefetch_count: int = 3
) -> torch.Tensor:
    """
    Exactly the same code you have now, just unchanged.
    ...
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
        # decoder_layer = LlamaDecoderLayer(model.config)
        decoder_layer = LlamaDecoderLayer(model.config, layer_idx=i)
        load_layer_weights_from_disk(decoder_layer, i, layers_dir)

        for j in range(i, min(i + prefetch_count + 1, num_layers)):
            if not layer_loaded[j] and j == i:
                # Asynchronously load this layer onto GPU
                with torch.cuda.stream(transfer_stream):
                    decoder_layer.to(device, non_blocking=True)
                layer_loaded[j] = True

        # Wait for load
        torch.cuda.current_stream(device).wait_stream(transfer_stream)

        batch_size, seq_len = hidden_states.shape[:2]
        position_ids = torch.arange(
            seq_len, dtype=torch.long, device=hidden_states.device
        ).unsqueeze(0).expand(batch_size, -1)
        # Forward
        with torch.cuda.stream(compute_stream):
            # hidden_states = decoder_layer(hidden_states)[0]
            hidden_states = decoder_layer(
                hidden_states,
                position_ids=position_ids
            )[0]

        # Wait for compute
        torch.cuda.current_stream(device).wait_stream(compute_stream)

        # Offload
        decoder_layer.to("cpu")
        layer_loaded[i] = False
        del decoder_layer

    # Final norm & head
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
    temperature=1.0
):
    """
    Same as your code, unchanged.
    """
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            logits = layer_by_layer_inference(
                model, 
                input_ids, 
                layers_dir=layers_dir,
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

    config = LlamaConfig.from_pretrained(layers_dir)
    tokenizer = AutoTokenizer.from_pretrained(layers_dir)

    # Create a truly empty (meta) model skeleton
    with init_empty_weights():
        model = LlamaForCausalLM(config)

    # Now load the small pieces (embed tokens, final norm, lm head) onto GPU or CPU
    load_embed_tokens_from_disk(model, layers_dir, device=device)
    load_final_norm_from_disk(model, layers_dir, device=device)
    load_lm_head_from_disk(model, layers_dir, device=device)

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