"""
save_model_layers.py

Use this script to break a Llama model into layer-by-layer checkpoint files 
on disk. The resulting folder can be used by offload_inference.py for 
memory-efficient, layer-by-layer inference.
"""

import os
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig


def save_llama_layers(model_name: str, output_dir: str = "model_layers"):
    """
    Load a Llama model from 'model_name' and save its components (embed_tokens, each decoder layer,
    final norm, and lm_head) as individual .pt files inside 'output_dir'.
    """
    # 1) Load the full model and tokenizer
    print(f"Loading model from: {model_name}")
    model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    config = model.config

    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # 2) Save model config (so we can reconstruct the skeleton without loading weights)
    config.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)  # so we can load the tokenizer from the same dir

    # 3) Save embedding
    print("Saving embedding layer...")
    torch.save(model.model.embed_tokens.state_dict(), os.path.join(output_dir, "embed_tokens.pt"))

    # 4) Save each decoder layer
    for i, layer in enumerate(model.model.layers):
        print(f"Saving layer {i}...")
        torch.save(layer.state_dict(), os.path.join(output_dir, f"layer_{i}.pt"))

    # 5) Save final norm & LM head
    print("Saving final norm and lm_head...")
    torch.save(model.model.norm.state_dict(), os.path.join(output_dir, "final_norm.pt"))
    torch.save(model.lm_head.state_dict(), os.path.join(output_dir, "lm_head.pt"))

    print(f"All layers saved to: {output_dir}")


if __name__ == "__main__":
    save_llama_layers("/mnt/e/llama", "/mnt/e/model_layers")
