"""
save_model_layers_multiprocessing.py

Use this script to break a Llama model into layer-by-layer checkpoint files 
on disk. The resulting folder can be used by offload_inference.py for 
memory-efficient, layer-by-layer inference.

This version uses multiprocessing to speed up I/O-bound saves of each layer.
"""

import os
import torch
import multiprocessing as mp
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaConfig


def _save_state_dict(args):
    """
    Helper function for parallel writing.
    Expects a tuple of (save_path, state_dict).
    """
    path, state_dict = args
    torch.save(state_dict, path)


def save_llama_layers(model_name: str, output_dir: str = "model_layers"):
    """
    Load a Llama model from 'model_name' and save its components (embed_tokens, each decoder layer,
    final norm, and lm_head) as individual .pt files inside 'output_dir', in parallel.
    """
    # 1) Load the full model and tokenizer (done in main process)
    print(f"Loading model from: {model_name}")

    config = LlamaConfig.from_pretrained("/mnt/e/Llama-3.1-70B/config.json")
    print(config)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.bfloat16, 
        device_map={"": "cpu"}
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = model.config

    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # 2) Save model config & tokenizer (so we can reconstruct skeleton + tokenizer)
    config.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # 3) Build the list of saving tasks (to run in parallel)
    save_tasks = []

    # Embedding
    save_tasks.append(
        (os.path.join(output_dir, "embed_tokens.pt"), model.model.embed_tokens.state_dict())
    )

    # Decoder layers
    for i, layer in enumerate(model.model.layers):
        save_tasks.append(
            (os.path.join(output_dir, f"layer_{i}.pt"), layer.state_dict())
        )

    # Final norm + LM head
    save_tasks.append(
        (os.path.join(output_dir, "final_norm.pt"), model.model.norm.state_dict())
    )
    save_tasks.append(
        (os.path.join(output_dir, "lm_head.pt"), model.lm_head.state_dict())
    )

    # 4) Use multiprocessing to save all state_dicts in parallel
    print("Saving layers in parallel. This may take a few minutes...")
    num_cpus = mp.cpu_count()
    print(f"Using up to {num_cpus} CPU cores for parallel saves.")

    with mp.Pool(processes=num_cpus) as pool:
        pool.map(_save_state_dict, save_tasks)

    print(f"All layers saved to: {output_dir}")


if __name__ == "__main__":
    save_llama_layers("/mnt/e/Llama-3.1-70B/", "/mnt/e/model_layers")
