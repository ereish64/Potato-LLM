"""
save_model_layers.py

Use this script to break a Llama model into layer-by-layer checkpoint files 
on disk. The resulting folder can be used by offload_inference.py for 
memory-efficient, layer-by-layer inference.
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaConfig, LlamaTokenizer, LlamaForCausalLM, PreTrainedTokenizerFast, LlamaTokenizerFast


def save_llama_layers(model_name: str, output_dir: str = "model_layers"):
    """
    Load a Llama model from 'model_name' and save its components (embed_tokens, each decoder layer,
    final norm, and lm_head) as individual .pt files inside 'output_dir'.
    """
    # 1) Load the full model and tokenizer
    print(f"Loading model from: {model_name}")
    torch.set_grad_enabled(False)

    model = LlamaForCausalLM.from_pretrained(model_name, device_map={"": "cpu"})
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)

    config = model.config
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    print(f"Special tokens: {tokenizer.special_tokens_map}")

    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # 2) Save model config (so we can reconstruct the skeleton without loading weights)
    config.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Get special token IDs
    special_token_ids = set(tokenizer.all_special_ids)
    print(f"Special token IDs: {special_token_ids}")

    # 3) Save embedding without special tokens
    print("Saving embedding layer...")
    savepath = output_dir+"/embed_tokens.pt"
    # embed_tokens = model.model.embed_tokens.weight.detach().clone()
    torch.save(model.model.embed_tokens.state_dict(), savepath)
    # filtered_embeddings = torch.index_select(
    #     embed_tokens, 0,
    #     torch.tensor(
    #             [idx for idx in range(embed_tokens.size(0)) if idx not in special_token_ids], 
    #             dtype=torch.long
    #         )
    #     )
    # torch.save(filtered_embeddings, os.path.join(output_dir, "embed_tokens.pt"))

    # 4) Save each decoder layer
    print("Saving layer state dictionaries...")
    for i, layer in enumerate(model.model.layers):
        print(f"Saving layer {i}...")
        torch.save(layer.state_dict(), output_dir+f"/layer_{i}.pt")

    # 5) Save final norm & LM head
    print("Saving final norm and lm_head...")
    torch.save(model.model.norm.state_dict(), output_dir+"/final_norm.pt")
    torch.save(model.lm_head.state_dict(), output_dir+"/lm_head.pt")

    print(f"All layers saved to: {output_dir}")

if __name__ == "__main__":
    # save_llama_layers("E:/Llama-2-70B/", "F:/70b_model_layers")
    save_llama_layers("E:/Llama-3.1-8B/", "F:/8b_model_layers")

