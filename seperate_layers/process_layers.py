"""
save_model_layers.py

Use this script to break a Llama model into layer-by-layer checkpoint files 
on disk. The resulting folder can be used by offload_inference.py for 
memory-efficient, layer-by-layer inference.
"""

import os
import torch
import math
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    """
    Enhanced Rotary Position Embedding for Llama3.
    """
    if position_ids is not None:
        cos = cos[position_ids].unsqueeze(1)
        sin = sin[position_ids].unsqueeze(1)
    else:
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
    
    q1, q2 = q[..., ::2], q[..., 1::2]
    k1, k2 = k[..., ::2], k[..., 1::2]
    cos = cos[..., :q1.shape[-1]]
    sin = sin[..., :q1.shape[-1]]
    q_rot = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)
    k_rot = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1)
    return q_rot, k_rot

class PatchedLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx)
        self.self_attn = PatchedLlamaAttention(config, layer_idx)

    def forward(self, hidden_states, attention_mask, position_ids=None, freqs_cis=None, past_key_value=None,
                use_cache=False, output_attentions=False):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_output, _, _ = self.self_attn(hidden_states, attention_mask, position_ids, freqs_cis, past_key_value,
                                           output_attentions, use_cache)
        hidden_states = residual + attn_output
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states, None, None
    
class PatchedLlamaAttention(LlamaAttention):
    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx)
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_key_value_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        self.num_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

    def forward(self, hidden_states, attention_mask, position_ids=None, freqs_cis=None, past_key_value=None,
                output_attentions=False, use_cache=False):
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if freqs_cis is not None:
            cos, sin = freqs_cis
            if position_ids is None:
                position_ids = torch.arange(q_len, device=hidden_states.device).unsqueeze(0).expand(bsz, -1)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if self.num_key_value_groups > 1:
            key_states = torch.repeat_interleave(key_states, self.num_key_value_groups, dim=1)
            value_states = torch.repeat_interleave(value_states, self.num_key_value_groups, dim=1)

        attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn_weights += attention_mask
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, self.hidden_size)
        return self.o_proj(attn_output), None, None

def save_llama_layers(model_name: str, output_dir: str = "model_layers"):
    """
    Load a Llama model from 'model_name' and save its components (embed_tokens, each decoder layer,
    final norm, and lm_head) as individual .pt files inside 'output_dir'.
    """
    # 1) Load the full model and tokenizer
    print(f"Loading model from: {model_name}")
    torch.set_grad_enabled(False)
    model = LlamaForCausalLM.from_pretrained(model_name, device_map="cpu", torch_dtype=torch.bfloat16)
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

    # 4) Save each decoder layer
    print("Saving layer state dictionaries...")
    for i, layer in enumerate(model.model.layers):
        print(f"Saving layer {i}...")
        decoder_layer = PatchedLlamaDecoderLayer(config, layer_idx=i)

        # loaded_state = torch.load(layer.state, map_location="cpu")
        decoder_layer.load_state_dict(layer.state_dict())
        # scripted_layer = torch.jit.script(layer)
        # scripted_layer.save(savepath)
        torch.save(layer, output_dir+f"/layer_{i}.pt")

    # 5) Save final norm & LM head
    print("Saving final norm and lm_head...")
    torch.save(model.model.norm.state_dict(), output_dir+"/final_norm.pt")
    torch.save(model.lm_head.state_dict(), output_dir+"/lm_head.pt")

    print(f"All layers saved to: {output_dir}")

if __name__ == "__main__":
    # save_llama_layers("E:/Llama-2-70B/", "F:/70b_model_layers")
    save_llama_layers("E:/Llama-3.1-70B/", "E:/Llama-3.1-70B-model-layers")

