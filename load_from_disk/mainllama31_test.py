import os
import queue
import threading
import torch
import time
import math
from torch import nn
from transformers import (
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
    AutoConfig,
)
from accelerate import init_empty_weights

##############################################################################
# Enhanced Rotary Positional Embeddings
##############################################################################
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
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

##############################################################################
# Custom Attention and Decoder Layer Classes
##############################################################################
class PatchedLlamaAttention(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.num_key_value_heads = getattr(config, "num_key_value_heads", self.num_heads)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, hidden_states, attention_mask, position_ids=None, freqs_cis=None):
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if freqs_cis is not None:
            cos, sin = freqs_cis
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if self.num_key_value_groups > 1:
            key_states = torch.repeat_interleave(key_states, self.num_key_value_groups, dim=1)
            value_states = torch.repeat_interleave(value_states, self.num_key_value_groups, dim=1)

        attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn_weights += attention_mask

        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, -1)
        return self.o_proj(attn_output)

class PatchedLlamaDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.self_attn = PatchedLlamaAttention(config, layer_idx=layer_idx)
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, 4 * config.hidden_size),
            nn.GELU(),
            nn.Linear(4 * config.hidden_size, config.hidden_size),
        )

    def forward(self, hidden_states, attention_mask, position_ids=None, freqs_cis=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask, position_ids, freqs_cis)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

##############################################################################
# Prefetch Worker and Model Loading Utilities
##############################################################################
layer_weights_cache = {}
prefetch_queue = queue.Queue()
stop_prefetch = False
cache_lock = threading.Lock()
cache_condition = threading.Condition(cache_lock)
scheduled_layers = set()

def prefetch_worker(layers_dir, config, dtype):
    global stop_prefetch
    try:
        while not stop_prefetch:
            try:
                layer_idx = prefetch_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if layer_idx is None:
                prefetch_queue.task_done()
                break

            with cache_lock:
                if layer_idx in layer_weights_cache:
                    scheduled_layers.discard(layer_idx)
                    prefetch_queue.task_done()
                    continue

            try:
                layer = PatchedLlamaDecoderLayer(config, layer_idx)
                state_path = os.path.join(layers_dir, f"layer_{layer_idx}.pt")
                layer.load_state_dict(torch.load(state_path, map_location="cpu"))
                layer.eval()
                with cache_condition:
                    layer_weights_cache[layer_idx] = layer
                    scheduled_layers.discard(layer_idx)
                    cache_condition.notify_all()
            finally:
                prefetch_queue.task_done()

    except Exception as e:
        print(f"Error in prefetch_worker: {e}")

##############################################################################
# Inference Utilities
##############################################################################
def generate_tokens_with_temperature(
    model,
    tokenizer,
    prompt,
    layers_dir,
    config,
    device=torch.device("cuda"),
    max_new_tokens=10,
    temperature=1.0,
    top_p=0.9,
):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    for _ in range(max_new_tokens):
        logits = model(input_ids).logits[:, -1, :]
        logits = logits / temperature
        sorted_logits, _ = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        top_p_mask = cumulative_probs > top_p
        sorted_logits[top_p_mask] = -float('inf')
        next_token = torch.multinomial(torch.softmax(sorted_logits, dim=-1), num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=-1)
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)

if __name__ == "__main__":
    layers_dir = "E:/Llama-3.1-8B-model-layers"
    config = AutoConfig.from_pretrained(layers_dir)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(layers_dir)
    model = LlamaForCausalLM(config).to(torch.device("cuda"))

    output = generate_tokens_with_temperature(
        model, tokenizer, "Hello, world!", layers_dir, config
    )
    print(output)
