import torch
import torch.nn as nn
from transformers import LlamaForCausalLM, LlamaTokenizer


##############################################################################
# 1. The Layer-By-Layer Offloading Inference Code
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
    Returns the output hidden states (the first element in the module's forward output).
    
    For LlamaDecoderLayer, .forward(...) typically returns a tuple:
    (hidden_states, presents, ...). We extract [0] for just hidden_states.
    """
    with torch.cuda.stream(stream):
        output = module(hidden_states)[0]  # (hidden_states, ...)
    return output

def layer_by_layer_inference(model: LlamaForCausalLM,
                             input_ids: torch.LongTensor,
                             device: torch.device = torch.device("cuda")) -> torch.Tensor:
    """
    Perform a forward pass on the Llama model one block at a time, offloading blocks
    to/from GPU so that at most one or two blocks occupy GPU memory concurrently.
    """

    transfer_stream = torch.cuda.Stream(device=device, priority=-1)
    compute_stream = torch.cuda.Stream(device=device, priority=0)
    
    # Llama model blocks (decoder layers)
    blocks = list(model.model.layers.children())  
    num_layers = len(blocks)
    
    # 1) Compute initial word embeddings on GPU
    model.model.embed_tokens.to(device)
    hidden_states = model.model.embed_tokens(input_ids.to(device))
    
    # Prepare final norm and lm_head
    final_norm = model.model.norm
    lm_head = model.lm_head
    
    # 2) Asynchronously load the *first* block
    async_load_module(blocks[0], device, transfer_stream)
    torch.cuda.current_stream(device).wait_stream(transfer_stream)

    # 3) Main loop over blocks
    for i in range(num_layers):
        # Begin loading the next block while computing the current one
        if i + 1 < num_layers:
            async_load_module(blocks[i+1], device, transfer_stream)
        
        # Forward pass on the i-th block
        hidden_states = forward_module(hidden_states, blocks[i], compute_stream)
        
        # Wait for forward pass to complete
        torch.cuda.current_stream(device).wait_stream(compute_stream)
        
        # Offload this block back to CPU
        blocks[i].to("cpu")
        
        # Wait for the next block to finish loading before next iteration
        if i + 1 < num_layers:
            torch.cuda.current_stream(device).wait_stream(transfer_stream)

    # 4) Final layer norm & LM head
    final_norm.to(device)
    hidden_states = final_norm(hidden_states)
    final_norm.to("cpu")
    
    lm_head.to(device)
    logits = lm_head(hidden_states)
    lm_head.to("cpu")
    
    return logits


##############################################################################
# 2. Multi-Token Generation Using Offloading
##############################################################################
def generate_tokens_with_temperature(
    model, 
    tokenizer, 
    prompt, 
    max_new_tokens=5, 
    device=torch.device("cuda"), 
    temperature=1.0
):
    """
    Generate multiple tokens from the model using temperature-based sampling.
    
    :param model: LlamaForCausalLM
    :param tokenizer: LlamaTokenizer
    :param prompt: Initial text prompt
    :param max_new_tokens: Number of tokens to generate
    :param device: 'cuda' or 'cpu'
    :param temperature: Controls the randomness: higher = more random, lower = more deterministic
    :return: The generated text (prompt + newly generated tokens)
    """
    # Move prompt tokens to the same device as the model
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            # 1) Run inference (returns logits on GPU)
            logits = layer_by_layer_inference(model, input_ids, device=device)
            
            # 2) Focus only on the last token's logits, then apply temperature
            next_logit = logits[:, -1, :] / temperature

            # 3) Convert logits to probabilities
            probs = torch.softmax(next_logit, dim=-1)

            # 4) Sample the next token
            next_token_id = torch.multinomial(probs, num_samples=1)

            # 5) Concatenate to input_ids
            input_ids = torch.cat([input_ids, next_token_id], dim=1)

    # Move back to CPU for decoding
    return tokenizer.decode(input_ids[0].cpu(), skip_special_tokens=True)


if __name__ == "__main__":
    # Make sure you have a GPU available
    if not torch.cuda.is_available():
        print("Warning: CUDA device not available. This example requires a GPU to see memory benefits.")
    
    # Load local Llama model from E:/llama
    model_name = "/mnt/e/llama"
    model = LlamaForCausalLM.from_pretrained(model_name)
    tokenizer = LlamaTokenizer.from_pretrained(model_name)

    # Example: generate 3 tokens from a prompt
    prompt_text = "Hello, how are you?\n\n"
    output_text = generate_tokens_with_temperature(model, tokenizer, prompt_text, max_new_tokens=3)
    print("Generated text:", output_text)
    