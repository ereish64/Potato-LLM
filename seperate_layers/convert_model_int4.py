import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# Replace this with your model name or path (which may be saved in bfloat16)
model_name_or_path = "E:/Llama-3.1-8B-INT4"

# Create a BitsAndBytesConfig for int4 quantization
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,  # or torch.float16
    bnb_4bit_quant_type="nf4"              # can also be "fp4"
)

# Load the bfloat16 model and quantize it to int4
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    quantization_config=quant_config,
    device_map="auto",  # automatically allocates layers across your GPUs/CPU
    trust_remote_code=True,  # if your model repo has custom code
)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

# Optionally, save the 4-bit model to a local folder
quantized_model_path = "E:/Llama-3.1-8B-INT4"
model.save_pretrained(quantized_model_path)
tokenizer.save_pretrained(quantized_model_path)
