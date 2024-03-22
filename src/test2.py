import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


# Load the tokenizer for the specified model.
tokenizer = AutoTokenizer.from_pretrained(model_id)
# Set the padding token to be the same as the end of sentence token.
tokenizer.pad_token = tokenizer.eos_token
# Define the quantization configuration for memory-efficient training.
bnb_config = BitsAndBytesConfig(
    # Load the model weights in 4-bit quantized format.
    load_in_4bit=True,
    # Specify the quantization type to use for 4-bit quantization.
    bnb_4bit_quant_type="nf4",
    # Specify the data type to use for computations during training.
    bnb_4bit_compute_dtype="float16",
    # Specify whether to use double quantization for 4-bit quantization.
    bnb_4bit_use_double_quant=True,
)
# Load the model from the specified model ID and apply the quantization configuration.
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")
# Disable cache to improve training speed.
model.config.use_cache = False
# Set the temperature for pretraining to 1.
model.config.pretraining_tp = 1
