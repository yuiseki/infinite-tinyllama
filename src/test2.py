import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers

tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

base_model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    torch_dtype=torch.float16,
    load_in_8bit=False,
    device_map="auto",
    trust_remote_code=True
)