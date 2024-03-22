import os
import re
import sys

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"


# 指定されたファイルパスからyamlファイルを読み込む
def load_yaml(file_path):
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)
    return data


# 実行時の1番目の引数をload_yamlに渡す
filepath = sys.argv[1]
train_config = load_yaml(filepath)

#
# Original model
#
model_id = train_config["base_model_id"]
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    load_in_8bit=False,
    device_map="auto",
    trust_remote_code=True,
)
# Load the tokenizer for the specified model.
tokenizer = AutoTokenizer.from_pretrained(model_id)
# Set the padding token to be the same as the end of sentence token.
tokenizer.pad_token = tokenizer.eos_token

#
# Merge model
#
model_path = os.path.join(
    train_config["output_base_dir"],
    train_config["model_name"],
    f"checkpoint-{train_config['train_max_steps']}",
)
peft_model = PeftModel.from_pretrained(
    base_model, model_path, from_transformers=True, device_map="auto"
)

merged_model = peft_model.merge_and_unload()

merged_model.push_to_hub(train_config["model_name"])
