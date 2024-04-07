import os
import sys

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

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
# Load the tokenizer for the specified model.
tokenizer = AutoTokenizer.from_pretrained(model_id)
# Set the padding token.
# NOTE: これやるならmodel.resize_token_embeddingsが必要
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})
# Set the padding token to be the same as the end of sentence token.
# NOTE: tokenizer.add_special_tokensやるならこれは不要
tokenizer.pad_token = tokenizer.eos_token
# Load the model.
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    load_in_8bit=False,
    device_map="auto",
    trust_remote_code=True,
)
# NOTE: tokenizer.add_special_tokensやるならこれが必要
# Resize the token embeddings to match the tokenizer.
# base_model.resize_token_embeddings(len(tokenizer))

#
# Merge model
#
checkpoints_dir_path = os.path.join(
    train_config["output_base_dir"],
    train_config["model_name"],
)

# checkpoints_dir内のディレクトリ一覧を取得
# 一番数字が大きいものが最後に保存されたモデル
last_checkpoint_dir_name = ""
last_checkpoint_dir_num = -1

for dir_name in os.listdir(checkpoints_dir_path):
    if not dir_name.startswith("checkpoint-"):
        continue
    dir_num = int(dir_name.split("-")[1])
    if dir_num > last_checkpoint_dir_num:
        last_checkpoint_dir_num = dir_num
        last_checkpoint_dir_name = dir_name

model_path = os.path.join(checkpoints_dir_path, last_checkpoint_dir_name)

peft_model = PeftModel.from_pretrained(base_model, model_path, from_transformers=True, device_map="auto")

merged_model = peft_model.merge_and_unload()

merged_model.push_to_hub(train_config["model_name"])
tokenizer.push_to_hub(train_config["model_name"])
