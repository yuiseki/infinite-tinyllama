import glob

import yaml
from datasets.load import load_dataset

# load_dataset("oscar")
load_dataset("cc100", "en", trust_remote_code=True)
load_dataset("cc100", "ja", trust_remote_code=True)
load_dataset("cerebras/SlimPajama-627B", trust_remote_code=True)
load_dataset("bigcode/starcoderdata", trust_remote_code=True)
load_dataset("Open-Orca/OpenOrca", trust_remote_code=True)
load_dataset("HuggingFaceH4/ultrafeedback_binarized", trust_remote_code=True)
load_dataset("HuggingFaceH4/ultrachat_200k", trust_remote_code=True)
load_dataset("cognitivecomputations/dolphin", trust_remote_code=True)
load_dataset("LDJnr/Capybara", trust_remote_code=True)
load_dataset("ise-uiuc/Magicoder-Evol-Instruct-110K", trust_remote_code=True)
load_dataset("allenai/c4", "en", trust_remote_code=True)
load_dataset("allenai/c4", "ja", trust_remote_code=True)
load_dataset("the_pile", "all", trust_remote_code=True)


# 指定されたファイルパスからyamlファイルを読み込む
def load_yaml(file_path):
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)
    return data


# datasets/huggingface/user_name/dataset_name.yaml にデータセットの概要が記述されている
# datasets/huggingface/**/*.yaml を読み込む
files = glob.glob("datasets/huggingface/**/*.yaml", recursive=True)

# filesのパスからYAMLファイルを読み込む
for file in files:
    data = load_yaml(file)
    print(data["id"])
    print(data["converted_size"])
    # エラー時はcontinueする
    try:
        # MBオーダーかどうか
        is_mb_dataset = data["converted_size"][-2:] == "MB"
        # 10GB以下のデータセットかどうか
        is_lte_10gb_dataset = data["converted_size"][-2:] == "GB" and float(data["converted_size"][:-2]) <= 10
        # MBオーダーか10GB以下のデータセットの場合のみ読み込む
        if is_mb_dataset or is_lte_10gb_dataset:
            dataset = load_dataset(data["id"], trust_remote_code=True)
            print(dataset)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        continue

# wiki40b ja
load_dataset("wiki40b", "ja")
