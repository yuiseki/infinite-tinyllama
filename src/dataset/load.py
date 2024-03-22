from datasets import load_dataset
import glob
import yaml

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
    print(data['id'])
    print(data['converted_size'])
    # エラー時はcontinueする
    try:
        # MBオーダーかどうか
        is_mb_dataset = data['converted_size'][-2:] == 'MB'
        # 10GB以下のデータセットかどうか
        is_lte_10gb_dataset = data['converted_size'][-2:] == 'GB' and float(data['converted_size'][:-2]) <= 10
        # MBオーダーか10GB以下のデータセットの場合のみ読み込む
        if is_mb_dataset or is_lte_10gb_dataset:
                dataset = load_dataset(data['id'])
                print(dataset)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        continue

# wiki40b ja
load_dataset('wiki40b', 'ja')
