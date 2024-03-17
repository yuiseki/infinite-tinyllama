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
    # FIXME: MB単位のやつだけload_datasetで準備しておく
    if data['converted_size'][-2:] == 'MB':
        dataset = load_dataset(data['id'])
        print(dataset)
