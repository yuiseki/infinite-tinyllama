from datasets import load_dataset

# テスト用データセット
load_dataset("burkelibbey/colors")

# コーディングデータセット（英語）
load_dataset("b-mc2/sql-create-context")
load_dataset("sahil2801/CodeAlpaca-20k")
load_dataset("bigcode/starcoderdata")

# 日本語データセット
load_dataset("shi3z/alpaca_cleaned_ja_json")
load_dataset("izumi-lab/llm-japanese-dataset")
load_dataset("hotchpotch/JQaRA")
load_dataset("kunishou/hh-rlhf-49k-ja")
load_dataset("kunishou/amenokaku-code-instruct")
load_dataset("fujiki/guanaco_ja")
load_dataset("allenai/c4", "ja")
