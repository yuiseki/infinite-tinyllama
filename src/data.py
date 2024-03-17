from datasets import load_dataset

# テスト用データセット
load_dataset("burkelibbey/colors")

# 日本語データセット
load_dataset("shi3z/alpaca_cleaned_ja_json")
load_dataset("izumi-lab/llm-japanese-dataset")
load_dataset("izumi-lab/llm-japanese-dataset-vanilla")
load_dataset("izumi-lab/wikipedia-ja-20230720")
load_dataset("hotchpotch/JQaRA")
load_dataset("kunishou/oasst1-89k-ja")
load_dataset("kunishou/oasst1-chat-44k-ja")
load_dataset("kunishou/hh-rlhf-49k-ja")
load_dataset("fujiki/guanaco_ja")
load_dataset("llm-jp/oasst1-21k-ja")
load_dataset("llm-jp/hh-rlhf-12k-ja")
load_dataset("llm-jp/databricks-dolly-15k-ja")
load_dataset("allenai/c4", "ja")
load_dataset("mkqa", "ja")

# 要約タスク用データセット（英語）
load_dataset("csebuetnlp/xlsum")
load_dataset("knkarthick/dialogsum")

# 要約タスク用データセット（日本語）
load_dataset("sudy-super/dialogsum-ja")


# コーディングデータセット（英語）
load_dataset("b-mc2/sql-create-context")
load_dataset("sahil2801/CodeAlpaca-20k")
load_dataset("bigcode/starcoderdata")

# コーディングデータセット（日本語）
load_dataset("kunishou/amenokaku-code-instruct")
load_dataset("kunishou/OpenMathInstruct-1-1.8m-ja")
