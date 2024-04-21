# infinite-tinyllama

## Overview

TinyLlama は Raspberry Pi 4 8GB でも実行できる、軽量かつ小規模だが一定の性能を発揮する LLM である。

TinyLlama を様々なオープンライセンスのデータセットで Lora ファインチューニングし、様々な言語や様々な分野に特化させて、それらを OSS として公開したい。

TinyLlama の Lora ファインチューニングはデータセットや Lora パラメーターのチューニングによっては 1GPU 12GB VRAM でも可能だが、データセットが巨大な場合やパラメーターによっては 24GB VRAM でも足りない。

VRAM 24GB で日本語データセットでファインチューニングしようとして VRAM 不足で失敗しており、既に困っている。

今回は与えられた GPU と VRAM 容量の限りで並列に TinyLlama の Lora ファインチューニングを実行することで、可能な限り多くの特化型 TinyLlama を生み出すことを目指す。

また、ローカルで動かしやすいように GGUF モデルとしても公開したい。GGUF に変換するためにも多くの VRAM が必要となる。

さらなる野望としては、TinyLlama が自律的に Huggingface のデータセットを探索しデータ構造を把握してファインチューニングを実行し新たな TinyLlama を生み出す「autonomous-infinite-tinyllama」を開発することや、TinyLlama のソースコードやアーキテクチャを参考にしつつフルスクラッチで LLM を実装して事前学習から着手することで、「WTFPL」というライセンスで「WTFPLLM」をリリースすることを目指したい。

## Project milestone

- **人力フェーズ**
  - 🔨 `datasets/huggingface/` 以下に、データセットの情報を機械可読な形式で丁寧に記述した YAML ファイルを追加していく
    - 例:
      - `datasets/huggingface/burkelibbey/colors.yaml`
  - 🔨 `tasks/` 以下に、「こういう TinyLlama が作りたい、このデータセットが使えそう」という Markdown ファイルを追加していく
    - 例:
      - `tasks/text-generation/text2rgbhex.md`
  - 🔨 `recipes/` 以下に、データセットの YAML ファイルと Markdown ファイルの説明から、TinyLlama を学習するための YAML ファイルを追加していく
    - 例:
      - `recipes/RTX_3060_12GB/color-coder.yaml`
- **シンギュラリティフェーズ**
  - 🤔 `datasets/huggingface/` 以下に、新たなファイルを生成して追加する TinyLlama を作る
  - 🤔 `tasks/` 以下に、新たなファイルを生成して追加する TinyLlama を作る
  - 🤔 `recipes/` 以下に、新たなファイルを生成して追加する TinyLlama を作る
  - 🤔 `recipes/` 以下にあるファイルで TinyLlama の学習を実行して、失敗したら修正して再実行する TinyLlama を作る

## Setup

```bash
conda create -n peft python=3.10
```

```bash
conda activate peft
```

```bash
conda install -c nvidia cuda-toolkit=12.4
```

## Training all models

```bash
make
```

## Evaluating all models

```bash
make eval-all
```
