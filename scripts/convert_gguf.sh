#!/bin/bash

MODEL_NAME=$1
echo $MODEL_NAME

git lfs install

export CUDA_VISIBLE_DEVICES=-1

REPO_URL=https://huggingface.co/yuiseki/$MODEL_NAME
echo $REPO_URL

status_code=$(curl --write-out %{http_code} --silent --output /dev/null $REPO_URL)

if [[ "$status_code" -ne 200 ]] ; then
  echo "!!! Model: ${MODEL_NAME} NOT FOUUND !!!"
  exit 1
fi

REPO_PATH=git@hf.co:yuiseki/$MODEL_NAME.git


git clone $REPO_PATH
python3 ~/llama.cpp/convert.py $MODEL_NAME
~/llama.cpp/quantize $MODEL_NAME/ggml-model-f16.gguf $MODEL_NAME/$MODEL_NAME-Q4_K_M.gguf Q4_K_M
