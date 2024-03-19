## Task: `text2onomatopoeia`

### Name of new TinyLLama

- `tinyllama-onomatopoeia-ja-v1`

### Status

- [x] Describe and define the task
- [x] List up available datasets for this task
- [x] Define recipes for this task
- [x] Train new TinyLlama for this task
- [x] Evaluate the performance of TinyLlama for this task
- [ ] Release the new TinyLlama for this task to huggingface

### Description

- `text2onomatopoeia` is a task of converting natural language text to onomatopoeia
- The task is to generate an onomatopoeia from a natural language text
- The onomatopoeia is used to represent the text as a sound
- Enable TinyLlama to respond in Japanese onomatopoeia to Japanese text

### Target language

- Japanese

### Datasets

- https://huggingface.co/datasets/yuiseki/onomatopoeia-ja-flat

### Recipes

- `recipes/RTX_3060_12GB/onomatopoeia-ja.yaml`
