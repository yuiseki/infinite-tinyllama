## Task: `fake-news-detection`

### Name of new TinyLLama

- `tinyllama-fake-news-detector-v1`

### Status

- [x] Describe and define the task
- [x] List up available datasets for this task
- [x] Define recipes for this task
- [x] Train new TinyLlama for this task
- [x] Evaluate the performance of TinyLlama for this task
- [ ] Release the new TinyLlama for this task to huggingface

### Description

- `fake-news-detection` is a task of detecting fake news
- The task is to classify news articles as fake or real
- Enable TinyLlama to respond in `fake` or `real` to news articles

### Target language

- English

### Datasets

- https://huggingface.co/datasets/mrm8488/fake-news

### Recipes

- `recipes/RTX_3060_12GB/fake-news-detector.yaml`
