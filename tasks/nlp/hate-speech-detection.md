## Task: `hate-speech-detection`

### Name of new TinyLLama

- `tinyllama-hate-speech-detector-v1`

### Status

- [x] Describe and define the task
- [x] List up available datasets for this task
- [x] Define recipes for this task
- [x] Train new TinyLlama for this task
- [x] Evaluate the performance of TinyLlama for this task
- [ ] Release the new TinyLlama for this task to huggingface

### Description

- `hate-speech-detection` is a task of detecting hate speech
- The task is to classify text as hate speech or not
- Enable TinyLlama to respond in `hate-speech` or `not-hate-speech` to text

### Target language

- English

### Datasets

- https://huggingface.co/datasets/tweets_hate_speech_detection

### Recipes

- `recipes/RTX_3060_12GB/hate-speech-detector.yaml`
