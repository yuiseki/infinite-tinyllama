## Task: `text2rgbhex`

### Name of new TinyLLama

- `tinyllama-color-coder-v1`

### Status

- [x] Describe and define the task
- [x] List up available datasets for this task
- [x] Define recipes for this task
- [x] Train new TinyLlama for this task
- [x] Evaluate the performance of TinyLlama for this task
- [ ] Release the new TinyLlama for this task to huggingface

### Description

- `text2rgbhex` is a task of converting natural language text to RGB hex color codes
- The task is to generate a RGB hex color code from a natural language text
- The RGB hex color code is used to represent the text as a color
- Enable TinyLlama to respond in RGB hex color codes to English text

### Target language

- English

### Datasets

- https://huggingface.co/datasets/burkelibbey/colors

### Recipes

- `recipes/RTX_3060_12GB/color-coder.yaml`
