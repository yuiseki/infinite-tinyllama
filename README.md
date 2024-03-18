# infinite-tinyllama

## Project milestone

- [x] Train new TinyLlama based on a YAML file in the `recipes` directory
- [ ] Automatically generate new YAML files in the `recipes` directory that archive one of the tasks in the `tasks` directory by specialized TinyLlama
- [ ] Automatically generate new Markdown files in the `tasks` directory by specialized TinyLlama

## Setup

```bash
conda create -n peft
```

```bash
conda activate peft
```

## Training all models

```bash
make
```

## Evaluating all models

```bash
make eval-all
```
