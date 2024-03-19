## Task: `text2sql`

### Name of new TinyLLama

- `tinyllama-sql-coder-v1`

### Status

- [x] Describe and define the task
- [x] List up available datasets for this task
- [x] Define recipes for this task
- [x] Train new TinyLlama for this task
- [ ] Evaluate the performance of TinyLlama for this task
- [ ] Release the new TinyLlama for this task to huggingface

### Description

- `text2sql` is a task of converting natural language questions to SQL queries
- The task is to generate a SQL query from a natural language question
- The SQL query is used to retrieve the answer from a database
- Enable TinyLlama to respond in SQL to English questions

### Target language

- English

### Datasets

- https://huggingface.co/datasets/b-mc2/sql-create-context
- https://huggingface.co/datasets/wikisql

### Recipes

- `recipes/RTX_3060_12GB/sql-coder.yaml`
