targets = \
	setup \
	output/tinyllama-color-coder-v1/checkpoint-200/README.md \
	output/tinyllama-sql-coder-v1/checkpoint-200/README.md

all: $(targets)

.PHONY: setup lock
setup:
	pip install -U pip
	pip install -Ur requirements.txt
lock:
	pip freeze > requirements.txt.lock

.PHONY: eval-all
eval-all: $(targets)
	python3 src/eval.py recipes/RTX_3060_12GB/color-coder.yaml
	python3 src/eval.py recipes/RTX_3060_12GB/sql-coder.yaml

output/tinyllama-color-coder-v1/checkpoint-200/README.md:
	python3 src/train.py recipes/RTX_3060_12GB/color-coder.yaml

output/tinyllama-sql-coder-v1/checkpoint-200/README.md:
	python3 src/train.py recipes/RTX_3060_12GB/sql-coder.yaml

docker:
	docker build --no-cache -t yuiseki/infinite-tinyllama:latest .

docker-run:
	docker run -it --rm --gpus all -v $(PWD):/app yuiseki/infinite-tinyllama:latest

ollama:
	ollama create elyza-llama2:7b-instruct -f ollama_models/ELYZA/Llama-2/7b-instruct/Modelfile
	ollama create elyza-codellama:7b-instruct -f ollama_models/ELYZA/CodeLlama/7b-instruct/Modelfile
	ollama create rakutenai:7b-instruct -f ollama_models/Rakuten/RakutenAI/7b-instruct/Modelfile
	ollama create rakutenai:7b-chat -f ollama_models/Rakuten/RakutenAI/7b-chat/Modelfile
