targets = \
	setup \
	output/tinyllama-color-coder-v1/checkpoint-200/README.md \
	output/tinyllama-sql-coder-v1/checkpoint-200/README.md

all: $(targets)

.PHONY: setup
setup:
	pip install -r requirements.txt

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
