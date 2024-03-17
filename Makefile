targets = \
	setup \
	output/tinyllama-colorist-v1/checkpoint-200/README.md \
	output/tinyllama-sql-v1/checkpoint-200/README.md

all: $(targets)

.PHONY: setup freeze
setup:
	pip install -Ur requirements.txt

lock:
	pip freeze > requirements.txt.lock

.PHONY: eval-all
eval-all: $(targets)
	python3 src/eval.py config/tinyllama-colorist.yaml
	python3 src/eval.py config/tinyllama-sql.yaml

output/tinyllama-colorist-v1/checkpoint-200/README.md:
	python3 src/train.py config/tinyllama-colorist.yaml

output/tinyllama-sql-v1/checkpoint-200/README.md:
	python3 src/train.py config/tinyllama-sql.yaml

docker:
	docker build --no-cache -t yuiseki/infinite-tinyllama:latest .
