.PHONY: lint format format-check typecheck test compile-pipelines docker-build all

lint:
	ruff check .

format:
	ruff format .

format-check:
	ruff format --check .

typecheck:
	mypy server/ --ignore-missing-imports
	mypy server-https/ --ignore-missing-imports

test:
	pytest tests/ -v --tb=short

compile-pipelines:
	cd pipelines && python3 kubeflow-pipeline.py
	cd pipelines && python3 incremental-pipeline.py

docker-build:
	docker build -t docs-agent-ws:dev ./server
	docker build -t docs-agent-https:dev ./server-https

all: lint format-check typecheck test compile-pipelines
