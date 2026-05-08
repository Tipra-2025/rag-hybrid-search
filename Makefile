.PHONY: help install test lint typecheck check fmt ingest ask serve docker clean

help:
	@echo "install     pip install -e .[dev]"
	@echo "test        pytest with coverage"
	@echo "lint        ruff check"
	@echo "typecheck   mypy --strict"
	@echo "check       lint + typecheck + test"
	@echo "fmt         ruff format + ruff check --fix"
	@echo "serve       rag serve"
	@echo "ingest      rag ingest --path docs/"
	@echo "ask         rag ask --question '...'"
	@echo "docker      build the docker image"
	@echo "clean       caches + index"

install:
	python3 -m pip install -e ".[dev]"

test:
	pytest --cov

lint:
	ruff check src tests

typecheck:
	mypy src

check: lint typecheck test

fmt:
	ruff format src tests
	ruff check src tests --fix

serve:
	rag serve

ingest:
	rag ingest --path docs/

ask:
	rag ask --question "What is RRF fusion?"

docker:
	docker build -t rag-hybrid-search:local .

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage coverage.xml htmlcov build dist *.egg-info
	rm -rf .rag-index/
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
