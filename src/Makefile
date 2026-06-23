export UV_CACHE_DIR := /goinfre/amostash/uv-cache
export HF_HOME := /goinfre/amostash/huggingface
export HUGGINGFACE_HUB_CACHE := /goinfre/amostash/huggingface/hub
export HF_DATASETS_CACHE := /goinfre/amostash/huggingface/datasets

install:
	mkdir -p $(UV_CACHE_DIR)
	mkdir -p $(HF_HOME)

	uv sync

run:
	uv run python -m src

lint:
	uv run flake8 .
	uv run mypy . \
		--warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -name .pytest_cache -exec rm -rf {} +
	find . -name .ruff_cache -exec rm -rf {} +
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete

.PHONY: install run lint clean
