.DEFAULT_GOAL := ready

.PHONY: ready setup check format test smoke build prepare run demo benchmark compare interactive

ready:
	$(MAKE) setup
	$(MAKE) check
	$(MAKE) test
	$(MAKE) build
	$(MAKE) prepare
	$(MAKE) benchmark
	@echo "준비 완료: make interactive 또는 make demo로 실행하세요."

setup:
	uv sync --frozen

check:
	uv run --frozen python scripts/check.py
	uv run --frozen ruff check .
	uv run --frozen ruff format --check .

format:
	uv run --frozen ruff check --fix .
	uv run --frozen ruff format .

test:
	uv run --frozen pytest

smoke:
	uv run --frozen pytest -m smoke

build:
	uv build

prepare:
	uv run --frozen python -m document_system build $(ARGS)

run:
	uv run --frozen document-system $(ARGS)

demo:
	uv run --frozen python -m document_system demo $(ARGS)

benchmark:
	uv run --frozen python -m document_system benchmark $(ARGS)

compare:
	uv run --frozen python -m document_system compare $(ARGS)

interactive:
	uv run --frozen python -m document_system interactive $(ARGS)
