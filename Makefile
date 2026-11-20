.PHONY: up down test lint typecheck check

up:
	docker compose up -d

down:
	docker compose down

test:
	uv run pytest -q

lint:
	uv run ruff check .
	uv run ruff format --check .

typecheck:
	uv run mypy --strict src

check: lint typecheck test
