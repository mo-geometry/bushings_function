.PHONY: test lint format typecheck clean all

all: lint typecheck test

test:
	pytest --cov=bushings_function --cov-report=term-missing

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

typecheck:
	mypy src/

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name '*.egg-info' -exec rm -rf {} +
