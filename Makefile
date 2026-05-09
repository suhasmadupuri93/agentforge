.PHONY: install test lint run docker-build clean

VENV=.venv
PY=$(VENV)/bin/python
PIP=$(VENV)/bin/pip

install:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

test:
	$(VENV)/bin/pytest -v

lint:
	$(VENV)/bin/ruff check src tests
	$(VENV)/bin/ruff format --check src tests

format:
	$(VENV)/bin/ruff format src tests

run:
	$(VENV)/bin/agentforge

clean:
	rm -rf $(VENV) build dist *.egg-info .pytest_cache .ruff_cache

docker-build:
	docker build -t agentforge:latest -f deployments/Dockerfile .
