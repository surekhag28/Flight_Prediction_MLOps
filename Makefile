# Makefile

VENV=.venv
PYTHON=$(VENV)/bin/python

.PHONY: help install dev run test lint clean

help:
	@echo "Available commands:"
	@echo "  make install   Install project dependencies"
	@echo "  make dev       Install all dev dependencies"
	@echo "  make run       Run application"
	@echo "  make test      Run tests"
	@echo "  make lint      Run linter"
	@echo "  make clean     Remove virtual environment"

install:
	uv sync

dev:
	uv sync --all-groups

run:
	uv run python main.py

test:
	uv run pytest

lint:
	uv run ruff check .

clean:
	rm -rf .venv