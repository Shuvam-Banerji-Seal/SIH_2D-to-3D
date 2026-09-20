.DEFAULT_GOAL := help
UV ?= uv
RUN := $(UV) run

.PHONY: help sync dev test lint format typecheck pre-commit clean run doctor lock

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

sync: ## Install runtime dependencies from uv.lock
	$(UV) sync --locked

dev: ## Install runtime + dev dependencies (default group) and git hooks
	$(UV) sync
	$(RUN) pre-commit install

all-extras: ## Install every optional backend (ai, sfm, mesh, geo, api)
	$(UV) sync --all-extras

lock: ## Regenerate uv.lock
	$(UV) lock

test: ## Run the test suite
	$(RUN) pytest

lint: ## Run ruff checks
	$(RUN) ruff check src tests

format: ## Auto-format with ruff
	$(RUN) ruff check --fix src tests
	$(RUN) ruff format src tests

typecheck: ## Run mypy (optional dependency)
	$(UV) run --with mypy mypy src/drone3d || true

doctor: ## Check environment and optional reconstruction backends
	$(RUN) drone3d doctor

run: ## Run the default pipeline against configs/default.yaml
	$(RUN) drone3d reconstruct --config configs/default.yaml

clean: ## Remove caches and build artifacts
	rm -rf build dist .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
	find src tests -type d -name '__pycache__' -prune -exec rm -rf {} +
