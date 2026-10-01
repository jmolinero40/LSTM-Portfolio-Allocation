# Convenience targets. Run `make help` for the list.
# On Windows without make, every target is one line: copy the command after the tab.
.PHONY: help install test test-fast lint format audit report reproduce smoke clean

PYTHON ?= python
CONFIG ?= configs/walkforward.yaml

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-11s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package with dev dependencies
	$(PYTHON) -m pip install -e ".[dev]"

test:  ## Full test suite (~2 min, includes TensorFlow tests)
	$(PYTHON) -m pytest

test-fast:  ## Tests without TensorFlow (~10 s)
	$(PYTHON) -m pytest -m "not tf and not slow"

lint:  ## Lint and format check
	ruff check src tests scripts
	ruff format --check src tests scripts

format:  ## Apply formatting
	ruff format src tests scripts
	ruff check --fix src tests scripts

audit:  ## Re-derive every number in docs/AUDIT.md from the thesis artefacts (~1 min)
	$(PYTHON) scripts/audit/reproduce_audit.py

report:  ## Tables, figure and README blocks from the committed forecasts (~10 s)
	$(PYTHON) -m lstm_portfolio.report --config $(CONFIG) --update-readme

reproduce:  ## Retrain everything: 9 folds x 3 seeds x 2 networks (hours on CPU)
	$(PYTHON) -m lstm_portfolio.walkforward --config $(CONFIG)
	$(PYTHON) -m lstm_portfolio.report --config $(CONFIG) --update-readme

smoke:  ## Two-minute end-to-end run with tiny budgets (numbers are meaningless)
	$(PYTHON) -m lstm_portfolio.walkforward --config configs/smoke.yaml --no-save-models
	$(PYTHON) -m lstm_portfolio.report --config configs/smoke.yaml --no-figures

clean:  ## Remove caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info src/*.egg-info
