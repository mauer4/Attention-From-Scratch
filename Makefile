.PHONY: setup setup-vast download-assets fetch-olmo allenai-help bootstrap-log

PYTHON ?= python3
VENV ?= .venv-olmo2

setup:
	@bash setup/bootstrap_host.sh

setup-vast:
	@SKIP_SYSTEM_PACKAGES=1 SKIP_CUDA_TOOLKIT=1 bash setup/bootstrap_host.sh --python-env "$(VENV)"

download-assets:
	@bash -lc "source \"$(VENV)/bin/activate\" && $(PYTHON) scripts/download_olmo2_assets.py"

fetch-olmo:
	@bash scripts/fetch_olmo2_repo.sh "$(VENV)"

allenai-help:
	@bash -lc 'source "$(VENV)/bin/activate" && cd llm_original/olmo_2_repo && $(PYTHON) -m olmo.generate --help'

bootstrap-log:
	@if ls -rt logs/bootstrap_*.log >/dev/null 2>&1; then \
		ls -rt logs/bootstrap_*.log | tail -n 1; \
	else \
		echo "No bootstrap logs found."; \
	fi
