.PHONY: setup setup-vast download-assets run-olmo fetch-olmo allenai-help bootstrap-log

PYTHON ?= python3
VENV ?= .venv-olmo2

setup:
	@bash setup/bootstrap_host.sh

setup-vast:
	@SKIP_SYSTEM_PACKAGES=1 SKIP_CUDA_TOOLKIT=1 bash setup/bootstrap_host.sh --python-env "$(VENV)"

download-assets:
	@bash -lc "source \"$(VENV)/bin/activate\" && $(PYTHON) scripts/download_weights.py --model-name olmo2"

run-olmo:
	@bash -lc 'source "$(VENV)/bin/activate" && $(PYTHON) scripts/run_from_snapshot.py $(ARGS)'

fetch-olmo:
	@bash scripts/fetch_olmo2_repo.sh "$(VENV)"

allenai-help:
	@bash -lc 'source "$(VENV)/bin/activate" && $(PYTHON) - <<"PY"
	import pathlib
	import olmo

	repo_root = pathlib.Path(olmo.__file__).resolve().parents[1]
	readme = repo_root / "inference" / "README.md"
	print("OLMo package located at:", repo_root)
	print("\nRefer to the AllenAI inference notes here:\n", readme)
	print("\n(Flow 1 tooling relies on scripts under llm_original/olmo_2_repo/inference/.)")
	PY'

bootstrap-log:
	@if ls -rt logs/bootstrap_*.log >/dev/null 2>&1; then \
		ls -rt logs/bootstrap_*.log | tail -n 1; \
	else \
		echo "No bootstrap logs found."; \
	fi
