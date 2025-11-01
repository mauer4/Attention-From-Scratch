# Change Log

- Moved legacy OLMo scripts into `scripts/` with deprecation shims left under `inference/Olmo_2/`.
- Normalised model assets under `weights/olmo2/` and wired up manifest generation.
- Added layered setup automation (`setup_env/*`) including GPU probe, dependency verification, and consolidated reporting.
- Introduced canonical downloader (`scripts/download_weights.py`) and GPU sanity harness (`scripts/test_inference.py`).
- Updated documentation and maintenance policy (`codex_rules.yaml`) to reference the new workflow.
