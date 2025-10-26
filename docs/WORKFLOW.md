# Repository Workflow

1. **Setup**
   - Run the appropriate venv script under `setup/venv/` (or `setup/bare_metal_setup.sh`) to provision the Python environment.
   - Install dependencies from `setup/requirements/requirements.txt` (handled automatically by the helper scripts).
   - Optional: build/run the Docker image for containerised development.

2. **LLM Setup**
   - `python scripts/download_olmo2_assets.py` fetches weights/metadata into `llm_raw/olmo_2/` and runs the analysis smoke tests.
   - Additional analysis utilities under `llm_setup/analysis/` regenerate inventories or verify shard integrity.

3. **LLM Raw**
   - Treat `llm_raw/olmo_2/` as read-only inputs (weights, tokenizer artefacts, snapshot cache, reference test scripts).

4. **LLM Original**
   - `scripts/fetch_olmo2_repo.sh` clones AllenAI's repository into `llm_original/olmo_2_repo/` for reference.

5. **Inference**
   - Use `inference/Olmo_2/` for generation scripts/logging with the staged snapshot.
   - `inference/From_Scratch/` is reserved for the custom engine once implemented.

This flow mirrors the diagram in the planning notes and keeps setup tasks, raw assets, and exploratory code cleanly separated.
