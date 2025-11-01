# Repository Workflow

1. **Bootstrap the environment**  
   Run `bash setup/bootstrap_host.sh` (or set the `SKIP_SYSTEM_PACKAGES` /
   `SKIP_CUDA_TOOLKIT` switches on restricted hosts). The script installs from
   the locked dependency set in `requirements/locks/olmo.lock`. See
   [`docs/ENVIRONMENT_AND_FLOWS.md`](ENVIRONMENT_AND_FLOWS.md) for the full
   decision matrix.

2. **Stage raw model assets**  
  `python scripts/download_weights.py --model-name olmo2` mirrors weights, tokenizer files,
   and metadata into `weights/olmo2/` and runs the validation suite. Both flows
   reuse this cache.

3. **Run the chosen flow**  
   - Flow 1 (Hugging Face bridge): `inference/Olmo_2/run_from_snapshot.py`
     generates text from the staged snapshot (`make run-olmo ARGS='...'`).
   - Flow 2 (custom engine): `src/`, `python_bindings/`, and
     `inference/From_Scratch/` will host the bespoke runtime as it lands.
   - Optional parity: `scripts/fetch_olmo2_repo.sh` syncs AllenAI’s repository
     when you want to cross-check with their CLI.

4. **Analysis and benchmarking**  
   Utilities under `scripts/` and `llm_setup/analysis/` inspect safetensors,
   summarise architecture layouts, and support Nsight profiling. Drop artefacts
   into `benchmarks/` as you collect numbers.

Make targets such as `make setup`, `make setup-vast`, `make download-assets`,
`make fetch-olmo`, and `make allenai-help` mirror the most common commands if
you prefer short invocations over typing each pipeline step.
