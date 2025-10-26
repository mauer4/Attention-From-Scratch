# Olmo 2 Inference Scripts

Utilities for running generation with the staged OLMo 2 snapshot.

- `run_from_snapshot.py` – invokes `llm_raw.olmo_2.test.run_olmo2_inference` so you can
  point at the local weights/metadata.
- `check_gpu.py` – wraps the GPU sanity script (loads the instruct checkpoint and prints
  diagnostics).
- `check_gpu_profiler.py` – runs the profiling variant that emits an Nsight trace.

Output logs/results can be stored alongside these scripts as needed.
