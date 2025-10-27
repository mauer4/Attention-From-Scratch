# Olmo 2 Inference Scripts

Utilities for running generation with the staged OLMo 2 snapshot.

- `run_from_snapshot.py` - invokes `run_olmo2_inference.py` so you can exercise the locally staged weights.
- `run_olmo2_inference.py` - loads the staged weights (`raw_weights/` + metadata) via AllenAI's inference helpers.
- `run_snapshot_sanity.py` - runs a quick inference and reports whether CUDA is active.
- `check_gpu.py` - wraps the GPU sanity script (loads the instruct checkpoint and prints diagnostics).
- `check_gpu_profiler.py` - runs the profiling variant that emits an Nsight trace.

Output logs/results can be stored alongside these scripts as needed.
