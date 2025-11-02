# Olmo 2 Inference Scripts

Legacy utilities for running generation with the staged OLMo 2 snapshot. The
primary inference flow now relies on AllenAI's repository; these scripts remain
available for debugging and quick experiments.

- `run_from_snapshot.py` - invokes `run_olmo2_inference.py` so you can exercise the locally staged weights.
- `run_olmo2_inference.py` - loads the staged weights (`$MODEL_WEIGHTS_ROOT/<model>/`, default `weights/olmo2/olmo2/`) via AllenAI's inference helpers.
- `run_snapshot_sanity.py` - runs a quick inference and reports whether CUDA is active.
- `check_gpu.py` - wraps the GPU sanity script (loads the instruct checkpoint and prints diagnostics).
- `check_gpu_profiler.py` - runs the profiling variant that emits an Nsight trace.

Output logs/results can be stored alongside these scripts as needed.
