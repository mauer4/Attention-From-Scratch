# Text Generation Scripts

Utility scripts migrated from the legacy `scripts/` directory that exercise
OLMo-2 inference:

- `run_olmo2_inference.py` - simple prompt-based generation using Hugging Face
  transformers.
- `check_olmo_gpu.py` - loads the instruction-tuned checkpoint, generates a
  short completion, and reports GPU utilisation.
- `check_olmo_gpu_profiler.py` - similar to `check_olmo_gpu.py`, but includes
  basic profiling hooks.

These scripts are treated as raw reference assets alongside the released model
weights in `llm_raw/olmo_2/`.
