# Troubleshooting Guide

Use this checklist when the environment or inference validation fails. Each entry references the layer that usually reports the problem.

## CUDA / Driver Mismatch

- **Symptom:** `setup_env/check_gpu.py` warns about missing cudart or mismatched CUDA versions.
- **Fix:** Reinstall the NVIDIA driver that matches your host CUDA toolkit. On Vast.ai, redeploy the instance with a CUDA 12.x image if you intend to use the latest torch wheels.
- **Verify:** Rerun `bash setup_env/install_all.sh` and confirm the report lists matching driver / CUDA versions.

## Missing Weights or Tokenizer

- **Symptom:** `setup_env/verify_config.py` reports missing `model-*.safetensors` or `tokenizer.json`.
- **Fix:** Download weights into `weights/olmo2/` via `python scripts/download_weights.py --model-name olmo2` (or update `configs/default.yaml` to point elsewhere). Tokenizer assets must reside in `weights/olmo2/tokenizer/`.
- **Verify:** Re-run the config verifier or the full pipeline. The warnings should disappear.

## Inference Memory Errors

- **Symptom:** `scripts/test_inference.py` or custom runs raise `CUDA out of memory`.
- **Fixes:**
  - Reduce batch size or sequence length during testing.
  - Set `runtime.dtype` to `bfloat16` in `configs/default.yaml`.
  - Empty CUDA cache between runs (`torch.cuda.empty_cache()`).
- **Verify:** Re-run the sanity script. Check `reports/test_summary.json` for updated status.

## Vast.ai Mount Permissions

- **Symptom:** Installer cannot write to `weights/` or `reports/` when running inside a Vast.ai container.
- **Fix:** Ensure the mounted directory is writable (`chmod -R u+rwX /path`). When binding host directories, pass `:rw` to the Docker `--mount` flag.
- **Verify:** Touch a file inside the mount, then re-run the installer.

## Regenerating Reports

- After adjusting drivers, weights, or configs, run:
  ```bash
  bash setup_env/install_all.sh
  python setup_env/run_env_report.py
  ```
- The Markdown summary at `reports/environment_report.md` should reflect new state each time.

Need more help? Open an issue with the failing command output and attach `reports/` artefacts for faster triage.
