# Inference Architecture

The project separates open-source engines (OLMo, Hugging Face loaders) from the evolving custom runtime. Both share the same environment, configuration files, and reporting surfaces.

```
Input → Tokenizer → Prefill → Decode → Output
```

## Engine Layers

- **Tokenizer** – Normalises text into token IDs (`weights/olmo2/tokenizer/`).
- **Prefill** – Runs the initial forward pass to populate KV caches.
- **Decode** – Iteratively generates new tokens while consulting KV caches and scheduler policies.
- **Output** – Converts generated IDs back to text and returns metadata (timings, token counts).

## Custom vs Open-Source Engines

| Aspect | Custom Engine (`custom_engine`) | Open-Source (`transformers` / `OLMo`) |
| ------ | -------------------------------- | ------------------------------------ |
| Kernels | Planned bespoke CUDA kernels with FlashAttention-style tiling | Vendor-provided kernels and fused operators |
| Memory Layout | Paged KV cache (in development) | Standard attention cache | 
| Extensibility | Direct hooks into benchmarking utilities and telemetry | Easy baseline, broad model coverage |

## Extending to New Models

1. Create a weights folder under `weights/<model_name>/` with safetensors shards.
2. Duplicate `configs/default.yaml` into a new config that points to the new directories.
3. Implement an adapter in `src/inference/` or extend `custom_engine`.
4. Add sanity and benchmark steps under `scripts/` or notebooks within `analysis/`.
5. Rerun the setup pipeline to refresh dependency locks and reports.

Tracking progress:

- `docs/PROJECT_PLAN.md` – Roadmap and milestones.
- `analysis/compare_engines.py` – Compare baseline and custom latency/throughput.
- `reports/environment_report.md` – Snapshot of environment health for each run.
