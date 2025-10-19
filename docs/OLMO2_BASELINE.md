# Olmo 2 Baseline Inference on Vast.AI

This guide sets up an isolated Python environment to run AllenAI's Olmo 2 checkpoints on a Vast.AI GPU instance. It keeps dependency management separate from the main project bootstrap so you can iterate and benchmark independently.

## 1. Prepare the Environment

```bash
git clone https://github.com/mauer4/Attention-From-Scratch.git
cd Attention-From-Scratch

# Create (or reuse) a dedicated venv for Olmo 2.
bash scripts/setup_olmo2_env.sh [.venv-olmo2]

source .venv-olmo2/bin/activate
```

The setup script installs the CUDA-enabled PyTorch wheels plus the Hugging Face stack (`transformers`, `accelerate`, `sentencepiece`, `bitsandbytes`, etc.). No changes are made to the system Python.

## 2. Run a Baseline Generation

```bash
python scripts/run_olmo2_inference.py \
  --model allenai/OLMo-2-7B \
  --prompt "Summarize the Olmo 2 architecture." \
  --max-new-tokens 128 \
  --temperature 0.7
```

What happens:

- The script loads the tokenizer and model with half precision on GPU (if available).
- It optionally reports tokens/sec for a quick throughput estimate.
- You can save output using `--output-path generated.txt`.
- Use `--compile` to experiment with `torch.compile` on PyTorch 2.0+.

To explore other checkpoints (e.g. the 13B variant), change the `--model` flag to the desired Hugging Face repo.

## 3. Next Steps Toward Profiling

The goal is to transition from “baseline hf-inference” into detailed kernel profiling with Nsight:

- Record consistent prompts and batch sizes to compare throughput vs. the forthcoming custom engine.
- Capture GPU utilization (`nvidia-smi dmon`) and memory footprints during runs.
- Install Nsight Systems / Nsight Compute inside the Vast container or stream the session back to a local GUI. NVIDIA’s container images already include the CLI tools (`nsys`, `ncu`).
- Re-run the baseline script under Nsight to collect timeline traces and kernel metrics (e.g., `nsys profile python scripts/run_olmo2_inference.py ...`).
- Store reports under `benchmarks/` (keep traces outside Git or add to `.gitignore`).

These measurements will anchor the performance targets for the from-scratch implementation described in `docs/PROJECT_PLAN.md`.
