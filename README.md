# Attention-From-Scratch

[Project microsite](https://mauer4.github.io/attention-from-scratch/)

## Project Vision

I'm mapping the landscape of large language models to see what's possible for an individual, not just for massive tech teams. The exciting news is that fully open-source LLMs exist: you can inspect the weights, study the architecture, run the inference code, and even examine the training stack. As someone who loves marrying hardware, software, and math (classic HW/SW co-design), I'm inspired by the breakthroughs that pushed LLM performance forward: from FlashAttention and its fused kernels, to asynchronous attention in FlashAttention 2/3, and the paged-attention serving tricks in engines like vLLM. With today's cloud options, renting enterprise-class GPUs for inference is also within reach. So I've set a personal roadmap that takes me from beginner to building production-grade inference infrastructure on top of open models like Olmo 2. I'm not aiming to train new models; I'm aiming to run them exceptionally well. Along the way, I hope to understand the algorithms behind one of the most impactful technologies of our time and maybe even find new ways to push them further.

## Repository Structure

- `src/` - core C++ and CUDA source code.
- `include/` - public headers for the inference engine.
- `python_bindings/` - pybind11 bindings and packaging glue.
- `benchmarks/` - performance measurement utilities.
- `scripts/` - automation for environment setup and workflows.
- `tests/` - unit and integration tests.
- `docs/` - design notes, technical reports, and planning.

See `docs/PROJECT_PLAN.md` for the 12-week roadmap guiding development.

## Containerized Environment

Prerequisites:

- Docker with the NVIDIA Container Toolkit (`nvidia-smi` works inside containers).
- `docker compose` plugin.

Quick start:

1. Build the image: `docker compose build`
2. Launch an interactive session with GPU access: `docker compose run --rm dev`
3. Inside the container, verify CUDA: `nvidia-smi`
4. Build the project (already run during image build) or rerun as needed: `cmake --build build`

The working directory is mounted at `/workspace`, so changes sync to your host. Use the container for consistent dependency management across GPUs and environments.
