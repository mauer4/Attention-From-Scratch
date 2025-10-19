# From-Scratch LLM Inference Engine: 12-Week Plan

This roadmap guides the development of a custom transformer inference stack focused on running open models efficiently on Vast.AI NVIDIA GPUs.

---

## Week 1 – Vast.AI Setup + Baseline Environment

**Goal:** Have a fully functioning development environment with GPU access, libraries, and tools. Run a small LLM (for example Mistral-7B or Llama-3-8B) locally for reference.

**Checklist:**

- [ ] Create Vast.AI account, allocate GPU (A100 40/80 GB or 4090).
- [ ] Create a volume for persistent model and build cache.
- [ ] Launch container (for example `nvcr.io/nvidia/pytorch:25.04-py3`).
- [ ] Install build tools: `git`, `cmake`, `ninja`, `g++`, `clang`, `python3-dev`.
- [ ] Install libraries: `torch`, `transformers`, `safetensors`, `sentencepiece`, `numpy`, `cutlass`, `pybind11`.
- [ ] Test CUDA and NVCC (`nvcc --version`, simple kernel test).
- [ ] Clone and run a reference open-source LLM (for example `mistralai/Mistral-7B-v0.1`).
- [ ] Measure baseline metrics:
  - Tokens/s for batch = 1 and 8
  - GPU memory usage
  - Prefill vs decode latency
  - CPU vs GPU utilization
- [ ] Save these numbers as the performance baseline report.

## Week 2 – Repo Scaffolding + Tokenizer + Weight Loader

**Goal:** Build the inference project skeleton and verify tokenizer and weight loading correctness.

**Checklist:**

- [ ] Initialize repo layout:
  - `src/` for C++ and CUDA code
  - `include/` headers
  - `python_bindings/` for pybind11 integration
  - `tests/`, `scripts/`, `benchmarks/`
- [ ] Implement `CMakeLists.txt` (build both shared and executable).
- [ ] Write a tokenizer wrapper for SentencePiece.
  - [ ] Encode/decode sample text and verify output matches HuggingFace.
- [ ] Write a weight loader for safetensors.
  - [ ] Read tensor metadata and store in custom memory layout (row-major fp16).
  - [ ] Verify all layer names and shapes match the model config.
- [ ] Add unit tests comparing tensor values (mean abs diff < 1e-5).

## Week 3 – Forward Pass Framework + Basic Kernels

**Goal:** Create the model forward graph (no optimization yet) and write basic CUDA kernels.

**Checklist:**

- [ ] Define `ModelConfig` (n_layers, n_heads, hidden_size, rope_theta, etc.).
- [ ] Implement RMSNorm kernel and verify vs PyTorch.
- [ ] Implement RoPE (rotary embeddings) kernel.
- [ ] Implement matmul wrapper (use cuBLAS or CUTLASS initially).
- [ ] Implement MLP (SwiGLU) forward function.
- [ ] Implement attention (naive): QKV projection → softmax → weighted sum.
- [ ] Chain layers for one forward pass (prefill only).
- [ ] Compare layerwise activations against HuggingFace forward (mean abs error < 1e-3).

## Week 4 – Prefill + Decode Loop + Sampling

**Goal:** Build an end-to-end inference loop generating tokens sequentially.

**Checklist:**

- [ ] Implement KV cache (contiguous [batch, layer, seq, dim]).
- [ ] Implement greedy decode loop:
  - Prefill sequence
  - Decode token by token
- [ ] Add top-k, top-p, and temperature sampling.
- [ ] Validate generated text matches HuggingFace outputs on short prompts.
- [ ] Measure throughput (tokens/s), latency, memory use.

## Week 5 – FlashAttention-Style Optimization

**Goal:** Replace naive attention with optimized block-sparse version.

**Checklist:**

- [ ] Review FlashAttention algorithm.
- [ ] Implement tiled softmax and attention kernel using shared memory.
- [ ] Test for correctness on 128–4096 tokens.
- [ ] Benchmark vs Week 4 attention (speed, VRAM).
- [ ] Integrate timing macros and CUDA events.
- [ ] Record kernel efficiency in Nsight Compute (check occupancy and memory bandwidth).

## Week 6 – Advanced KV Cache + CUDA Graphs

**Goal:** Create a paged KV cache and introduce CUDA Graph capture for decode efficiency.

**Checklist:**

- [ ] Implement paged or slab KV cache allocator (inspired by vLLM).
- [ ] Support variable sequence lengths and in-flight batching.
- [ ] Add CUDA Graphs for the decode kernel chain.
- [ ] Create micro-benchmark to measure graph capture overhead.
- [ ] Achieve less than or equal to 1 microsecond per token scheduling overhead.

## Week 7 – Quantization (AWQ / GPTQ / NF4)

**Goal:** Reduce bandwidth by weight-only quantization.

**Checklist:**

- [ ] Implement group-wise quantization (for example 128-element groups).
- [ ] Compute scales, zero-points, and store compressed weights.
- [ ] Modify GEMM kernel to dequantize on the fly.
- [ ] Validate logits within acceptable tolerance vs fp16 baseline.
- [ ] Evaluate throughput and memory improvement.
- [ ] Test with `bitsandbytes` or `AutoAWQ` reference.

## Week 8 – Batching, Streams, and Scheduler

**Goal:** Improve throughput via concurrent streams and multi-request batching.

**Checklist:**

- [ ] Create request queue system (prefill and decode queues).
- [ ] Assign separate CUDA streams for prefill and copy.
- [ ] Implement host-side tokenization overlap (H2D async copy).
- [ ] Benchmark tokens/s vs Week 6 (expect at least 2x speedup with batching).
- [ ] Test with random mixed prompt lengths.

## Week 9 – Parallelism and Memory Optimization

**Goal:** Extend to multi-GPU or at least multi-stream execution.

**Checklist:**

- [ ] Add tensor parallelism (split QKV/MLP across GPUs).
- [ ] Use NCCL for all-reduce synchronization.
- [ ] Measure scaling from 1 to 2 GPUs.
- [ ] Introduce activation checkpointing for long contexts.
- [ ] Start implementing speculative lookahead decoding skeleton.

## Week 10 – Speculative and Lookahead Decoding

**Goal:** Integrate dependency-graph decoding acceleration.

**Checklist:**

- [ ] Implement lookahead decoding (parallel compute of next few tokens).
- [ ] Evaluate correctness (rollback when mismatch occurs).
- [ ] Measure speedup ratio vs normal decode.
- [ ] Integrate optional Medusa head (multi-token prediction).
- [ ] Log per-token latency improvement.

## Week 11 – Serving Layer

**Goal:** Expose the engine as a local API service.

**Checklist:**

- [ ] Build lightweight HTTP server (FastAPI or C++ alternative).
- [ ] Add `/v1/completions` and `/v1/chat/completions` endpoints.
- [ ] Implement batching inside the server.
- [ ] Add metrics: tokens/s, active sessions, latency histogram.
- [ ] Add KV eviction policy (LRU by session ID).
- [ ] Test concurrency with at least 8 parallel clients.

## Week 12 – Validation, Docs, and Release

**Goal:** Publish and benchmark the engine; compare to open baselines.

**Checklist:**

- [ ] Validate numerical parity with HuggingFace (logit diff < 1e-3).
- [ ] Benchmark tokens/s vs `llama.cpp`, `vLLM`, and `TensorRT-LLM`.
- [ ] Run stress tests (long prompts > 8k tokens).
- [ ] Write full technical report:
  - Architecture diagrams
  - Profiling results
  - Performance comparisons
  - Lessons learned
- [ ] Publish repo (GitHub), include docs and build scripts.
- [ ] Optional: record demo video or release blog.

---

## Phase Summary

| Phase | Milestone | Core Output |
| ----- | --------- | ----------- |
| Weeks 1-4 | Baseline and correctness | Functional prefill and decode engine matching HuggingFace output |
| Weeks 5-8 | Performance | FlashAttention, paged KV cache, batching, quantization |
| Weeks 9-10 | Research | Speculative and lookahead decoding prototypes |
| Weeks 11-12 | Deployment | API server, benchmarks, public release |

