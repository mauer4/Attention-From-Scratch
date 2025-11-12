#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace custom_engine::v1 {

// Shape descriptor for a row-major linear layer: batch rows x in_dim features
// projected into out_dim outputs.
struct LinearShape {
    int batch = 0;
    int in_dim = 0;
    int out_dim = 0;
};

// Shape descriptor used by the prefill attention kernel.
struct PrefillAttentionShape {
    int seq_len = 0;
    int num_heads = 0;
    int head_dim = 0;
};

// Shape descriptor for decode attention (single token, many heads).
struct DecodeAttentionShape {
    int num_heads = 0;
    int head_dim = 0;
};

void launch_linear(const float* input,
                   const std::uint16_t* weight_bf16,
                   float* output,
                   const LinearShape& shape,
                   cudaStream_t stream);

void launch_prefill_attention(const float* q,
                              const float* k,
                              const float* v,
                              float* attn_out,
                              const PrefillAttentionShape& shape,
                              cudaStream_t stream);

void launch_decode_attention(const float* q,
                             const float* k_cache,
                             const float* v_cache,
                             float* attn_out,
                             const DecodeAttentionShape& shape,
                             int past_len,
                             cudaStream_t stream);

void launch_rms_norm(const float* input,
                     const std::uint16_t* weight_bf16,
                     float* output,
                     int tokens,
                     int hidden,
                     float eps,
                     cudaStream_t stream);

void launch_residual_add(const float* a,
                         const float* b,
                         float* out,
                         std::size_t count,
                         cudaStream_t stream);

void launch_silu_inplace(float* data, std::size_t count, cudaStream_t stream);
void launch_pointwise_mul(const float* a,
                          const float* b,
                          float* out,
                          std::size_t count,
                          cudaStream_t stream);

}  // namespace custom_engine::v1
