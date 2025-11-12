#include "custom_engine/v1/kernels.cuh"

#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "custom_engine/utils/cuda_utils.cuh"

namespace {

__device__ inline float bf16_to_float_device(std::uint16_t bits) {
    std::uint32_t full = static_cast<std::uint32_t>(bits) << 16;
    return __uint_as_float(full);
}

__global__ void linear_kernel(const float* input,
                              const std::uint16_t* weight,
                              float* output,
                              int batch,
                              int in_dim,
                              int out_dim) {
    int row = blockIdx.x;
    int col = blockIdx.y * blockDim.x + threadIdx.x;
    if (row >= batch || col >= out_dim) {
        return;
    }

    const float* in_ptr = input + row * in_dim;
    const std::uint16_t* w_ptr = weight + col * in_dim;
    float acc = 0.0f;
    for (int k = 0; k < in_dim; ++k) {
        acc += in_ptr[k] * bf16_to_float_device(w_ptr[k]);
    }
    output[row * out_dim + col] = acc;
}

__global__ void causal_prefill_kernel(const float* q,
                                      const float* k,
                                      const float* v,
                                      float* attn_out,
                                      int seq_len,
                                      int num_heads,
                                      int head_dim) {
    int idx = blockIdx.x;
    const int total = seq_len * num_heads;
    if (idx >= total) {
        return;
    }

    const int token = idx / num_heads;
    const int head = idx % num_heads;
    const int head_stride = num_heads * head_dim;
    const float* q_vec = q + idx * head_dim;
    float* out_vec = attn_out + idx * head_dim;
    const float inv_scale = rsqrtf(static_cast<float>(head_dim));

    float max_score = -FLT_MAX;
    for (int key = 0; key <= token; ++key) {
        const float* k_vec = k + (key * head_stride) + head * head_dim;
        float score = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            score += q_vec[d] * k_vec[d];
        }
        score *= inv_scale;
        max_score = fmaxf(max_score, score);
    }

    float denom = 0.0f;
    for (int key = 0; key <= token; ++key) {
        const float* k_vec = k + (key * head_stride) + head * head_dim;
        float score = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            score += q_vec[d] * k_vec[d];
        }
        score = (score * inv_scale) - max_score;
        denom += __expf(score);
    }
    denom = denom > 0.0f ? denom : 1.0f;

    extern __shared__ float accum[];
    for (int d = 0; d < head_dim; ++d) {
        accum[d] = 0.0f;
    }

    for (int key = 0; key <= token; ++key) {
        const float* k_vec = k + (key * head_stride) + head * head_dim;
        const float* v_vec = v + (key * head_stride) + head * head_dim;
        float score = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            score += q_vec[d] * k_vec[d];
        }
        score = (score * inv_scale) - max_score;
        float weight = __expf(score) / denom;
        for (int d = 0; d < head_dim; ++d) {
            accum[d] += weight * v_vec[d];
        }
    }

    for (int d = 0; d < head_dim; ++d) {
        out_vec[d] = accum[d];
    }
}

__global__ void causal_decode_kernel(const float* q,
                                     const float* k_cache,
                                     const float* v_cache,
                                     float* attn_out,
                                     int past_len,
                                     int num_heads,
                                     int head_dim) {
    int head = blockIdx.x;
    if (head >= num_heads) {
        return;
    }

    const float* q_vec = q + head * head_dim;
    float* out_vec = attn_out + head * head_dim;
    const int head_stride = num_heads * head_dim;
    const float inv_scale = rsqrtf(static_cast<float>(head_dim));

    float max_score = -FLT_MAX;
    for (int key = 0; key < past_len; ++key) {
        const float* k_vec = k_cache + (key * head_stride) + head * head_dim;
        float score = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            score += q_vec[d] * k_vec[d];
        }
        score *= inv_scale;
        max_score = fmaxf(max_score, score);
    }

    float denom = 0.0f;
    for (int key = 0; key < past_len; ++key) {
        const float* k_vec = k_cache + (key * head_stride) + head * head_dim;
        float score = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            score += q_vec[d] * k_vec[d];
        }
        score = (score * inv_scale) - max_score;
        denom += __expf(score);
    }
    denom = denom > 0.0f ? denom : 1.0f;

    extern __shared__ float accum[];
    for (int d = 0; d < head_dim; ++d) {
        accum[d] = 0.0f;
    }
    for (int key = 0; key < past_len; ++key) {
        const float* k_vec = k_cache + (key * head_stride) + head * head_dim;
        const float* v_vec = v_cache + (key * head_stride) + head * head_dim;
        float score = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            score += q_vec[d] * k_vec[d];
        }
        score = (score * inv_scale) - max_score;
        float weight = __expf(score) / denom;
        for (int d = 0; d < head_dim; ++d) {
            accum[d] += weight * v_vec[d];
        }
    }

    for (int d = 0; d < head_dim; ++d) {
        out_vec[d] = accum[d];
    }
}

__global__ void rms_norm_kernel(const float* input,
                                const std::uint16_t* weight,
                                float* output,
                                int hidden,
                                float eps) {
    int token = blockIdx.x;
    const float* in_vec = input + token * hidden;
    float* out_vec = output + token * hidden;

    float accum = 0.0f;
    for (int i = 0; i < hidden; ++i) {
        accum += in_vec[i] * in_vec[i];
    }
    accum /= hidden;
    const float scale = rsqrtf(accum + eps);
    for (int i = 0; i < hidden; ++i) {
        out_vec[i] = in_vec[i] * scale * bf16_to_float_device(weight[i]);
    }
}

__global__ void residual_add_kernel(const float* a,
                                    const float* b,
                                    float* out,
                                    std::size_t count) {
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }
    out[idx] = a[idx] + b[idx];
}

__global__ void silu_kernel(float* data, std::size_t count) {
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }
    const float x = data[idx];
    data[idx] = x / (1.0f + expf(-x));
}

__global__ void pointwise_mul_kernel(const float* a,
                                     const float* b,
                                     float* out,
                                     std::size_t count) {
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }
    out[idx] = a[idx] * b[idx];
}

}  // namespace

namespace custom_engine::v1 {

void launch_linear(const float* input,
                   const std::uint16_t* weight,
                   float* output,
                   const LinearShape& shape,
                   cudaStream_t stream) {
    if (shape.batch == 0 || shape.in_dim == 0 || shape.out_dim == 0) {
        return;
    }
    dim3 block(128, 1, 1);
    dim3 grid(shape.batch, (shape.out_dim + block.x - 1) / block.x, 1);
    linear_kernel<<<grid, block, 0, stream>>>(input, weight, output, shape.batch, shape.in_dim, shape.out_dim);
    CUDA_CHECK(cudaGetLastError());
}

void launch_prefill_attention(const float* q,
                              const float* k,
                              const float* v,
                              float* attn_out,
                              const PrefillAttentionShape& shape,
                              cudaStream_t stream) {
    if (shape.seq_len == 0 || shape.num_heads == 0 || shape.head_dim == 0) {
        return;
    }
    const int total = shape.seq_len * shape.num_heads;
    const dim3 grid(total, 1, 1);
    const dim3 block(1, 1, 1);
    const std::size_t shared_bytes = static_cast<std::size_t>(shape.head_dim) * sizeof(float);
    causal_prefill_kernel<<<grid, block, shared_bytes, stream>>>(q, k, v, attn_out,
                                                                shape.seq_len,
                                                                shape.num_heads,
                                                                shape.head_dim);
    CUDA_CHECK(cudaGetLastError());
}

void launch_decode_attention(const float* q,
                             const float* k_cache,
                             const float* v_cache,
                             float* attn_out,
                             const DecodeAttentionShape& shape,
                             int past_len,
                             cudaStream_t stream) {
    if (shape.num_heads == 0 || shape.head_dim == 0 || past_len == 0) {
        return;
    }
    const dim3 grid(shape.num_heads, 1, 1);
    const dim3 block(1, 1, 1);
    const std::size_t shared_bytes = static_cast<std::size_t>(shape.head_dim) * sizeof(float);
    causal_decode_kernel<<<grid, block, shared_bytes, stream>>>(q, k_cache, v_cache, attn_out,
                                                                past_len,
                                                                shape.num_heads,
                                                                shape.head_dim);
    CUDA_CHECK(cudaGetLastError());
}

void launch_rms_norm(const float* input,
                     const std::uint16_t* weight,
                     float* output,
                     int tokens,
                     int hidden,
                     float eps,
                     cudaStream_t stream) {
    if (tokens == 0 || hidden == 0) {
        return;
    }
    const dim3 grid(tokens, 1, 1);
    const dim3 block(1, 1, 1);
    rms_norm_kernel<<<grid, block, 0, stream>>>(input, weight, output, hidden, eps);
    CUDA_CHECK(cudaGetLastError());
}

void launch_residual_add(const float* a,
                         const float* b,
                         float* out,
                         std::size_t count,
                         cudaStream_t stream) {
    if (count == 0) {
        return;
    }
    const dim3 block(256, 1, 1);
    const dim3 grid((count + block.x - 1) / block.x, 1, 1);
    residual_add_kernel<<<grid, block, 0, stream>>>(a, b, out, count);
    CUDA_CHECK(cudaGetLastError());
}

void launch_silu_inplace(float* data, std::size_t count, cudaStream_t stream) {
    if (count == 0) {
        return;
    }
    const dim3 block(256, 1, 1);
    const dim3 grid((count + block.x - 1) / block.x, 1, 1);
    silu_kernel<<<grid, block, 0, stream>>>(data, count);
    CUDA_CHECK(cudaGetLastError());
}

void launch_pointwise_mul(const float* a,
                          const float* b,
                          float* out,
                          std::size_t count,
                          cudaStream_t stream) {
    if (count == 0) {
        return;
    }
    const dim3 block(256, 1, 1);
    const dim3 grid((count + block.x - 1) / block.x, 1, 1);
    pointwise_mul_kernel<<<grid, block, 0, stream>>>(a, b, out, count);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace custom_engine::v1
