#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "custom_engine/utils/cuda_utils.cuh"
#include "custom_engine/utils/logging.h"

namespace custom_engine::v1 {

struct AttentionConfig {
    int hidden_size = 0;
    int num_heads = 0;
    int head_dim = 0;
    float rms_norm_eps = 1e-6f;
};

struct LayerWeights {
    const std::uint16_t* w_q = nullptr;
    const std::uint16_t* w_k = nullptr;
    const std::uint16_t* w_v = nullptr;
    const std::uint16_t* w_o = nullptr;
    const std::uint16_t* w_gate = nullptr;
    const std::uint16_t* w_up = nullptr;
    const std::uint16_t* w_down = nullptr;
    const std::uint16_t* rms_attn = nullptr;
    const std::uint16_t* rms_mlp = nullptr;
};

class KVCache {
public:
    KVCache() = default;

    bool init(int max_seq_len, int num_heads, int head_dim, cudaStream_t stream = nullptr);
    void reset();
    void append(const float* k_src, const float* v_src, int tokens, cudaStream_t stream = nullptr);

    float* k_data() { return k_cache_.get(); }
    float* v_data() { return v_cache_.get(); }
    const float* k_data() const { return k_cache_.get(); }
    const float* v_data() const { return v_cache_.get(); }

    int size() const { return cursor_; }
    int capacity() const { return capacity_; }
    int num_heads() const { return num_heads_; }
    int head_dim() const { return head_dim_; }

private:
    custom_engine::utils::DeviceBuffer<float> k_cache_;
    custom_engine::utils::DeviceBuffer<float> v_cache_;
    int capacity_ = 0;
    int num_heads_ = 0;
    int head_dim_ = 0;
    int cursor_ = 0;
};

void prefill_forward(const float* hidden_in,
                     float* hidden_out,
                     const LayerWeights& weights,
                     int seq_len,
                     const AttentionConfig& cfg,
                     KVCache* cache,
                     cudaStream_t stream = nullptr);

void decode_forward(const float* hidden_in,
                    float* hidden_out,
                    const LayerWeights& weights,
                    const AttentionConfig& cfg,
                    KVCache* cache,
                    cudaStream_t stream = nullptr);

}  // namespace custom_engine::v1
