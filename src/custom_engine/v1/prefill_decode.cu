#include "custom_engine/v1/prefill_decode.hpp"

#include "custom_engine/utils/logging.h"
#include "custom_engine/v1/kernels.cuh"

namespace custom_engine::v1 {

namespace {

inline int hidden_stride(int tokens, int hidden) {
    return tokens * hidden;
}

}  // namespace

bool KVCache::init(int max_seq_len, int num_heads, int head_dim, cudaStream_t stream) {
    capacity_ = max_seq_len;
    num_heads_ = num_heads;
    head_dim_ = head_dim;
    cursor_ = 0;
    const std::size_t elements = static_cast<std::size_t>(max_seq_len) * num_heads * head_dim;
    k_cache_ = custom_engine::utils::DeviceBuffer<float>(elements, stream);
    v_cache_ = custom_engine::utils::DeviceBuffer<float>(elements, stream);
    return k_cache_.get() != nullptr && v_cache_.get() != nullptr;
}

void KVCache::reset() {
    cursor_ = 0;
}

void KVCache::append(const float* k_src, const float* v_src, int tokens, cudaStream_t stream) {
    ATTN_CHECK(k_src != nullptr && v_src != nullptr);
    ATTN_CHECK(cursor_ + tokens <= capacity_);
    const std::size_t slice = static_cast<std::size_t>(tokens) * num_heads_ * head_dim_;
    float* k_dst = k_cache_.get() + static_cast<std::size_t>(cursor_) * num_heads_ * head_dim_;
    float* v_dst = v_cache_.get() + static_cast<std::size_t>(cursor_) * num_heads_ * head_dim_;
    CUDA_CHECK(cudaMemcpyAsync(k_dst, k_src, slice * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(v_dst, v_src, slice * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    cursor_ += tokens;
}

static void check_attention_config(const AttentionConfig& cfg) {
    ATTN_CHECK(cfg.hidden_size == cfg.num_heads * cfg.head_dim);
}

void prefill_forward(const float* hidden_in,
                     float* hidden_out,
                     const LayerWeights& weights,
                     int seq_len,
                     const AttentionConfig& cfg,
                     KVCache* cache,
                     cudaStream_t stream) {
    check_attention_config(cfg);
    ATTN_CHECK(hidden_in && hidden_out);
    ATTN_CHECK(weights.w_q && weights.w_k && weights.w_v && weights.w_o);
    ATTN_CHECK(seq_len > 0);
    ATTN_CHECK(cache != nullptr);

    const int hidden = cfg.hidden_size;
    const std::size_t tokens_elements = hidden_stride(seq_len, hidden);
    custom_engine::utils::DeviceBuffer<float> norm_attn(tokens_elements, stream);
    custom_engine::utils::DeviceBuffer<float> q(tokens_elements, stream);
    custom_engine::utils::DeviceBuffer<float> k(tokens_elements, stream);
    custom_engine::utils::DeviceBuffer<float> v(tokens_elements, stream);
    custom_engine::utils::DeviceBuffer<float> attn(tokens_elements, stream);
    custom_engine::utils::DeviceBuffer<float> resid(tokens_elements, stream);
    custom_engine::utils::DeviceBuffer<float> norm_mlp(tokens_elements, stream);
    custom_engine::utils::DeviceBuffer<float> gate(tokens_elements, stream);
    custom_engine::utils::DeviceBuffer<float> up(tokens_elements, stream);
    custom_engine::utils::DeviceBuffer<float> mlp(tokens_elements, stream);

    launch_rms_norm(hidden_in, weights.rms_attn, norm_attn.get(), seq_len, hidden, cfg.rms_norm_eps, stream);

    const LinearShape linear_shape{seq_len, hidden, hidden};
    launch_linear(norm_attn.get(), weights.w_q, q.get(), linear_shape, stream);
    launch_linear(norm_attn.get(), weights.w_k, k.get(), linear_shape, stream);
    launch_linear(norm_attn.get(), weights.w_v, v.get(), linear_shape, stream);

    const PrefillAttentionShape attn_shape{seq_len, cfg.num_heads, cfg.head_dim};
    launch_prefill_attention(q.get(), k.get(), v.get(), attn.get(), attn_shape, stream);

    launch_linear(attn.get(), weights.w_o, resid.get(), linear_shape, stream);
    launch_residual_add(hidden_in, resid.get(), resid.get(), tokens_elements, stream);

    launch_rms_norm(resid.get(), weights.rms_mlp, norm_mlp.get(), seq_len, hidden, cfg.rms_norm_eps, stream);

    launch_linear(norm_mlp.get(), weights.w_gate, gate.get(), linear_shape, stream);
    launch_linear(norm_mlp.get(), weights.w_up, up.get(), linear_shape, stream);
    launch_silu_inplace(gate.get(), tokens_elements, stream);
    launch_pointwise_mul(gate.get(), up.get(), gate.get(), tokens_elements, stream);
    launch_linear(gate.get(), weights.w_down, mlp.get(), linear_shape, stream);

    launch_residual_add(resid.get(), mlp.get(), hidden_out, tokens_elements, stream);

    cache->append(k.get(), v.get(), seq_len, stream);
}

void decode_forward(const float* hidden_in,
                    float* hidden_out,
                    const LayerWeights& weights,
                    const AttentionConfig& cfg,
                    KVCache* cache,
                    cudaStream_t stream) {
    check_attention_config(cfg);
    ATTN_CHECK(hidden_in && hidden_out);
    ATTN_CHECK(weights.w_q && weights.w_k && weights.w_v && weights.w_o);
    ATTN_CHECK(cache != nullptr);
    ATTN_CHECK(cache->size() <= cache->capacity());

    const int hidden = cfg.hidden_size;
    const LinearShape linear_shape{1, hidden, hidden};
    custom_engine::utils::DeviceBuffer<float> norm_attn(hidden, stream);
    custom_engine::utils::DeviceBuffer<float> q(hidden, stream);
    custom_engine::utils::DeviceBuffer<float> k(hidden, stream);
    custom_engine::utils::DeviceBuffer<float> v(hidden, stream);
    custom_engine::utils::DeviceBuffer<float> attn(hidden, stream);
    custom_engine::utils::DeviceBuffer<float> resid(hidden, stream);
    custom_engine::utils::DeviceBuffer<float> norm_mlp(hidden, stream);
    custom_engine::utils::DeviceBuffer<float> gate(hidden, stream);
    custom_engine::utils::DeviceBuffer<float> up(hidden, stream);
    custom_engine::utils::DeviceBuffer<float> mlp(hidden, stream);

    launch_rms_norm(hidden_in, weights.rms_attn, norm_attn.get(), 1, hidden, cfg.rms_norm_eps, stream);

    launch_linear(norm_attn.get(), weights.w_q, q.get(), linear_shape, stream);
    launch_linear(norm_attn.get(), weights.w_k, k.get(), linear_shape, stream);
    launch_linear(norm_attn.get(), weights.w_v, v.get(), linear_shape, stream);

    cache->append(k.get(), v.get(), 1, stream);
    const int total_tokens = cache->size();
    const DecodeAttentionShape attn_shape{cfg.num_heads, cfg.head_dim};
    launch_decode_attention(q.get(), cache->k_data(), cache->v_data(), attn.get(),
                           attn_shape, total_tokens, stream);

    launch_linear(attn.get(), weights.w_o, resid.get(), linear_shape, stream);
    launch_residual_add(hidden_in, resid.get(), resid.get(), static_cast<std::size_t>(hidden), stream);

    launch_rms_norm(resid.get(), weights.rms_mlp, norm_mlp.get(), 1, hidden, cfg.rms_norm_eps, stream);

    launch_linear(norm_mlp.get(), weights.w_gate, gate.get(), linear_shape, stream);
    launch_linear(norm_mlp.get(), weights.w_up, up.get(), linear_shape, stream);
    launch_silu_inplace(gate.get(), static_cast<std::size_t>(hidden), stream);
    launch_pointwise_mul(gate.get(), up.get(), gate.get(), static_cast<std::size_t>(hidden), stream);
    launch_linear(gate.get(), weights.w_down, mlp.get(), linear_shape, stream);

    launch_residual_add(resid.get(), mlp.get(), hidden_out, static_cast<std::size_t>(hidden), stream);
}

}  // namespace custom_engine::v1
