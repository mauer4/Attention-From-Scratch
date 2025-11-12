#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "custom_engine/utils/cuda_utils.cuh"
#include "custom_engine/utils/logging.h"
#include "custom_engine/v1/prefill_decode.hpp"

using custom_engine::v1::AttentionConfig;
using custom_engine::v1::KVCache;
using custom_engine::v1::LayerWeights;

namespace {

float dot(const float* a, const float* b, int dim) {
    float acc = 0.0f;
    for (int i = 0; i < dim; ++i) {
        acc += a[i] * b[i];
    }
    return acc;
}

std::uint16_t float_to_bf16(float value) {
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return static_cast<std::uint16_t>(bits >> 16);
}

std::vector<float> cpu_rms_norm(const std::vector<float>& input,
                                int tokens,
                                int hidden,
                                float eps,
                                const std::vector<float>& weight) {
    std::vector<float> out(tokens * hidden);
    for (int t = 0; t < tokens; ++t) {
        const float* in_vec = input.data() + t * hidden;
        float* out_vec = out.data() + t * hidden;
        float acc = 0.0f;
        for (int i = 0; i < hidden; ++i) {
            acc += in_vec[i] * in_vec[i];
        }
        acc /= hidden;
        const float scale = 1.0f / std::sqrt(acc + eps);
        for (int i = 0; i < hidden; ++i) {
            out_vec[i] = in_vec[i] * scale * weight[i];
        }
    }
    return out;
}

void cpu_causal_attention(const float* q,
                          const float* k,
                          const float* v,
                          float* out,
                          int seq_len,
                          int num_heads,
                          int head_dim) {
    const float inv_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const int head_stride = num_heads * head_dim;
    for (int token = 0; token < seq_len; ++token) {
        for (int head = 0; head < num_heads; ++head) {
            const float* q_vec = q + (token * head_stride) + head * head_dim;
            float* out_vec = out + (token * head_stride) + head * head_dim;

            float max_score = -std::numeric_limits<float>::infinity();
            for (int key = 0; key <= token; ++key) {
                const float* k_vec = k + (key * head_stride) + head * head_dim;
                float score = dot(q_vec, k_vec, head_dim) * inv_scale;
                if (score > max_score) {
                    max_score = score;
                }
            }

            float denom = 0.0f;
            for (int key = 0; key <= token; ++key) {
                const float* k_vec = k + (key * head_stride) + head * head_dim;
                float score = dot(q_vec, k_vec, head_dim) * inv_scale;
                denom += std::exp(score - max_score);
            }
            if (denom == 0.0f) denom = 1.0f;

            for (int d = 0; d < head_dim; ++d) {
                out_vec[d] = 0.0f;
            }
            for (int key = 0; key <= token; ++key) {
                const float* k_vec = k + (key * head_stride) + head * head_dim;
                const float* v_vec = v + (key * head_stride) + head * head_dim;
                float score = dot(q_vec, k_vec, head_dim) * inv_scale;
                float weight = std::exp(score - max_score) / denom;
                for (int d = 0; d < head_dim; ++d) {
                    out_vec[d] += weight * v_vec[d];
                }
            }
        }
    }
}

bool nearly_equal(const std::vector<float>& a, const std::vector<float>& b, float tol = 1e-4f) {
    if (a.size() != b.size()) {
        return false;
    }
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (std::fabs(a[i] - b[i]) > tol) {
            return false;
        }
    }
    return true;
}

}  // namespace

int main() {
    const AttentionConfig cfg{4, 2, 2};
    const int seq_len = 2;
    const int hidden = cfg.hidden_size;

    std::vector<float> h_input = {
        1.f, 2.f, 3.f, 4.f,
        5.f, 6.f, 7.f, 8.f,
    };
    std::vector<float> h_decode = {9.f, 10.f, 11.f, 12.f};

    std::vector<std::uint16_t> identity(hidden * hidden, float_to_bf16(0.0f));
    for (int i = 0; i < hidden; ++i) {
        identity[i * hidden + i] = float_to_bf16(1.0f);
    }
    std::vector<std::uint16_t> rms_weight(hidden, float_to_bf16(1.0f));
    std::vector<float> rms_weight_float(hidden, 1.0f);

    float *d_input = nullptr, *d_output = nullptr;
    float *d_decode_in = nullptr, *d_decode_out = nullptr;
    std::uint16_t *d_wq = nullptr, *d_wk = nullptr, *d_wv = nullptr, *d_wo = nullptr;
    std::uint16_t *d_wgate = nullptr, *d_wup = nullptr, *d_wdown = nullptr;
    std::uint16_t *d_rms1 = nullptr, *d_rms2 = nullptr;

    CUDA_CHECK(cudaMalloc(&d_input, h_input.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, h_input.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_decode_in, h_decode.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_decode_out, h_decode.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_wq, identity.size() * sizeof(std::uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_wk, identity.size() * sizeof(std::uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_wv, identity.size() * sizeof(std::uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_wo, identity.size() * sizeof(std::uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_wgate, identity.size() * sizeof(std::uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_wup, identity.size() * sizeof(std::uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_wdown, identity.size() * sizeof(std::uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_rms1, hidden * sizeof(std::uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_rms2, hidden * sizeof(std::uint16_t)));

    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_decode_in, h_decode.data(), h_decode.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_wq, identity.data(), identity.size() * sizeof(std::uint16_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_wk, identity.data(), identity.size() * sizeof(std::uint16_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_wv, identity.data(), identity.size() * sizeof(std::uint16_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_wo, identity.data(), identity.size() * sizeof(std::uint16_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_wgate, identity.data(), identity.size() * sizeof(std::uint16_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_wup, identity.data(), identity.size() * sizeof(std::uint16_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_wdown, identity.data(), identity.size() * sizeof(std::uint16_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rms1, rms_weight.data(), hidden * sizeof(std::uint16_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rms2, rms_weight.data(), hidden * sizeof(std::uint16_t), cudaMemcpyHostToDevice));

    LayerWeights weights{d_wq, d_wk, d_wv, d_wo, d_wgate, d_wup, d_wdown, d_rms1, d_rms2};
    KVCache cache;
    cache.init(8, cfg.num_heads, cfg.head_dim);

    prefill_forward(d_input, d_output, weights, seq_len, cfg, &cache);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> gpu_prefill(h_input.size());
    CUDA_CHECK(cudaMemcpy(gpu_prefill.data(), d_output, gpu_prefill.size() * sizeof(float), cudaMemcpyDeviceToHost));

    auto cpu_norm_attn = cpu_rms_norm(h_input, seq_len, hidden, cfg.rms_norm_eps, rms_weight_float);
std::vector<float> cpu_attn(h_input.size());
cpu_causal_attention(cpu_norm_attn.data(), cpu_norm_attn.data(), cpu_norm_attn.data(),
                     cpu_attn.data(), seq_len, cfg.num_heads, cfg.head_dim);
std::vector<float> cpu_resid(h_input.size());
for (std::size_t i = 0; i < cpu_resid.size(); ++i) {
    cpu_resid[i] = h_input[i] + cpu_attn[i];
}
    auto cpu_norm_mlp = cpu_rms_norm(cpu_resid, seq_len, hidden, cfg.rms_norm_eps, rms_weight_float);
std::vector<float> cpu_gate = cpu_norm_mlp;
for (float& v : cpu_gate) {
    v = v / (1.0f + std::exp(-v));
}
std::vector<float> cpu_gate_mul(cpu_gate.size());
for (std::size_t i = 0; i < cpu_gate.size(); ++i) {
    cpu_gate_mul[i] = cpu_gate[i] * cpu_norm_mlp[i];
}
std::vector<float> cpu_prefill_block(cpu_gate_mul.size());
for (std::size_t i = 0; i < cpu_prefill_block.size(); ++i) {
    cpu_prefill_block[i] = cpu_resid[i] + cpu_gate_mul[i];
}

if (!nearly_equal(gpu_prefill, cpu_prefill_block)) {
    ATTN_LOGE("Prefill mismatch");
    return 1;
}

    decode_forward(d_decode_in, d_decode_out, weights, cfg, &cache);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> gpu_decode(h_decode.size());
    CUDA_CHECK(cudaMemcpy(gpu_decode.data(), d_decode_out, gpu_decode.size() * sizeof(float), cudaMemcpyDeviceToHost));

std::vector<float> all_tokens = h_input;
all_tokens.insert(all_tokens.end(), h_decode.begin(), h_decode.end());
const int total_tokens = seq_len + 1;
    auto norm_attn_all = cpu_rms_norm(all_tokens, total_tokens, hidden, cfg.rms_norm_eps, rms_weight_float);
std::vector<float> attn_all(total_tokens * hidden);
cpu_causal_attention(norm_attn_all.data(), norm_attn_all.data(), norm_attn_all.data(),
                     attn_all.data(), total_tokens, cfg.num_heads, cfg.head_dim);
std::vector<float> resid_all(total_tokens * hidden);
for (std::size_t i = 0; i < resid_all.size(); ++i) {
    resid_all[i] = all_tokens[i] + attn_all[i];
}
    auto norm_mlp_all = cpu_rms_norm(resid_all, total_tokens, hidden, cfg.rms_norm_eps, rms_weight_float);
std::vector<float> gate_all = norm_mlp_all;
for (float& v : gate_all) {
    v = v / (1.0f + std::exp(-v));
}
std::vector<float> gate_mul_all(gate_all.size());
for (std::size_t i = 0; i < gate_mul_all.size(); ++i) {
    gate_mul_all[i] = gate_all[i] * norm_mlp_all[i];
}
std::vector<float> cpu_final(total_tokens * hidden);
for (std::size_t i = 0; i < cpu_final.size(); ++i) {
    cpu_final[i] = resid_all[i] + gate_mul_all[i];
}
std::vector<float> cpu_last(hidden);
std::copy(cpu_final.end() - hidden, cpu_final.end(), cpu_last.begin());

if (!nearly_equal(gpu_decode, cpu_last)) {
    ATTN_LOGE("Decode mismatch");
    return 1;
}

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_decode_in));
    CUDA_CHECK(cudaFree(d_decode_out));
    CUDA_CHECK(cudaFree(d_wq));
    CUDA_CHECK(cudaFree(d_wk));
    CUDA_CHECK(cudaFree(d_wv));
    CUDA_CHECK(cudaFree(d_wo));
    CUDA_CHECK(cudaFree(d_wgate));
    CUDA_CHECK(cudaFree(d_wup));
    CUDA_CHECK(cudaFree(d_wdown));
    CUDA_CHECK(cudaFree(d_rms1));
    CUDA_CHECK(cudaFree(d_rms2));

    return 0;
}
