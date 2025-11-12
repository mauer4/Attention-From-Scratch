#include <cmath>
#include <numeric>
#include <string>
#include <vector>

#include "custom_engine/utils/cuda_utils.cuh"
#include "custom_engine/utils/logging.h"
#include "custom_engine/v1/prefill_decode.hpp"
#include "custom_engine/v1/weight_loader.hpp"

int main(int argc, char** argv) {
    bool verbose = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string_view(argv[i]) == "--verbose") {
            verbose = true;
        }
    }

    custom_engine::v1::TensorLoader loader;
    custom_engine::v1::LayerWeightsDevice device_weights;
    if (!device_weights.load_layer(loader, 0)) {
        ATTN_LOGE("Failed to load layer weights");
        return 1;
    }
    const auto cfg_model = loader.model_config();
    if (cfg_model.hidden_size <= 0) {
        ATTN_LOGE("Model config missing valid hidden_size");
        return 1;
    }
    if (cfg_model.num_heads <= 0) {
        ATTN_LOGE("Model config missing valid num_heads");
        return 1;
    }
    if (cfg_model.hidden_size % cfg_model.num_heads != 0) {
        ATTN_LOGE("hidden_size (%d) not divisible by num_heads (%d)", cfg_model.hidden_size, cfg_model.num_heads);
        return 1;
    }
    if (cfg_model.rms_norm_eps <= 0.0f) {
        ATTN_LOGE("Model config missing valid rms_norm_eps");
        return 1;
    }

    if (verbose) {
        ATTN_LOGI("Model config: hidden_size=%d, num_heads=%d, num_kv_heads=%d, "
                  "intermediate_size=%d, rms_norm_eps=%g",
                  cfg_model.hidden_size,
                  cfg_model.num_heads,
                  cfg_model.num_kv_heads,
                  cfg_model.intermediate_size,
                  cfg_model.rms_norm_eps);
    }

    const int head_dim = cfg_model.hidden_size / cfg_model.num_heads;
    custom_engine::v1::AttentionConfig cfg{cfg_model.hidden_size, cfg_model.num_heads, head_dim, cfg_model.rms_norm_eps};

    custom_engine::v1::LayerWeights weights = device_weights.view();

    custom_engine::v1::KVCache cache;
    cache.init(8, cfg.num_heads, cfg.head_dim);

    std::vector<float> host_input(cfg.hidden_size);
    for (int i = 0; i < cfg.hidden_size; ++i) {
        host_input[i] = static_cast<float>((i % 23) - 11) / 17.0f;
    }
    std::vector<float> host_decode(cfg.hidden_size);
    for (int i = 0; i < cfg.hidden_size; ++i) {
        host_decode[i] = static_cast<float>((i % 19) - 9) / 13.0f;
    }

    float *d_input = nullptr, *d_output = nullptr;
    float *d_decode_in = nullptr, *d_decode_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, host_input.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, host_input.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_decode_in, host_decode.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_decode_out, host_decode.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, host_input.data(), host_input.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_decode_in, host_decode.data(), host_decode.size() * sizeof(float), cudaMemcpyHostToDevice));

    if (verbose) {
        ATTN_LOGI("Beginning layer0 prefill/decode run with seq_len=1 | hidden=%d | heads=%d | head_dim=%d",
                  cfg.hidden_size,
                  cfg.num_heads,
                  cfg.head_dim);
    }

    prefill_forward(d_input, d_output, weights, 1, cfg, &cache);
    decode_forward(d_decode_in, d_decode_out, weights, cfg, &cache);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> host_out(host_input.size());
    CUDA_CHECK(cudaMemcpy(host_out.data(), d_output, host_out.size() * sizeof(float), cudaMemcpyDeviceToHost));
    std::vector<float> host_dec_out(host_decode.size());
    CUDA_CHECK(cudaMemcpy(host_dec_out.data(), d_decode_out, host_dec_out.size() * sizeof(float), cudaMemcpyDeviceToHost));

    auto is_finite = [](const std::vector<float>& data) {
        for (float v : data) {
            if (!std::isfinite(v)) {
                return false;
            }
        }
        return true;
    };

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_decode_in));
    CUDA_CHECK(cudaFree(d_decode_out));

    if (!is_finite(host_out) || !is_finite(host_dec_out)) {
        ATTN_LOGE("Layer0 inference produced non-finite outputs");
        return 1;
    }

    ATTN_LOGI("Layer0 smoke test produced finite outputs (prefill sum=%f, decode sum=%f)",
              std::accumulate(host_out.begin(), host_out.end(), 0.0f),
              std::accumulate(host_dec_out.begin(), host_dec_out.end(), 0.0f));
    return 0;
}
