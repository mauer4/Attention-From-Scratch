#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

#include "custom_engine/utils/cuda_utils.cuh"
#include "custom_engine/utils/model_paths.hpp"
#include "custom_engine/utils/safetensors.hpp"
#include "custom_engine/v1/prefill_decode.hpp"

namespace custom_engine::v1 {

struct ModelConfig {
    int hidden_size = 0;
    int num_heads = 0;
    int num_kv_heads = 0;
    int intermediate_size = 0;
    float rms_norm_eps = 1e-6f;
};

class TensorLoader {
public:
    explicit TensorLoader(std::filesystem::path project_root = custom_engine::utils::determine_project_root());

    const ModelConfig& model_config() const { return config_; }

    std::vector<float> load_tensor(const std::string& name);
    const custom_engine::utils::TensorInfo* tensor_info(const std::string& name) const;

    bool copy_tensor_to_device(const std::string& name,
                               custom_engine::utils::DeviceBuffer<float>& dst,
                               cudaStream_t stream = nullptr);

    bool copy_tensor_to_device_bf16(const std::string& name,
                                    custom_engine::utils::DeviceBuffer<std::uint16_t>& dst,
                                    cudaStream_t stream = nullptr);

private:
    std::filesystem::path project_root_;
    std::filesystem::path snapshot_dir_;
    std::unordered_map<std::string, std::string> weight_map_;
    std::unordered_map<std::string, custom_engine::utils::SafeTensorShard> shards_;
    ModelConfig config_{};

    void load_index();
    void load_config_json();
    custom_engine::utils::SafeTensorShard& ensure_shard(const std::string& shard_name);
};

class LayerWeightsDevice {
public:
    bool load_layer(TensorLoader& loader, int layer_index, cudaStream_t stream = nullptr);
    LayerWeights view() const;

private:
    custom_engine::utils::DeviceBuffer<std::uint16_t> w_q_;
    custom_engine::utils::DeviceBuffer<std::uint16_t> w_k_;
    custom_engine::utils::DeviceBuffer<std::uint16_t> w_v_;
    custom_engine::utils::DeviceBuffer<std::uint16_t> w_o_;
    custom_engine::utils::DeviceBuffer<std::uint16_t> w_gate_;
    custom_engine::utils::DeviceBuffer<std::uint16_t> w_up_;
    custom_engine::utils::DeviceBuffer<std::uint16_t> w_down_;
    custom_engine::utils::DeviceBuffer<std::uint16_t> rms_attn_;
    custom_engine::utils::DeviceBuffer<std::uint16_t> rms_mlp_;
};

}  // namespace custom_engine::v1
