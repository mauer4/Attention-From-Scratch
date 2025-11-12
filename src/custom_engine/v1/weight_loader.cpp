#include "custom_engine/v1/weight_loader.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>

#include "custom_engine/utils/bf16.h"
#include "custom_engine/utils/logging.h"
#include "custom_engine/utils/safetensors.hpp"

namespace custom_engine::v1 {

namespace {

std::string read_text(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return {};
    }
    std::ostringstream oss;
    oss << file.rdbuf();
    return oss.str();
}

int parse_int_field(const std::string& json, const std::string& key) {
    const std::string needle = '"' + key + '"';
    const auto pos = json.find(needle);
    if (pos == std::string::npos) {
        return 0;
    }
    const auto colon = json.find(':', pos + needle.size());
    if (colon == std::string::npos) {
        return 0;
    }
    const auto end = json.find_first_of(",}\n", colon + 1);
    const auto value = json.substr(colon + 1, end - colon - 1);
    return std::stoi(value);
}

float parse_float_field(const std::string& json, const std::string& key) {
    const std::string needle = '"' + key + '"';
    const auto pos = json.find(needle);
    if (pos == std::string::npos) {
        return 0.0f;
    }
    const auto colon = json.find(':', pos + needle.size());
    if (colon == std::string::npos) {
        return 0.0f;
    }
    const auto end = json.find_first_of(",}\n", colon + 1);
    const auto value = json.substr(colon + 1, end - colon - 1);
    return std::stof(value);
}

}  // namespace

TensorLoader::TensorLoader(std::filesystem::path project_root)
    : project_root_(std::move(project_root)) {
    snapshot_dir_ = custom_engine::utils::resolve_snapshot_dir(project_root_);
    load_index();
    load_config_json();
}

void TensorLoader::load_index() {
    const auto index_path = snapshot_dir_ / "model.safetensors.index.json";
    const std::string json = read_text(index_path);
    if (json.empty()) {
        ATTN_LOGE("Failed to read weight index: %s", index_path.string().c_str());
        return;
    }
    if (!custom_engine::utils::parse_index_weight_map(json, weight_map_) || weight_map_.empty()) {
        ATTN_LOGE("Failed to parse weight_map from %s", index_path.string().c_str());
    }
}

void TensorLoader::load_config_json() {
    const auto path = snapshot_dir_ / "config.json";
    const std::string json = read_text(path);
    config_.hidden_size = parse_int_field(json, "hidden_size");
    config_.num_heads = parse_int_field(json, "num_attention_heads");
    config_.num_kv_heads = parse_int_field(json, "num_key_value_heads");
    config_.intermediate_size = parse_int_field(json, "intermediate_size");
    config_.rms_norm_eps = parse_float_field(json, "rms_norm_eps");
    if (config_.rms_norm_eps == 0.0f) {
        config_.rms_norm_eps = 1e-6f;
    }
}

custom_engine::utils::SafeTensorShard& TensorLoader::ensure_shard(const std::string& shard_name) {
    auto it = shards_.find(shard_name);
    if (it == shards_.end()) {
        custom_engine::utils::SafeTensorShard shard((snapshot_dir_ / shard_name).string());
        shard.load();
        it = shards_.emplace(shard_name, std::move(shard)).first;
    }
    return it->second;
}

const custom_engine::utils::TensorInfo* TensorLoader::tensor_info(const std::string& name) const {
    auto it = weight_map_.find(name);
    if (it == weight_map_.end()) {
        return nullptr;
    }
    auto shard_it = shards_.find(it->second);
    if (shard_it == shards_.end()) {
        auto* self = const_cast<TensorLoader*>(this);
        auto& shard = self->ensure_shard(it->second);
        return shard.find(name);
    }
    return shard_it->second.find(name);
}

std::vector<float> TensorLoader::load_tensor(const std::string& name) {
    auto map_it = weight_map_.find(name);
    if (map_it == weight_map_.end()) {
        ATTN_LOGE("Tensor %s not found in weight map", name.c_str());
        return {};
    }
    auto& shard = ensure_shard(map_it->second);
    const auto* info = shard.find(name);
    if (!info) {
        ATTN_LOGE("Tensor %s missing in shard %s", name.c_str(), map_it->second.c_str());
        return {};
    }
    std::vector<std::uint8_t> raw;
    shard.read_bytes(name, raw);
    std::size_t numel = 1;
    for (int64_t dim : info->shape) {
        numel *= static_cast<std::size_t>(dim);
    }
    std::vector<float> result(numel);
    if (info->dtype == custom_engine::utils::DType::F32) {
        auto* data32 = reinterpret_cast<const float*>(raw.data());
        std::copy(data32, data32 + numel, result.data());
    } else {
        ATTN_LOGE("Unsupported dtype for tensor %s when requesting float view", name.c_str());
        result.clear();
    }
    return result;
}

bool TensorLoader::copy_tensor_to_device(const std::string& name,
                                         custom_engine::utils::DeviceBuffer<float>& dst,
                                         cudaStream_t stream) {
    std::vector<float> host = load_tensor(name);
    if (host.empty()) {
        return false;
    }
    dst = custom_engine::utils::DeviceBuffer<float>(host.size(), stream);
    CUDA_CHECK(cudaMemcpyAsync(dst.get(), host.data(), host.size() * sizeof(float),
                               cudaMemcpyHostToDevice, stream));
    return true;
}

bool TensorLoader::copy_tensor_to_device_bf16(const std::string& name,
                                              custom_engine::utils::DeviceBuffer<std::uint16_t>& dst,
                                              cudaStream_t stream) {
    auto map_it = weight_map_.find(name);
    if (map_it == weight_map_.end()) {
        ATTN_LOGE("Tensor %s not found in weight map", name.c_str());
        return false;
    }
    auto& shard = ensure_shard(map_it->second);
    const auto* info = shard.find(name);
    if (!info) {
        ATTN_LOGE("Tensor %s missing in shard %s", name.c_str(), map_it->second.c_str());
        return false;
    }
    if (info->dtype != custom_engine::utils::DType::BF16) {
        ATTN_LOGE("Tensor %s is not BF16", name.c_str());
        return false;
    }
    std::vector<std::uint8_t> raw;
    if (!shard.read_bytes(name, raw)) {
        ATTN_LOGE("Failed to read tensor %s", name.c_str());
        return false;
    }
    const std::size_t numel = info->nbytes() / sizeof(std::uint16_t);
    dst = custom_engine::utils::DeviceBuffer<std::uint16_t>(numel, stream);
    CUDA_CHECK(cudaMemcpyAsync(dst.get(), raw.data(), numel * sizeof(std::uint16_t),
                               cudaMemcpyHostToDevice, stream));
    return true;
}

bool LayerWeightsDevice::load_layer(TensorLoader& loader, int layer_index, cudaStream_t stream) {
    auto tensor_name = [layer_index](const std::string& suffix) {
        std::ostringstream oss;
        oss << "model.layers." << layer_index << suffix;
        return oss.str();
    };

    if (!loader.copy_tensor_to_device_bf16(tensor_name(".self_attn.q_proj.weight"), w_q_, stream)) return false;
    if (!loader.copy_tensor_to_device_bf16(tensor_name(".self_attn.k_proj.weight"), w_k_, stream)) return false;
    if (!loader.copy_tensor_to_device_bf16(tensor_name(".self_attn.v_proj.weight"), w_v_, stream)) return false;
    if (!loader.copy_tensor_to_device_bf16(tensor_name(".self_attn.o_proj.weight"), w_o_, stream)) return false;
    if (!loader.copy_tensor_to_device_bf16(tensor_name(".mlp.gate_proj.weight"), w_gate_, stream)) return false;
    if (!loader.copy_tensor_to_device_bf16(tensor_name(".mlp.up_proj.weight"), w_up_, stream)) return false;
    if (!loader.copy_tensor_to_device_bf16(tensor_name(".mlp.down_proj.weight"), w_down_, stream)) return false;
    if (!loader.copy_tensor_to_device_bf16(tensor_name(".post_attention_layernorm.weight"), rms_attn_, stream)) return false;
    if (!loader.copy_tensor_to_device_bf16(tensor_name(".post_feedforward_layernorm.weight"), rms_mlp_, stream)) return false;
    return true;
}

LayerWeights LayerWeightsDevice::view() const {
    LayerWeights view{};
    view.w_q = w_q_.get();
    view.w_k = w_k_.get();
    view.w_v = w_v_.get();
    view.w_o = w_o_.get();
    view.w_gate = w_gate_.get();
    view.w_up = w_up_.get();
    view.w_down = w_down_.get();
    view.rms_attn = rms_attn_.get();
    view.rms_mlp = rms_mlp_.get();
    return view;
}

}  // namespace custom_engine::v1
