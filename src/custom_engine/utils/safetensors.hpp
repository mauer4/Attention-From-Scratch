#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "custom_engine/utils/dtype.h"

namespace custom_engine::utils {

// Description of a tensor stored inside a safetensors shard header.
struct TensorInfo {
    std::string name;
    std::string dtype_str;
    DType dtype = DType::UNKNOWN;
    std::vector<int64_t> shape;
    uint64_t data_start = 0;  // relative to tensor data section (after 8 + header_len)
    uint64_t data_end = 0;    // exclusive

    uint64_t nbytes() const { return data_end > data_start ? (data_end - data_start) : 0; }
};

// Lightweight reader that parses a shard header and exposes tensor metadata
// plus bounded byte reads for correctness checks.
class SafeTensorShard {
public:
    SafeTensorShard() = default;
    explicit SafeTensorShard(std::string path);

    // Read the shard header from disk. Returns true on success.
    bool load();
    bool valid() const { return loaded_; }
    const std::string& path() const { return path_; }
    uint64_t header_len() const { return header_len_; }

    bool has(const std::string& tensor_name) const;
    const TensorInfo* find(const std::string& tensor_name) const;
    const std::unordered_map<std::string, TensorInfo>& tensors() const { return tensors_; }

    // Copy up to max_bytes of raw tensor payload into out. Useful for manual
    // validation or converting weights outside of a framework.
    bool read_bytes(const std::string& tensor_name, std::vector<std::uint8_t>& out,
                    std::size_t max_bytes = SIZE_MAX) const;

private:
    std::string path_;
    bool loaded_ = false;
    uint64_t header_len_ = 0;
    std::unordered_map<std::string, TensorInfo> tensors_;

    bool parse_header_(const std::string& json);
};

// Parse model.safetensors.index.json -> tensor name -> shard map.
bool parse_index_weight_map(const std::string& json,
                            std::unordered_map<std::string, std::string>& out);

}  // namespace custom_engine::utils
