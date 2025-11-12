#include <cinttypes>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "attention/core.hpp"
#include "custom_engine/utils/logging.h"
#include "custom_engine/utils/model_paths.hpp"
#include "custom_engine/utils/safetensors.hpp"

namespace {

// Read a file into memory. Used for JSON manifests.
std::string read_file(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) return {};
    return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

}  // namespace

// Smoke-test that we can parse the safetensors index and pull bytes from one tensor.
int main() {
    // Ensure placeholder symbol remains referenced while new utilities are added.
    (void)attention::placeholder();

    const auto root = custom_engine::utils::determine_project_root();
    const auto snapshot = custom_engine::utils::resolve_snapshot_dir(root);
    if (snapshot.empty()) {
        ATTN_LOGE("Failed to resolve snapshot directory (expected path: %s)", (root / "weights").string().c_str());
        return 1;
    }

    const std::filesystem::path index_path = snapshot / "model.safetensors.index.json";
    std::string index_json = read_file(index_path);
    if (index_json.empty()) {
        ATTN_LOGE("Index file missing: %s", index_path.string().c_str());
        return 1;
    }

    std::unordered_map<std::string, std::string> weight_map;
    if (!custom_engine::utils::parse_index_weight_map(index_json, weight_map) || weight_map.empty()) {
        ATTN_LOGE("Failed to parse weight_map");
        return 1;
    }

    const auto& entry = *weight_map.begin();
    const std::filesystem::path shard_path = snapshot / entry.second;

    custom_engine::utils::SafeTensorShard shard(shard_path.string());
    if (!shard.load()) {
        ATTN_LOGE("Failed to load shard header: %s", shard_path.string().c_str());
        return 1;
    }

    if (!shard.has(entry.first)) {
        ATTN_LOGE("Tensor %s not present in %s", entry.first.c_str(), shard_path.string().c_str());
        return 1;
    }

    std::vector<std::uint8_t> bytes;
    constexpr std::size_t kMaxBytes = 1 << 20;  // 1 MiB cap to keep tests light.
    if (!shard.read_bytes(entry.first, bytes, kMaxBytes)) {
        ATTN_LOGE("Failed to read bytes for tensor %s", entry.first.c_str());
        return 1;
    }

    std::uint64_t checksum = 0;
    for (std::uint8_t b : bytes) {
        checksum += b;
    }
    ATTN_LOGI("Read %zu bytes from %s (checksum=%" PRIu64 ")",
              bytes.size(), entry.first.c_str(), checksum);
    return 0;
}
