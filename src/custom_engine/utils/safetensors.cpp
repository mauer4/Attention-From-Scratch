#include "custom_engine/utils/safetensors.hpp"

#include <algorithm>
#include <cctype>
#include <cinttypes>
#include <cstdio>
#include <fstream>
#include <limits>
#include <string>

#include "custom_engine/utils/logging.h"

namespace custom_engine::utils {

namespace {

// Advance past ASCII whitespace inside header JSON.
void skip_ws(const char*& p, const char* end) {
    while (p < end && std::isspace(static_cast<unsigned char>(*p))) {
        ++p;
    }
}

// Parse the next JSON string literal into out.
bool parse_string(const char*& p, const char* end, std::string& out) {
    skip_ws(p, end);
    if (p >= end || *p != '"') return false;
    ++p;
    out.clear();
    while (p < end) {
        char c = *p++;
        if (c == '"') return true;
        if (c == '\\') {
            if (p >= end) return false;
            char esc = *p++;
            switch (esc) {
                case '"': out.push_back('"'); break;
                case '\\': out.push_back('\\'); break;
                case '/': out.push_back('/'); break;
                case 'b': out.push_back('\b'); break;
                case 'f': out.push_back('\f'); break;
                case 'n': out.push_back('\n'); break;
                case 'r': out.push_back('\r'); break;
                case 't': out.push_back('\t'); break;
                default:
                    // Minimal parser: treat other escapes literally.
                    out.push_back(esc);
                    break;
            }
        } else {
            out.push_back(c);
        }
    }
    return false;
}

// Parse an unsigned integer value.
bool parse_uint64(const char*& p, const char* end, uint64_t& out) {
    skip_ws(p, end);
    if (p >= end || *p < '0' || *p > '9') return false;
    uint64_t value = 0;
    while (p < end && *p >= '0' && *p <= '9') {
        uint64_t digit = static_cast<uint64_t>(*p - '0');
        if (value > (std::numeric_limits<uint64_t>::max() - digit) / 10) return false;
        value = value * 10 + digit;
        ++p;
    }
    out = value;
    return true;
}

// Parse a signed integer value.
bool parse_int64(const char*& p, const char* end, int64_t& out) {
    skip_ws(p, end);
    bool neg = false;
    if (p < end && *p == '-') {
        neg = true;
        ++p;
    }
    uint64_t mag = 0;
    if (!parse_uint64(p, end, mag)) return false;
    if (neg) {
        if (mag > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) + 1) return false;
        out = -static_cast<int64_t>(mag);
    } else {
        if (mag > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) return false;
        out = static_cast<int64_t>(mag);
    }
    return true;
}

// Parse an array of signed integers (used for tensor shapes).
bool parse_array_int64(const char*& p, const char* end, std::vector<int64_t>& out) {
    skip_ws(p, end);
    if (p >= end || *p != '[') return false;
    ++p;
    out.clear();
    skip_ws(p, end);
    if (p < end && *p == ']') {
        ++p;
        return true;
    }
    while (p < end) {
        int64_t value = 0;
        if (!parse_int64(p, end, value)) return false;
        out.push_back(value);
        skip_ws(p, end);
        if (p >= end) return false;
        if (*p == ',') {
            ++p;
            continue;
        }
        if (*p == ']') {
            ++p;
            return true;
        }
        return false;
    }
    return false;
}

// Parse two unsigned integers inside an array describing [start, end] offsets.
bool parse_array_offsets(const char*& p, const char* end, uint64_t& start, uint64_t& finish) {
    skip_ws(p, end);
    if (p >= end || *p != '[') return false;
    ++p;
    if (!parse_uint64(p, end, start)) return false;
    skip_ws(p, end);
    if (p >= end || *p != ',') return false;
    ++p;
    if (!parse_uint64(p, end, finish)) return false;
    skip_ws(p, end);
    if (p >= end || *p != ']') return false;
    ++p;
    return true;
}

bool skip_value(const char*& p, const char* end);

// Ignore a JSON string value (used when skipping metadata).
bool skip_string_value(const char*& p, const char* end) {
    std::string ignored;
    return parse_string(p, end, ignored);
}

// Ignore a primitive literal (number, true/false/null).
bool skip_literal(const char*& p, const char* end) {
    while (p < end && *p != ',' && *p != ']' && *p != '}' && *p != '\n' && *p != '\r') {
        ++p;
    }
    return true;
}

// Skip an arbitrary JSON array.
bool skip_array(const char*& p, const char* end) {
    if (p >= end || *p != '[') return false;
    int depth = 0;
    do {
        if (*p == '[') {
            ++depth;
            ++p;
        } else if (*p == ']') {
            --depth;
            ++p;
        } else if (*p == '"') {
            if (!skip_string_value(p, end)) return false;
        } else {
            ++p;
        }
    } while (p < end && depth > 0);
    return depth == 0;
}

// Skip an arbitrary JSON object, honoring nested strings.
bool skip_object(const char*& p, const char* end) {
    if (p >= end || *p != '{') return false;
    int depth = 0;
    do {
        if (*p == '{') {
            ++depth;
            ++p;
        } else if (*p == '}') {
            --depth;
            ++p;
        } else if (*p == '"') {
            if (!skip_string_value(p, end)) return false;
        } else {
            ++p;
        }
    } while (p < end && depth > 0);
    return depth == 0;
}

// Dispatch helper that skips any JSON value.
bool skip_value(const char*& p, const char* end) {
    skip_ws(p, end);
    if (p >= end) return false;
    if (*p == '{') return skip_object(p, end);
    if (*p == '[') return skip_array(p, end);
    if (*p == '"') return skip_string_value(p, end);
    return skip_literal(p, end);
}

// Parse the inner metadata object for a single tensor.
bool parse_tensor_object(const char*& p, const char* end, TensorInfo& info) {
    if (p >= end || *p != '{') return false;
    ++p;
    bool got_dtype = false;
    bool got_shape = false;
    bool got_offsets = false;

    skip_ws(p, end);
    while (p < end && *p != '}') {
        std::string key;
        if (!parse_string(p, end, key)) return false;
        skip_ws(p, end);
        if (p >= end || *p != ':') return false;
        ++p;
        skip_ws(p, end);

        if (key == "dtype") {
            std::string dtype_str;
            if (!parse_string(p, end, dtype_str)) return false;
            info.dtype_str = dtype_str;
            info.dtype = dtype_from_string(dtype_str);
            got_dtype = true;
        } else if (key == "shape") {
            if (!parse_array_int64(p, end, info.shape)) return false;
            got_shape = true;
        } else if (key == "data_offsets") {
            uint64_t s = 0, e = 0;
            if (!parse_array_offsets(p, end, s, e)) return false;
            info.data_start = s;
            info.data_end = e;
            got_offsets = true;
        } else {
            if (!skip_value(p, end)) return false;
        }

        skip_ws(p, end);
        if (p < end && *p == ',') {
            ++p;
            skip_ws(p, end);
        }
    }

    if (p >= end || *p != '}') return false;
    ++p;
    return got_dtype && got_shape && got_offsets;
}

// Parse the top-level safetensors header JSON into tensor metadata.
bool parse_safetensors_root(const char*& p, const char* end,
                            std::unordered_map<std::string, TensorInfo>& tensors) {
    skip_ws(p, end);
    if (p >= end || *p != '{') return false;
    ++p;
    skip_ws(p, end);
    if (p < end && *p == '}') {
        ++p;
        return true;
    }

    while (p < end) {
        std::string tensor_name;
        if (!parse_string(p, end, tensor_name)) return false;
        skip_ws(p, end);
        if (p >= end || *p != ':') return false;
        ++p;
        skip_ws(p, end);

        if (tensor_name == "__metadata__") {
            if (!skip_value(p, end)) return false;
        } else {
            TensorInfo info;
            info.name = tensor_name;
            if (!parse_tensor_object(p, end, info)) return false;
            tensors.emplace(info.name, std::move(info));
        }

        skip_ws(p, end);
        if (p < end && *p == ',') {
            ++p;
            skip_ws(p, end);
            continue;
        }
        if (p < end && *p == '}') {
            ++p;
            return true;
        }
        return false;
    }
    return false;
}

// Parse the "weight_map" object from the safetensors index JSON.
bool parse_weight_map_object(const char*& p, const char* end,
                             std::unordered_map<std::string, std::string>& out) {
    skip_ws(p, end);
    if (p >= end || *p != '{') return false;
    ++p;
    skip_ws(p, end);
    if (p < end && *p == '}') {
        ++p;
        return true;
    }

    while (p < end) {
        std::string key;
        if (!parse_string(p, end, key)) return false;
        skip_ws(p, end);
        if (p >= end || *p != ':') return false;
        ++p;
        skip_ws(p, end);
        std::string value;
        if (!parse_string(p, end, value)) return false;
        out.emplace(std::move(key), std::move(value));
        skip_ws(p, end);
        if (p < end && *p == ',') {
            ++p;
            skip_ws(p, end);
            continue;
        }
        if (p < end && *p == '}') {
            ++p;
            return true;
        }
        return false;
    }
    return false;
}

}  // namespace

// Construct a shard reader bound to a specific on-disk file path.
SafeTensorShard::SafeTensorShard(std::string path) : path_(std::move(path)) {}

// Convert raw JSON text into the tensor metadata map.
bool SafeTensorShard::parse_header_(const std::string& json) {
    const char* cursor = json.data();
    const char* end = cursor + json.size();
    tensors_.clear();
    return parse_safetensors_root(cursor, end, tensors_);
}

// Load and parse the safetensors header from disk.
bool SafeTensorShard::load() {
    loaded_ = false;
    header_len_ = 0;
    tensors_.clear();

    std::ifstream file(path_, std::ios::binary);
    if (!file) {
        ATTN_LOGE("Failed to open shard: %s", path_.c_str());
        return false;
    }

    std::uint8_t len_bytes[8] = {0};
    file.read(reinterpret_cast<char*>(len_bytes), 8);
    if (!file) {
        ATTN_LOGE("Failed to read header length for %s", path_.c_str());
        return false;
    }

    uint64_t header_len = 0;
    for (int i = 7; i >= 0; --i) {
        header_len = (header_len << 8) | static_cast<uint64_t>(len_bytes[i]);
    }
    header_len_ = header_len;

    std::string header;
    header.resize(static_cast<std::size_t>(header_len));
    file.read(header.data(), static_cast<std::streamsize>(header_len));
    if (!file) {
        ATTN_LOGE("Failed to read header JSON for %s", path_.c_str());
        return false;
    }

    if (!parse_header_(header)) {
        ATTN_LOGE("Failed to parse safetensors header for %s", path_.c_str());
        return false;
    }

    loaded_ = true;
    return true;
}

// Return true if header metadata contains the requested tensor.
bool SafeTensorShard::has(const std::string& tensor_name) const {
    return tensors_.find(tensor_name) != tensors_.end();
}

// Fetch tensor metadata pointer if present.
const TensorInfo* SafeTensorShard::find(const std::string& tensor_name) const {
    auto it = tensors_.find(tensor_name);
    if (it == tensors_.end()) return nullptr;
    return &it->second;
}

// Copy up to max_bytes of tensor payload into the provided vector.
bool SafeTensorShard::read_bytes(const std::string& tensor_name, std::vector<std::uint8_t>& out,
                                 std::size_t max_bytes) const {
    if (!loaded_) return false;
    auto it = tensors_.find(tensor_name);
    if (it == tensors_.end()) return false;

    const TensorInfo& info = it->second;
    uint64_t nbytes64 = info.nbytes();
    std::size_t to_read = static_cast<std::size_t>(std::min<uint64_t>(nbytes64, SIZE_MAX));
    if (max_bytes != SIZE_MAX && max_bytes < to_read) {
        to_read = max_bytes;
    }
    out.resize(to_read);

    std::ifstream file(path_, std::ios::binary);
    if (!file) return false;
    uint64_t offset = 8 + header_len_ + info.data_start;
    file.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    if (!file) return false;
    file.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(to_read));
    return static_cast<std::size_t>(file.gcount()) == to_read;
}

// Populate tensor -> shard filename map from the model index JSON.
bool parse_index_weight_map(const std::string& json,
                            std::unordered_map<std::string, std::string>& out) {
    const char* cursor = json.data();
    const char* end = cursor + json.size();
    skip_ws(cursor, end);
    if (cursor >= end || *cursor != '{') return false;
    ++cursor;

    while (cursor < end) {
        skip_ws(cursor, end);
        if (cursor < end && *cursor == '}') {
            ++cursor;
            return !out.empty();
        }

        std::string key;
        if (!parse_string(cursor, end, key)) return false;
        skip_ws(cursor, end);
        if (cursor >= end || *cursor != ':') return false;
        ++cursor;
        skip_ws(cursor, end);

        if (key == "weight_map") {
            if (!parse_weight_map_object(cursor, end, out)) return false;
        } else {
            if (!skip_value(cursor, end)) return false;
        }

        skip_ws(cursor, end);
        if (cursor < end && *cursor == ',') {
            ++cursor;
            continue;
        }
    }
    return !out.empty();
}

}  // namespace custom_engine::utils
