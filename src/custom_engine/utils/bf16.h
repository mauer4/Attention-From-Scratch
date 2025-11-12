#pragma once

#include <cstdint>
#include <cstring>
#include <vector>

namespace custom_engine::utils {

inline float bf16_to_float(std::uint16_t value) {
    std::uint32_t bits = static_cast<std::uint32_t>(value) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

inline void bf16_buffer_to_float(const std::uint16_t* src, float* dst, std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) {
        dst[i] = bf16_to_float(src[i]);
    }
}

}  // namespace custom_engine::utils

