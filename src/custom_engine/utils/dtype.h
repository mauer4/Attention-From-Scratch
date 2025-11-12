#pragma once

#include <cstddef>
#include <string_view>

namespace custom_engine::utils {

// Simple dtype enum that mirrors the labels stored inside safetensors headers
// and Olmo2 config metadata. The UNKNOWN tag allows graceful handling of new dtypes.
enum class DType {
    BF16,
    F16,
    F32,
    F64,
    I8,
    I16,
    I32,
    I64,
    U8,
    BOOL,
    UNKNOWN,
};

// Return the byte-width for the provided dtype. Unknown dtypes map to zero so
// callers can bail out without undefined behavior.
inline constexpr std::size_t dtype_size(DType dt) {
    switch (dt) {
        case DType::BF16:
        case DType::F16:
        case DType::I16:
            return 2;
        case DType::F32:
        case DType::I32:
            return 4;
        case DType::F64:
        case DType::I64:
            return 8;
        case DType::I8:
        case DType::U8:
        case DType::BOOL:
            return 1;
        default:
            return 0;
    }
}

// Provide a stable string name suitable for logging or manifest generation.
inline constexpr const char* dtype_name(DType dt) {
    switch (dt) {
        case DType::BF16: return "BF16";
        case DType::F16:  return "F16";
        case DType::F32:  return "F32";
        case DType::F64:  return "F64";
        case DType::I8:   return "I8";
        case DType::I16:  return "I16";
        case DType::I32:  return "I32";
        case DType::I64:  return "I64";
        case DType::U8:   return "U8";
        case DType::BOOL: return "BOOL";
        default:          return "UNKNOWN";
    }
}

// Convert safetensors/JSON dtype strings into the enum. Both uppercase labels
// and lowercase friendly names are accepted to match config variations.
inline DType dtype_from_string(std::string_view s) {
    if (s == "BF16" || s == "bfloat16") return DType::BF16;
    if (s == "F16"  || s == "float16")  return DType::F16;
    if (s == "F32"  || s == "float32")  return DType::F32;
    if (s == "F64"  || s == "float64")  return DType::F64;
    if (s == "I8"   || s == "int8")     return DType::I8;
    if (s == "U8"   || s == "uint8")    return DType::U8;
    if (s == "I16"  || s == "int16")    return DType::I16;
    if (s == "I32"  || s == "int32")    return DType::I32;
    if (s == "I64"  || s == "int64")    return DType::I64;
    if (s == "BOOL" || s == "bool")     return DType::BOOL;
    return DType::UNKNOWN;
}

}  // namespace custom_engine::utils
