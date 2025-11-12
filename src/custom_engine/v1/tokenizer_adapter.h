#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace custom_engine::v1 {

// Minimal tokenizer interface so the engine can remain agnostic to how tokens
// are produced/consumed (Python bridge, C API, etc.).
class ITokenizer {
public:
    virtual ~ITokenizer() = default;
    virtual std::vector<int> encode(std::string_view text) = 0;
    virtual std::string decode(const std::vector<int>& ids) = 0;
};

using EncodeCallback = std::vector<int> (*)(const char*, std::size_t, void*);
using DecodeCallback = std::string (*)(const int*, std::size_t, void*);

// Adapter that forwards encode/decode to user-provided callbacks.
class TokenizerBridge final : public ITokenizer {
public:
    TokenizerBridge(void* user_ctx, EncodeCallback encode_cb, DecodeCallback decode_cb);

    std::vector<int> encode(std::string_view text) override;
    std::string decode(const std::vector<int>& ids) override;

private:
    void* ctx_ = nullptr;
    EncodeCallback encode_cb_ = nullptr;
    DecodeCallback decode_cb_ = nullptr;
};

// Helper for constructing the bridge while keeping ownership simple.
ITokenizer* make_tokenizer_bridge(void* ctx, EncodeCallback encode_cb, DecodeCallback decode_cb);

}  // namespace custom_engine::v1
