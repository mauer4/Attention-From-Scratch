#include "custom_engine/v1/tokenizer_adapter.h"

#include "custom_engine/utils/logging.h"

namespace custom_engine::v1 {

// Store user callbacks/context for later encode/decode invocations.
TokenizerBridge::TokenizerBridge(void* user_ctx, EncodeCallback encode_cb, DecodeCallback decode_cb)
    : ctx_(user_ctx), encode_cb_(encode_cb), decode_cb_(decode_cb) {}

// Convert UTF-8 text into token ids via the registered callback.
std::vector<int> TokenizerBridge::encode(std::string_view text) {
    if (!encode_cb_) {
        ATTN_LOGE("Tokenizer encode callback not set");
        return {};
    }
    return encode_cb_(text.data(), text.size(), ctx_);
}

// Convert token ids back into text via the registered callback.
std::string TokenizerBridge::decode(const std::vector<int>& ids) {
    if (!decode_cb_) {
        ATTN_LOGE("Tokenizer decode callback not set");
        return {};
    }
    return decode_cb_(ids.data(), ids.size(), ctx_);
}

// Convenience factory for callers that prefer raw pointers over std::unique_ptr for now.
ITokenizer* make_tokenizer_bridge(void* ctx, EncodeCallback encode_cb, DecodeCallback decode_cb) {
    return new TokenizerBridge(ctx, encode_cb, decode_cb);
}

}  // namespace custom_engine::v1
