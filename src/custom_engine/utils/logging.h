#pragma once

#include <cstdarg>
#include <cstdio>
#include <cstdlib>

namespace custom_engine::utils {

// Low-level helper that prints a formatted log line with a severity label.
inline void vlogf(const char* level, const char* fmt, va_list ap) {
    std::fprintf(stderr, "[%s] ", level);
    std::vfprintf(stderr, fmt, ap);
    std::fputc('\n', stderr);
    std::fflush(stderr);
}

// Emit an INFO log. Accepts standard printf-style arguments.
inline void log_info(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vlogf("INFO", fmt, ap);
    va_end(ap);
}

// Emit an ERROR log. Accepts standard printf-style arguments.
inline void log_error(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vlogf("ERROR", fmt, ap);
    va_end(ap);
}

}  // namespace custom_engine::utils

// Convenience macros so call sites stay clean.
#define ATTN_LOGI(...) ::custom_engine::utils::log_info(__VA_ARGS__)
#define ATTN_LOGE(...) ::custom_engine::utils::log_error(__VA_ARGS__)

// Abort the process when a condition fails, logging the location for debugging.
#define ATTN_CHECK(cond)                                                         \
    do {                                                                         \
        if (!(cond)) {                                                           \
            ATTN_LOGE("Check failed: %s (%s:%d)", #cond, __FILE__, __LINE__);    \
            std::abort();                                                        \
        }                                                                        \
    } while (0)
