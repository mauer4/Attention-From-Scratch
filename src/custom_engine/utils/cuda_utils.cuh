#pragma once

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <utility>

// Wrap CUDA API calls and abort with a descriptive log if any call fails.
#define CUDA_CHECK(expr)                                                               \
    do {                                                                               \
        cudaError_t _err = (expr);                                                     \
        if (_err != cudaSuccess) {                                                     \
            std::fprintf(stderr, "CUDA error %s:%d: %s (%d) in %s\n",               \
                         __FILE__, __LINE__, cudaGetErrorString(_err),                \
                         static_cast<int>(_err), #expr);                              \
            std::abort();                                                              \
        }                                                                              \
    } while (0)

namespace custom_engine::utils {

// RAII device allocation that frees asynchronously on destruction. The class
// is move-only so buffers can be transferred between helper routines.
template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;

    // Allocate count elements (bytes = count * sizeof(T)) on the provided stream.
    explicit DeviceBuffer(std::size_t count, cudaStream_t stream = nullptr)
        : ptr_(nullptr), count_(count) {
        if (count_ > 0) {
            CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&ptr_), count_ * sizeof(T), stream));
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept : ptr_(other.ptr_), count_(other.count_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            reset();
            ptr_ = other.ptr_;
            count_ = other.count_;
            other.ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    ~DeviceBuffer() { reset(); }

    // Release backing storage (if any). Safe to call multiple times.
    void reset(cudaStream_t stream = nullptr) {
        if (ptr_) {
            CUDA_CHECK(cudaFreeAsync(ptr_, stream));
            ptr_ = nullptr;
            count_ = 0;
        }
    }

    T* get() const { return ptr_; }
    std::size_t size() const { return count_; }
    std::size_t bytes() const { return count_ * sizeof(T); }

private:
    T* ptr_ = nullptr;
    std::size_t count_ = 0;
};

}  // namespace custom_engine::utils
