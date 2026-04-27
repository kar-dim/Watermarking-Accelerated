#pragma once
#include <cuda_runtime.h>

/*!
 *  \brief  GPU Buffer class that manages memory allocation, deallocation, and data transfer between host and device. It supports 1D and 2D arrays with multiple channels, and is designed to work with
 *          CUDA streams for async operations and better performance. It also provides utility methods for common operations like filling with zeros, cloning, and trimming the memory pool. The class
 *          is moveable but not copyable to ensure proper resource management.
 *  \author Dimitris Karatzas
 */
template <typename T>
class GpuArray {
    T* ptr_ = nullptr;
    unsigned int rows_ = 0;
    unsigned int cols_ = 0;
    unsigned int channels_ = 1;
    cudaStream_t stream_ = nullptr;

    void alloc() {
        if (size() > 0)
            cudaMallocAsync(&ptr_, bytes(), stream_);
    }

  public:
    GpuArray() = default;

    explicit GpuArray(unsigned int count, cudaStream_t stream) : rows_(count), cols_(1), stream_(stream) { alloc(); }

    GpuArray(unsigned int rows, unsigned int cols, cudaStream_t stream) : rows_(rows), cols_(cols), stream_(stream) { alloc(); }

    GpuArray(unsigned int rows, unsigned int cols, unsigned int channels, cudaStream_t stream) : rows_(rows), cols_(cols), channels_(channels), stream_(stream) { alloc(); }

    GpuArray(unsigned int rows, unsigned int cols, const T* hostData, cudaStream_t stream) : rows_(rows), cols_(cols), stream_(stream) {
        alloc();
        cudaMemcpyAsync(ptr_, hostData, bytes(), cudaMemcpyHostToDevice, stream_);
    }

    GpuArray(unsigned int rows, unsigned int cols, unsigned int channels, const T* hostData, cudaStream_t stream) : rows_(rows), cols_(cols), channels_(channels), stream_(stream) {
        alloc();
        cudaMemcpyAsync(ptr_, hostData, bytes(), cudaMemcpyHostToDevice, stream_);
    }

    ~GpuArray() {
        if (ptr_)
            cudaFreeAsync(ptr_, stream_);
    }

    GpuArray(const GpuArray&) = delete;
    GpuArray& operator=(const GpuArray&) = delete;

    GpuArray(GpuArray&& o) noexcept : ptr_(o.ptr_), rows_(o.rows_), cols_(o.cols_), channels_(o.channels_), stream_(o.stream_) {
        o.ptr_ = nullptr;
        o.rows_ = o.cols_ = 0;
        o.channels_ = 1;
    }

    GpuArray& operator=(GpuArray&& o) noexcept {
        if (this != &o) {
            if (ptr_)
                cudaFreeAsync(ptr_, stream_);
            ptr_ = o.ptr_;
            rows_ = o.rows_;
            cols_ = o.cols_;
            channels_ = o.channels_;
            stream_ = o.stream_;
            o.ptr_ = nullptr;
            o.rows_ = o.cols_ = 0;
            o.channels_ = 1;
        }
        return *this;
    }

    T* data() { return ptr_; }
    const T* data() const { return ptr_; }
    unsigned int rows() const { return rows_; }
    unsigned int cols() const { return cols_; }
    unsigned int channels() const { return channels_; }
    unsigned int size() const { return rows_ * cols_ * channels_; }
    size_t bytes() const { return static_cast<size_t>(size()) * sizeof(T); }
    bool empty() const { return ptr_ == nullptr; }
    cudaStream_t stream() const { return stream_; }

    void fillZero() {
        if (ptr_)
            cudaMemsetAsync(ptr_, 0, bytes(), stream_);
    }

    T scalar() const {
        T val{};
        if (ptr_) {
            cudaMemcpyAsync(&val, ptr_, sizeof(T), cudaMemcpyDeviceToHost, stream_);
            cudaStreamSynchronize(stream_);
        }
        return val;
    }

    void toHost(T* dst) const {
        if (ptr_) {
            cudaMemcpyAsync(dst, ptr_, bytes(), cudaMemcpyDeviceToHost, stream_);
            cudaStreamSynchronize(stream_);
        }
    }

    void toHostAsync(T* dst) const {
        if (ptr_)
            cudaMemcpyAsync(dst, ptr_, bytes(), cudaMemcpyDeviceToHost, stream_);
    }

    static GpuArray zeros(unsigned int count, cudaStream_t stream) {
        GpuArray arr(count, stream);
        arr.fillZero();
        return arr;
    }

    static GpuArray zeros(unsigned int rows, unsigned int cols, cudaStream_t stream) {
        GpuArray arr(rows, cols, stream);
        arr.fillZero();
        return arr;
    }

    GpuArray clone() const {
        GpuArray copy(rows_, cols_, channels_, stream_);
        if (ptr_)
            cudaMemcpyAsync(copy.ptr_, ptr_, bytes(), cudaMemcpyDeviceToDevice, stream_);
        return copy;
    }

    static void trimMemoryPool() {
        cudaMemPool_t pool;
        cudaDeviceGetDefaultMemPool(&pool, 0);
        cudaMemPoolTrimTo(pool, 0);
    }
};
