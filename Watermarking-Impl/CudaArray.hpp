#pragma once
#include <cuda_runtime.h>

/*!
 *  \brief  GPU Buffer class that manages memory allocation, deallocation, and data transfer between host and device. It supports 1D and 2D arrays with multiple channels, and is designed to work with
 *          CUDA streams for async operations and better performance. It also provides utility methods for common operations like filling with zeros, cloning, and trimming the memory pool. The class
 *          is moveable but not copyable to ensure proper resource management.
 *  \author Dimitris Karatzas
 */
template <typename T>
class CudaArray {
  private:
    T* ptr_ = nullptr;
    unsigned int rows = 0;
    unsigned int cols = 0;
    unsigned int channels = 1;
    cudaStream_t stream = nullptr;

    void alloc() {
        if (size() > 0)
            cudaMallocAsync(&ptr_, bytes(), stream);
    }

  public:
    CudaArray() = default;

    explicit CudaArray(unsigned int count, cudaStream_t stream) : rows(count), cols(1), stream(stream) { alloc(); }

    CudaArray(unsigned int rows, unsigned int cols, cudaStream_t stream) : rows(rows), cols(cols), stream(stream) { alloc(); }

    CudaArray(unsigned int rows, unsigned int cols, unsigned int channels, cudaStream_t stream) : rows(rows), cols(cols), channels(channels), stream(stream) { alloc(); }

    CudaArray(unsigned int rows, unsigned int cols, const T* hostData, cudaStream_t stream) : rows(rows), cols(cols), stream(stream) {
        alloc();
        cudaMemcpyAsync(ptr_, hostData, bytes(), cudaMemcpyHostToDevice, stream);
    }

    CudaArray(unsigned int rows, unsigned int cols, unsigned int channels, const T* hostData, cudaStream_t stream) : rows(rows), cols(cols), channels(channels), stream(stream) {
        alloc();
        cudaMemcpyAsync(ptr_, hostData, bytes(), cudaMemcpyHostToDevice, stream);
    }

    ~CudaArray() {
        if (ptr_)
            cudaFreeAsync(ptr_, stream);
    }

    CudaArray(const CudaArray&) = delete;
    CudaArray& operator=(const CudaArray&) = delete;

    CudaArray(CudaArray&& o) noexcept : ptr_(o.ptr_), rows(o.rows), cols(o.cols), channels(o.channels), stream(o.stream) {
        o.ptr_ = nullptr;
        o.rows = o.cols = 0;
        o.channels = 1;
    }

    CudaArray& operator=(CudaArray&& o) noexcept {
        if (this != &o) {
            if (ptr_)
                cudaFreeAsync(ptr_, stream);
            ptr_ = o.ptr_;
            rows = o.rows;
            cols = o.cols;
            channels = o.channels;
            stream = o.stream;
            o.ptr_ = nullptr;
            o.rows = o.cols = 0;
            o.channels = 1;
        }
        return *this;
    }

    T* data() { return ptr_; }
    const T* data() const { return ptr_; }
    unsigned int getRows() const { return rows; }
    unsigned int getCols() const { return cols; }
    unsigned int getChannels() const { return channels; }
    unsigned int size() const { return rows * cols * channels; }
    size_t bytes() const { return static_cast<size_t>(size()) * sizeof(T); }
    bool empty() const { return ptr_ == nullptr; }
    cudaStream_t getStream() const { return stream; }

    void fillZero() {
        if (ptr_)
            cudaMemsetAsync(ptr_, 0, bytes(), stream);
    }

    T scalar() const {
        T val{};
        if (ptr_) {
            cudaMemcpyAsync(&val, ptr_, sizeof(T), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
        }
        return val;
    }

    void toHost(T* dst) const {
        if (ptr_) {
            cudaMemcpyAsync(dst, ptr_, bytes(), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
        }
    }

    void toHostAsync(T* dst) const {
        if (ptr_)
            cudaMemcpyAsync(dst, ptr_, bytes(), cudaMemcpyDeviceToHost, stream);
    }

    static CudaArray zeros(unsigned int count, cudaStream_t stream) {
        CudaArray arr(count, stream);
        arr.fillZero();
        return arr;
    }

    static CudaArray zeros(unsigned int rows, unsigned int cols, cudaStream_t stream) {
        CudaArray arr(rows, cols, stream);
        arr.fillZero();
        return arr;
    }

    CudaArray clone() const {
        CudaArray copy(rows, cols, channels, stream);
        if (ptr_)
            cudaMemcpyAsync(copy.ptr_, ptr_, bytes(), cudaMemcpyDeviceToDevice, stream);
        return copy;
    }

    static void trimMemoryPool() {
        cudaMemPool_t pool;
        cudaDeviceGetDefaultMemPool(&pool, 0);
        cudaMemPoolTrimTo(pool, 0);
    }
};
