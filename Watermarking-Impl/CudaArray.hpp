#pragma once
#include "CudaStreamManager.hpp"
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
    int rows = 0;
    int cols = 0;
    int channels = 1;
    cudaStream_t stream = nullptr;

    void alloc() {
        if (size() > 0)
            ptr_ = static_cast<T*>(CudaStreamManager::getInstance().getPool().acquire(bytes(), stream));
    }

    void freeArray() {
        if (ptr_) {
            CudaStreamManager::getInstance().getPool().release(bytes(), ptr_, stream);
            ptr_ = nullptr;
        }
    }

  public:
    CudaArray() = default;

    explicit CudaArray(const int count, cudaStream_t stream) : rows(count), cols(1), stream(stream) { alloc(); }

    CudaArray(const int rows, const int cols, cudaStream_t stream) : rows(rows), cols(cols), stream(stream) { alloc(); }

    CudaArray(const int rows, const int cols, int channels, cudaStream_t stream) : rows(rows), cols(cols), channels(channels), stream(stream) { alloc(); }

    CudaArray(const int rows, const int cols, const T* hostData, cudaStream_t stream) : rows(rows), cols(cols), stream(stream) {
        alloc();
        cudaMemcpyAsync(ptr_, hostData, bytes(), cudaMemcpyHostToDevice, stream);
    }

    CudaArray(const int rows, const int cols, const int channels, const T* hostData, cudaStream_t stream) : rows(rows), cols(cols), channels(channels), stream(stream) {
        alloc();
        cudaMemcpyAsync(ptr_, hostData, bytes(), cudaMemcpyHostToDevice, stream);
    }

    ~CudaArray() { freeArray(); }

    CudaArray(const CudaArray&) = delete;
    CudaArray& operator=(const CudaArray&) = delete;

    CudaArray(CudaArray&& o) noexcept : ptr_(o.ptr_), rows(o.rows), cols(o.cols), channels(o.channels), stream(o.stream) {
        o.ptr_ = nullptr;
        o.rows = o.cols = 0;
        o.channels = 1;
    }

    CudaArray& operator=(CudaArray&& o) noexcept {
        if (this != &o) {
            freeArray();
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
    int getRows() const { return rows; }
    int getCols() const { return cols; }
    int getChannels() const { return channels; }
    int size() const { return rows * cols * channels; }
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

    static CudaArray zeros(const int count, cudaStream_t stream) {
        CudaArray arr(count, stream);
        arr.fillZero();
        return arr;
    }

    static CudaArray zeros(const int rows, const int cols, cudaStream_t stream) {
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
};
