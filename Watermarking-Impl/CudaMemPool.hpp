#pragma once
#include <cuda_runtime.h>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

/*!
 *  \brief  Free-list memory pool for CUDA device pointers, allocations are rounded up to the next
 *          power of 2 so that images with similar dimensions are put in the same "bucket",
 *          a maximum capacity cap, set based on available VRAM prevents overflow to RAM.
 *          Owned by CudaStreamManager.
 *  \author Dimitris Karatzas
 */
class CudaMemPool {
  private:
    std::unordered_multimap<size_t, void*> memList;
    size_t pooledBytes = 0;
    size_t maxPoolBytes = 0;
    std::mutex mtx;

    static size_t roundUpPow2(size_t v) {
        if (v <= 1)
            return 1;
        v--;
        v |= v >> 1;
        v |= v >> 2;
        v |= v >> 4;
        v |= v >> 8;
        v |= v >> 16;
        v |= v >> 32;
        return v + 1;
    }

  public:
    CudaMemPool() = default;
    ~CudaMemPool() {
        for (auto& [sz, ptr] : memList)
            cudaFree(ptr);
    }

    CudaMemPool(const CudaMemPool&) = delete;
    CudaMemPool& operator=(const CudaMemPool&) = delete;

    void setCapacity(const size_t vramBytes) { maxPoolBytes = static_cast<size_t>(vramBytes * 0.95); }

    void* acquire(const size_t bytes, cudaStream_t stream) {
        const size_t rounded = roundUpPow2(bytes);
        std::lock_guard lock(mtx);
        auto it = memList.find(rounded);
        if (it != memList.end()) {
            void* ptr = it->second;
            pooledBytes -= rounded;
            memList.erase(it);
            return ptr;
        }
        void* ptr = nullptr;
        cudaError_t err = cudaMallocAsync(&ptr, rounded, stream);
        if (err != cudaSuccess)
            throw std::runtime_error("cudaMallocAsync failed: " + std::string(cudaGetErrorString(err)));
        return ptr;
    }

    void release(const size_t bytes, void* ptr, cudaStream_t stream) {
        if (!ptr)
            return;
        const size_t rounded = roundUpPow2(bytes);
        std::lock_guard lock(mtx);
        if (maxPoolBytes > 0 && pooledBytes + rounded > maxPoolBytes) {
            cudaFreeAsync(ptr, stream);
            return;
        }
        pooledBytes += rounded;
        memList.emplace(rounded, ptr);
    }

    void reset(cudaStream_t stream) {
        std::lock_guard lock(mtx);
        for (auto& [sz, ptr] : memList)
            cudaFreeAsync(ptr, stream);
        memList.clear();
        pooledBytes = 0;
    }
};
