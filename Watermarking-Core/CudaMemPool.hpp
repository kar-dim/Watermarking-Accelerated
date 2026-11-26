#pragma once
#include "CheckedSize.hpp"
#include "CudaCheck.hpp"
#include <algorithm>
#include <bit>
#include <cuda_runtime.h>
#include <mutex>
#include <string>
#include <unordered_map>

/*!
 *  \brief  Free-list CUDA device pool owned by CudaStreamManager
 *          Allocations round to powers of 2 for bucket reuse, capped by free VRAM
 *  \author Dimitris Karatzas
 */
class CudaMemPool {
  private:
    std::unordered_multimap<size_t, void*> memList;
    size_t pooledBytes = 0;
    size_t maxPoolBytes = 0;
    std::mutex mtx;

    void clearIdle(cudaStream_t stream) {
        for (auto& [sz, ptr] : memList)
            cudaFreeAsync(ptr, stream);
        memList.clear();
        pooledBytes = 0;
    }

  public:
    CudaMemPool() = default;
    ~CudaMemPool() { reset(nullptr); }

    CudaMemPool(const CudaMemPool&) = delete;
    CudaMemPool& operator=(const CudaMemPool&) = delete;

    void setCapacity(const size_t totalVramBytes) {
        // Idle buffers can crowd out allocations of other sizes
        constexpr size_t maxIdleBytes = 512ull << 20;
        maxPoolBytes = std::clamp<size_t>(totalVramBytes / 8, 1, maxIdleBytes);
    }

    void* acquire(const size_t bytes, cudaStream_t stream) {
        const size_t rounded = InternalUtils::checkedPowerOfTwo(bytes);
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
        if (err == cudaErrorMemoryAllocation && !memList.empty()) {
            cudaGetLastError();
            clearIdle(stream);
            err = cudaMallocAsync(&ptr, rounded, stream);
        }
        // the error code lets batches retire a lost device
        if (err != cudaSuccess)
            throw cuda_utils::CudaError(err, "cudaMallocAsync failed: " + std::string(cudaGetErrorString(err)));
        return ptr;
    }

    void release(const size_t bytes, void* ptr, cudaStream_t stream) noexcept {
        if (!ptr)
            return;
        const size_t rounded = std::bit_ceil(bytes);
        std::lock_guard lock(mtx);
        if (maxPoolBytes > 0 && (rounded > maxPoolBytes || pooledBytes > maxPoolBytes - rounded)) {
            cudaFreeAsync(ptr, stream);
            return;
        }
        try {
            memList.emplace(rounded, ptr);
            pooledBytes += rounded;
        } catch (...) { cudaFreeAsync(ptr, stream); }
    }

    void reset(cudaStream_t stream) {
        std::lock_guard lock(mtx);
        clearIdle(stream);
    }
};
