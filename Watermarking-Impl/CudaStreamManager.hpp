#pragma once
#include "CudaMemPool.hpp"
#include <cuda_runtime.h>

/*!
 *  \brief  Singleton that owns CUDA streams and the shared memory pool
 *  \author Dimitris Karatzas
 */
class CudaStreamManager {
  public:
    static CudaStreamManager& getInstance() {
        static CudaStreamManager instance;
        return instance;
    }

    cudaStream_t getComputeStream() const { return computeStream; }
    cudaStream_t getTransferStream() const { return transferStream; }
    CudaMemPool& getPool() { return pool; }

    CudaStreamManager(const CudaStreamManager&) = delete;
    CudaStreamManager& operator=(const CudaStreamManager&) = delete;
    CudaStreamManager(CudaStreamManager&&) = delete;
    CudaStreamManager& operator=(CudaStreamManager&&) = delete;

  private:
    cudaStream_t computeStream;
    cudaStream_t transferStream;
    CudaMemPool pool;

    CudaStreamManager() {
        cudaStreamCreate(&computeStream);
        cudaStreamCreate(&transferStream);
        size_t free = 0, total = 0;
        cudaMemGetInfo(&free, &total);
        pool.setCapacity(total);
    }

    ~CudaStreamManager() {
        pool.reset(computeStream);
        if (computeStream)
            cudaStreamDestroy(computeStream);
        if (transferStream)
            cudaStreamDestroy(transferStream);
    }
};
