#pragma once
#include "../CudaCheck.hpp"
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
    cudaStream_t computeStream = nullptr;
    cudaStream_t transferStream = nullptr;
    CudaMemPool pool;

    CudaStreamManager() {
        try {
            CUDA_CHECK(cudaStreamCreate(&computeStream));
            CUDA_CHECK(cudaStreamCreate(&transferStream));
            size_t freeMem = 0;
            size_t totalMem = 0;
            CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));
            pool.setCapacity(totalMem);
        } catch (...) {
            if (transferStream)
                cudaStreamDestroy(transferStream);
            if (computeStream)
                cudaStreamDestroy(computeStream);
            throw;
        }
    }

    ~CudaStreamManager() {
        pool.reset(computeStream);
        if (computeStream)
            cudaStreamDestroy(computeStream);
        if (transferStream)
            cudaStreamDestroy(transferStream);
    }
};
