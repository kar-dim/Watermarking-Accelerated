#pragma once
#include <cuda_runtime.h>

/*!
 *  \brief  Simple helper utility class for handling CUDA streams.
 *  \author Dimitris Karatzas
 */
class CudaStreamManager {
  public:
    static CudaStreamManager& getInstance() {
        static CudaStreamManager instance;
        return instance;
    }

    cudaStream_t getComputeStream() const { return m_computeStream; }
    cudaStream_t getTransferStream() const { return m_transferStream; }

    CudaStreamManager(const CudaStreamManager&) = delete;
    CudaStreamManager& operator=(const CudaStreamManager&) = delete;
    CudaStreamManager(CudaStreamManager&&) = delete;
    CudaStreamManager& operator=(CudaStreamManager&&) = delete;

  private:
    cudaStream_t m_computeStream;
    cudaStream_t m_transferStream;

    CudaStreamManager() {
        cudaStreamCreate(&m_computeStream);
        cudaStreamCreate(&m_transferStream);
    }

    ~CudaStreamManager() {
        if (m_computeStream)
            cudaStreamDestroy(m_computeStream);
        if (m_transferStream)
            cudaStreamDestroy(m_transferStream);
    }
};
