#pragma once
#include <cuda_runtime.h>

/*!
 *  \brief  Simple helper utility class for handling CUDA streams.
 *  \author Dimitris Karatzas
 */
class CudaStreamManager
{
public:
    static CudaStreamManager& getInstance();

    cudaStream_t getCustomStream() const;
    cudaStream_t getAfStream() const;

    CudaStreamManager(const CudaStreamManager&) = delete;
    CudaStreamManager& operator=(const CudaStreamManager&) = delete;
    CudaStreamManager(CudaStreamManager&&) = delete;
    CudaStreamManager& operator=(CudaStreamManager&&) = delete;

private:
    CudaStreamManager();
    ~CudaStreamManager();
    cudaStream_t m_stream;
    cudaStream_t m_afStream;
};