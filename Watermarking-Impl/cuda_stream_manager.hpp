#pragma once
#include <cuda_runtime.h>

/*!
 *  \brief  Simple helper utility class for handling CUDA streams.
 *  \author Dimitris Karatzas
 */
class CudaStreamManager
{
public:
    static void init();
    static cudaStream_t& getStream();
    static cudaStream_t getAfStream();
private:
    cudaStream_t stream_;
    CudaStreamManager();
    ~CudaStreamManager();
    CudaStreamManager(const CudaStreamManager&) = delete;
    CudaStreamManager& operator=(const CudaStreamManager&) = delete;
    CudaStreamManager(CudaStreamManager&&) = delete;
    CudaStreamManager& operator=(CudaStreamManager&&) = delete;
};