#include "cuda_stream_manager.hpp"
#include <cuda_runtime.h>
#include <af/cuda.h>
#include <arrayfire.h>

CudaStreamManager::CudaStreamManager()
{
    cudaStreamCreate(&m_stream);
    m_afStream = afcu::getStream(afcu::getNativeId(af::getDevice()));
}

CudaStreamManager::~CudaStreamManager()
{
    if (m_stream)
        cudaStreamDestroy(m_stream);
}

CudaStreamManager& CudaStreamManager::getInstance()
{
    static CudaStreamManager instance;
    return instance;
}

cudaStream_t CudaStreamManager::getCustomStream() const
{
    return m_stream;
}

cudaStream_t CudaStreamManager::getAfStream() const
{
    return m_afStream;
}