#pragma once
#include <af/cuda.h>
#include <arrayfire.h>
#include <cuda_runtime.h>

/*!
 *  \brief  Simple helper utility class for handling CUDA streams.
 *  \author Dimitris Karatzas
 */
class CudaStreamManager
{
public:
    static CudaStreamManager& getInstance()
    {
        static CudaStreamManager instance;
        return instance;
    }

    cudaStream_t getCustomStream() const { return m_stream; }
    cudaStream_t getAfStream() const { return m_afStream; }

    CudaStreamManager(const CudaStreamManager&) = delete;
    CudaStreamManager& operator=(const CudaStreamManager&) = delete;
    CudaStreamManager(CudaStreamManager&&) = delete;
    CudaStreamManager& operator=(CudaStreamManager&&) = delete;

private:
    cudaStream_t m_stream;
    cudaStream_t m_afStream;

    CudaStreamManager()
    {
        cudaStreamCreate(&m_stream);
        m_afStream = afcu::getStream(afcu::getNativeId(af::getDevice()));
    }

    ~CudaStreamManager()
    {
        if (m_stream)
            cudaStreamDestroy(m_stream);
    }
};