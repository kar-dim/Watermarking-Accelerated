#include "cuda_stream_manager.hpp"
#include <cuda_runtime.h>
#include <af/cuda.h>
#include <arrayfire.h>

CudaStreamManager::CudaStreamManager() { cudaStreamCreate(&stream_); }
CudaStreamManager::~CudaStreamManager() { cudaStreamDestroy(stream_); }
void CudaStreamManager::init() { getStream(); }
cudaStream_t CudaStreamManager::getAfStream() { return afcu::getStream(afcu::getNativeId(af::getDevice())); }

cudaStream_t& CudaStreamManager::getStream() 
{
    static CudaStreamManager instance;
    return instance.stream_;
}