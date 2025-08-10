#pragma once
#include <cstdint>
#include <cuda_runtime.h>

/*!
 *  \brief  Helper utility functions related to CUDA.
 *  \author Dimitris Karatzas
 */
namespace cuda_utils 
{
    dim3 gridSizeCalculate(const dim3 blockSize, const int rows, const int cols);
    cudaDeviceProp getDeviceProperties();
    void launchNV12ToYUV420pKernel(const uint8_t* uvSrc, const int uvPitch, uint8_t* uvDst, const int uvWidth, const int uvHeight);
}