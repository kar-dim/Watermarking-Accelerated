#pragma once
#include <cuda_runtime.h>

/*!
 *  \brief  Helper utility functions related to CUDA.
 *  \author Dimitris Karatzas
 */
namespace cuda_utils 
{
    dim3 gridSizeCalculate(const dim3 blockSize, const int rows, const int cols);
    cudaDeviceProp getDeviceProperties();
}