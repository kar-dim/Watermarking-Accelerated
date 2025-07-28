#include "cuda_utils.hpp"
#include <cuda_runtime.h>

namespace cuda_utils 
{
    //Helper method to calculate kernel grid size from given 2D dimensions and blockSize
    dim3 gridSizeCalculate(const dim3 blockSize, const int rows, const int cols)
    {
        return dim3((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);
    }

    //get a cudaDeviceProp handle to query for various device information
    cudaDeviceProp getDeviceProperties()
    {
        int device;
        cudaGetDevice(&device);
        cudaDeviceProp properties;
        cudaGetDeviceProperties(&properties, device);
        return properties;
    }
}