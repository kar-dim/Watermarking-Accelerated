#include "CudaStreamManager.hpp"
#include "cuda_utils.hpp"
#include "kernels/kernels.cuh"
#include <cstdint>
#include <cuda_runtime.h>

namespace cuda_utils 
{
    //get a cudaDeviceProp handle to query for various device information
    cudaDeviceProp getDeviceProperties()
    {
        int device;
        cudaGetDevice(&device);
        cudaDeviceProp properties;
        cudaGetDeviceProperties(&properties, device);
        return properties;
    }

	//convert NV12 UV plane to YUV420p format
    void launchNV12ToYUV420pKernel(const uint8_t* uvSrc, const int uvPitch, uint8_t* uvDst, const int uvWidth, const int uvHeight, const cudaStream_t stream)
    {
        constexpr int blockSize = 256;
        const int totalPixels = uvWidth * uvHeight;
        const int gridSize = (totalPixels + blockSize - 1) / blockSize;
        nV12ToYUV420p << <gridSize, blockSize, 0, stream >> > (uvSrc, uvPitch, uvDst, uvWidth, uvHeight);
    }

    //convert pitched memory to float
    void launchPitchedToFloatKernel(const uint8_t* ySrc, float* yDst, const int width, const int height, const int pitch, const cudaStream_t stream)
    {
        constexpr dim3 blockSize(16, 16);
        const dim3 gridSize = cuda_utils::gridSizeCalculate(blockSize, height, width);
        pitchedToFloat<< <gridSize, blockSize, 0, stream >> > (ySrc, yDst, width, height, pitch);
    }
}