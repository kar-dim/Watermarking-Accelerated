#include "cuda_stream_manager.hpp"
#include "cuda_utils.hpp"
#include "kernels/kernels.cuh"
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

    void launchNV12ToYUV420pKernel(const uint8_t* uvSrc, const int uvPitch, uint8_t* uvDst, const int uvWidth, const int uvHeight, const cudaStream_t stream)
    {
        constexpr int pairsPerThread = 4; //uint4 -> 4 UV pairs (pixels) per thread
        constexpr int blockSize = 256;
        const int totalPixels = uvWidth * uvHeight;
        const int threadsNeeded = (totalPixels + pairsPerThread - 1) / pairsPerThread;
        const int gridSize = (threadsNeeded + blockSize - 1) / blockSize;
        nV12ToYUV420p << <gridSize, blockSize, 0, stream >> > (uvSrc, uvPitch, uvDst, uvWidth, uvHeight);
    }

    void launchU8PitchedToFloatKernel(const uint8_t* ySrc, float* yDst, const int width, const int height, const int pitch, const cudaStream_t stream)
    {
        constexpr dim3 blockSize(16, 16);
        const dim3 gridSize = cuda_utils::gridSizeCalculate(blockSize, height, width);
        u8PitchedToFloat<< <gridSize, blockSize, 0, stream >> > (ySrc, yDst, width, height, pitch);
    }
}