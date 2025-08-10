#include "cuda_utils.hpp"
#include "kernels/kernels.cuh"
#include <cuda_runtime.h>
#include <arrayfire.h>
#include <af/cuda.h>

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

    void launchNV12ToYUV420pKernel(const uint8_t* uvSrc, const int uvPitch, uint8_t* uvDst, const int uvWidth, const int uvHeight)
    {
        static cudaStream_t afStream = afcu::getStream(afcu::getNativeId(af::getDevice()));
        constexpr int pairsPerThread = 4; //uint4 -> 4 UV pairs (pixels) per thread
        constexpr int blockSize = 256;
        const int totalPixels = uvWidth * uvHeight;
        const int threadsNeeded = (totalPixels + pairsPerThread - 1) / pairsPerThread;
        const int gridSize = (threadsNeeded + blockSize - 1) / blockSize;
        NV12ToYUV420p << <gridSize, blockSize, 0, afStream >> > (uvSrc, uvPitch, uvDst, uvWidth, uvHeight);
    }
}