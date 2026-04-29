#include "cuda_utils.hpp"
#include "kernels/kernels.cuh"
#include <cstdint>
#include <cuda_runtime.h>

namespace cuda_utils {
// convert NV12 UV plane to YUV420p format
void launchNV12ToYUV420pKernel(const uint8_t* uvSrc, const int uvPitch, uint8_t* uvDst, const int uvWidth, const int uvHeight, const cudaStream_t stream) {
    constexpr int blockSize = 256;
    const int totalPixels = uvWidth * uvHeight;
    const int gridSize = (totalPixels + blockSize - 1) / blockSize;
    nV12ToYUV420p<<<gridSize, blockSize, 0, stream>>>(uvSrc, uvPitch, uvDst, uvWidth, uvHeight);
}

// convert pitched memory to float
void launchPitchedToFloatKernel(const uint8_t* ySrc, float* yDst, const int width, const int height, const int pitch, const cudaStream_t stream) {
    constexpr dim3 blockSize(32, 8);
    const dim3 gridSize((width + 31) / 32, (height + 31) / 32);
    pitchedToFloat<<<gridSize, blockSize, 0, stream>>>(ySrc, yDst, width, height, pitch);
}

// transpose column-major uint8 to row-major uint8
void launchColMajorToRowMajorU8Kernel(const uint8_t* src, uint8_t* dst, const int width, const int height, const cudaStream_t stream) {
    constexpr dim3 blockSize(32, 8);
    const dim3 gridSize((width + 31) / 32, (height + 31) / 32);
    colMajorToRowMajorU8<<<gridSize, blockSize, 0, stream>>>(src, dst, width, height);
}
// transpose row-major float (CImg) to column-major float (GpuArray)
void launchRowMajorToColMajorFloatKernel(const float* src, float* dst, const int width, const int height, const int channels, const cudaStream_t stream) {
    constexpr dim3 blockSize(32, 8);
    const dim3 gridSize((width + 31) / 32, (height + 31) / 32, channels);
    rowMajorToColMajorFloat<<<gridSize, blockSize, 0, stream>>>(src, dst, width, height);
}
} // namespace cuda_utils