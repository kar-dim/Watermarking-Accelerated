#pragma once
#include "HdrTonemap.hpp"
#include <algorithm>
#include <cstdint>
#include <cuda_runtime.h>

/*!
 *  \brief  Helper utility functions related to CUDA.
 *  \author Dimitris Karatzas
 */
namespace cuda_utils {
void launchNV12ToYUV420pKernel(const uint8_t* uvSrc, const int uvPitch, uint8_t* uvDst, const int uvWidth, const int uvHeight, const cudaStream_t stream);
void launchPitchedToFloatKernel(const uint8_t* ySrc, float* yDst, const int width, const int height, const int pitch, const cudaStream_t stream);
void launchU8ToFloatGrayKernel(const uint8_t* input, float* output, const int planeSize, const int numChannels, const cudaStream_t stream);
void launchColMajorToRowMajorU8Kernel(const uint8_t* src, uint8_t* dst, const int width, const int height, const int channels, const cudaStream_t stream);
void launchRowMajorToColMajorFloatKernel(const float* src, float* dst, const int width, const int height, const int channels, const cudaStream_t stream);
void launchRowMajorRGBToColMajorGrayKernel(const float* src, float* dst, const int width, const int height, const cudaStream_t stream);
// HDR (P010LE BT.2020 PQ) -> SDR (BT.709) conversion kernels
void launchP010HdrYToSdrFloatKernel(const uint16_t* ySrc, const int yPitchBytes, const uint16_t* uvSrc, const int uvPitchBytes, float* yDst, const int width, const int height,
                                    const video_utils::MobiusParams& mobius, const cudaStream_t stream);
void launchP010HdrUVToSdrNV12Kernel(const uint16_t* ySrc, const int yPitchBytes, const uint16_t* uvSrc, const int uvPitchBytes, uint8_t* uvDst, const int width, const int height,
                                    const video_utils::MobiusParams& mobius, const cudaStream_t stream);
void launchP010HdrYToSdrU8Kernel(const uint16_t* ySrc, const int yPitchBytes, const uint16_t* uvSrc, const int uvPitchBytes, uint8_t* yDst, const int width, const int height,
                                 const video_utils::MobiusParams& mobius, const cudaStream_t stream);
// helper method to calculate kernel grid size from given 2D dimensions and blockSize
inline dim3 gridSizeCalculate(const dim3 blockSize, const int rows, const int cols) { return dim3((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y); }
// helper method to calculate a 1D grid size for a given number of elements and block size, with a maximum of 2560 blocks (used for grid-stride kernels only)
inline int gridSize1DStridedCalculate(const int N, const int blockSize) { return std::min<int>((N + blockSize - 1) / blockSize, 2560); }
// helper method to calculate prediction error kernel 1D grid size based on the number of SMs of the GPU
template <typename KernelFunc>
inline unsigned int gridSizeMeCalculate(KernelFunc kernel, const int blockSize) {
    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    int maxBlocksPerSM;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksPerSM, kernel, blockSize, 0);
    // SM * Max blocks per SM for perfect persistent threads strategy
    return static_cast<unsigned int>(numSMs * maxBlocksPerSM);
}
} // namespace cuda_utils