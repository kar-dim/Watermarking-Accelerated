#pragma once
#include <cstdint>
#include <cuda_runtime.h>

/*!
 *  \brief  Helper utility functions related to CUDA.
 *  \author Dimitris Karatzas
 */
namespace cuda_utils {
void launchNV12ToYUV420pKernel(const uint8_t* uvSrc, const int uvPitch, uint8_t* uvDst, const int uvWidth, const int uvHeight, const cudaStream_t stream);
void launchPitchedToFloatKernel(const uint8_t* ySrc, float* yDst, const int width, const int height, const int pitch, const cudaStream_t stream);
// Helper method to calculate kernel grid size from given 2D dimensions and blockSize
inline dim3 gridSizeCalculate(const dim3 blockSize, const int rows, const int cols) {
    return dim3((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);
}
} // namespace cuda_utils