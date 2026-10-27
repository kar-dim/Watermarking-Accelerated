#pragma once
#include "CudaCheck.hpp"
#include "HdrTonemap.hpp"
#include "VideoPreviewParams.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

/*!
 *  \brief  Utility functions for CUDA operations
 *  \author Dimitris Karatzas
 */
namespace cuda_utils {
void launchVideoPreview(const uint8_t* y, const uint8_t* u, const uint8_t* v, uint8_t* output, VideoPreviewParams params, cudaStream_t stream);
void launchNV12ToYUV420pKernel(const uint8_t* uvSrc, const int uvPitch, uint8_t* uvDst, const int uvWidth, const int uvHeight, const cudaStream_t stream);
void launchYUV420pToNV12Kernel(const uint8_t* uvSrc, uint8_t* uvDst, const int uvPitch, const int uvWidth, const int uvHeight, const cudaStream_t stream);
void launchPitchedToFloatKernel(const uint8_t* ySrc, float* yDst, const int width, const int height, const int pitch, const cudaStream_t stream);
void launchU8ToFloatGrayKernel(const uint8_t* input, float* output, const int planeSize, const int numChannels, const cudaStream_t stream);
void launchColMajorToRowMajorU8Kernel(const uint8_t* src, uint8_t* dst, const int width, const int height, const int channels, const cudaStream_t stream);
void launchColMajorToInterleavedU8Kernel(const uint8_t* src, uint8_t* dst, const int width, const int height, const int channels, const cudaStream_t stream);
void launchOrientRowMajorToColMajorKernel(
    const uint8_t* src, uint8_t* rgbDst, float* grayDst, uint8_t* display, const int srcWidth, const int srcHeight, const int channels, const int orientation, const cudaStream_t stream);
// packed planar 10-bit YUV420 to P010 Y and interleaved UV planes
void launchYUV420p10ToP010Kernel(const uint16_t* planar, uint16_t* yDst, uint16_t* uvDst, const int width, const int height, const cudaStream_t stream);
// HDR (P010LE BT.2020 PQ) to SDR (BT.709) conversion, outputs optional
void launchP010HdrToSdrKernel(const uint16_t* ySrc, const int yPitchBytes, const uint16_t* uvSrc, const int uvPitchBytes, uint8_t* yDst, const int yDstPitch, float* lumaDst, uint8_t* uvDst,
    const int uvDstPitch, const int width, const int height, const video_utils::ChromaSiting& siting, const video_utils::MobiusParams& mobius, const cudaStream_t stream);
// watermark generation: numElements half values from ChaCha20 start state
void launchGenerateWatermarkKernel(const std::array<uint32_t, 16>& baseState, __half* watermark, const int64_t numElements, const cudaStream_t stream);
// test-only Box-Muller transform on (x1, x2) pairs
void launchBoxMullerKernel(const uint32_t* randomPairs, float* normals, const int pairs, const cudaStream_t stream);
// 2D kernel grid size from image dimensions and blockSize
inline dim3 gridSizeCalculate(const dim3 blockSize, const int cols, const int rows) { return dim3((rows + blockSize.x - 1) / blockSize.x, (cols + blockSize.y - 1) / blockSize.y); }
// 1D grid size with 2560 block cap for grid-stride kernels
inline int gridSize1DStridedCalculate(const int N, const int blockSize) { return std::min<int>((N + blockSize - 1) / blockSize, 2560); }
// prediction error grid size based on GPU SM count
template <typename KernelFunc>
inline unsigned int gridSizeMeCalculate(KernelFunc kernel, const int blockSize) {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    int numSMs;
    CUDA_CHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, device));
    int maxBlocksPerSM;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksPerSM, kernel, blockSize, 0));
    // persistent threads grid sizing: SM count * max blocks per SM
    return static_cast<unsigned int>(numSMs * maxBlocksPerSM);
}
} // namespace cuda_utils
