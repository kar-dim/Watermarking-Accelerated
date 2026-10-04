#include "cuda_utils.hpp"
#include "kernels/kernels.cuh"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

/*!
 *  \brief  CUDA kernel launch wrappers for image conversions, watermark generation, and reductions
 *  \author Dimitris Karatzas
 */
namespace cuda_utils {

// convert NV12 UV plane to YUV420p format
void launchNV12ToYUV420pKernel(const uint8_t* uvSrc, const int uvPitch, uint8_t* uvDst, const int uvWidth, const int uvHeight, const cudaStream_t stream) {
    constexpr int blockSize = 256;
    const int totalPixels = uvWidth * uvHeight;
    const int gridSize = (totalPixels + blockSize - 1) / blockSize;
    nV12ToYUV420p<<<gridSize, blockSize, 0, stream>>>(uvSrc, uvPitch, uvDst, uvWidth, uvHeight);
    CUDA_CHECK(cudaGetLastError());
}

// convert pitched memory to float
void launchPitchedToFloatKernel(const uint8_t* ySrc, float* yDst, const int width, const int height, const int pitch, const cudaStream_t stream) {
    constexpr dim3 blockSize(32, 8);
    const dim3 gridSize((width + 31) / 32, (height + 31) / 32);
    pitchedToFloat<<<gridSize, blockSize, 0, stream>>>(ySrc, yDst, width, height, pitch);
    CUDA_CHECK(cudaGetLastError());
}

// uint8 col-major to float col-major grayscale
void launchU8ToFloatGrayKernel(const uint8_t* input, float* output, const int planeSize, const int numChannels, const cudaStream_t stream) {
    constexpr int blockSize = 768;
    const int gridSize = gridSize1DStridedCalculate(planeSize, blockSize);
    u8ToFloatGray<<<gridSize, blockSize, 0, stream>>>(input, output, planeSize, numChannels);
    CUDA_CHECK(cudaGetLastError());
}

// transpose column-major uint8 to row-major uint8
void launchColMajorToRowMajorU8Kernel(const uint8_t* src, uint8_t* dst, const int width, const int height, const int channels, const cudaStream_t stream) {
    constexpr dim3 blockSize(32, 8);
    const dim3 gridSize((width + 31) / 32, (height + 31) / 32, channels);
    colMajorToRowMajorU8<<<gridSize, blockSize, 0, stream>>>(src, dst, width, height);
    CUDA_CHECK(cudaGetLastError());
}

// transpose column-major planar uint8 to row-major interleaved uint8 (RGBRGB..., the display layout)
void launchColMajorToInterleavedU8Kernel(const uint8_t* src, uint8_t* dst, const int width, const int height, const int channels, const cudaStream_t stream) {
    constexpr dim3 blockSize(32, 8);
    const dim3 gridSize((width + 31) / 32, (height + 31) / 32);
    colMajorToInterleavedU8<<<gridSize, blockSize, 0, stream>>>(src, dst, width, height, channels);
    CUDA_CHECK(cudaGetLastError());
}
// row-major planar 8-bit image (CImg, 1 or 3 channels) to the displayed (EXIF oriented) column-major 8-bit RGB planes + float luma (CudaArray), and the optional display copy
void launchOrientRowMajorToColMajorKernel(
    const uint8_t* src, uint8_t* rgbDst, float* grayDst, uint8_t* display, const int srcWidth, const int srcHeight, const int channels, const int orientation, const cudaStream_t stream) {
    const bool swapAxes = orientation >= 5;
    const int width = swapAxes ? srcHeight : srcWidth;
    const int height = swapAxes ? srcWidth : srcHeight;
    constexpr dim3 blockSize(32, 8);
    const dim3 gridSize((width + 31) / 32, (height + 31) / 32);
    if (channels == 3)
        orientRowMajorToColMajor<3><<<gridSize, blockSize, 0, stream>>>(src, rgbDst, grayDst, display, srcWidth, srcHeight, orientation);
    else
        orientRowMajorToColMajor<1><<<gridSize, blockSize, 0, stream>>>(src, rgbDst, grayDst, display, srcWidth, srcHeight, orientation);
    CUDA_CHECK(cudaGetLastError());
}

void launchP010HdrYToSdrFloatKernel(const uint16_t* ySrc, const int yPitchBytes, const uint16_t* uvSrc, const int uvPitchBytes, float* yDst, const int width, const int height,
    const video_utils::MobiusParams& mobius, const cudaStream_t stream) {
    constexpr dim3 blockSize(32, 8);
    const dim3 gridSize((width + 31) / 32, (height + 31) / 32);
    p010HdrYToSdrFloat<<<gridSize, blockSize, 0, stream>>>(ySrc, yPitchBytes, uvSrc, uvPitchBytes, yDst, width, height, mobius.a, mobius.b, mobius.k);
    CUDA_CHECK(cudaGetLastError());
}

void launchP010HdrUVToSdrNV12Kernel(const uint16_t* ySrc, const int yPitchBytes, const uint16_t* uvSrc, const int uvPitchBytes, uint8_t* uvDst, const int width, const int height,
    const video_utils::MobiusParams& mobius, const cudaStream_t stream) {
    constexpr dim3 blockSize(32, 8);
    const dim3 gridSize((width / 2 + 31) / 32, (height / 2 + 7) / 8);
    p010HdrUVToSdrNV12<<<gridSize, blockSize, 0, stream>>>(ySrc, yPitchBytes, uvSrc, uvPitchBytes, uvDst, width, height, mobius.a, mobius.b, mobius.k);
    CUDA_CHECK(cudaGetLastError());
}

void launchP010HdrYToSdrU8Kernel(const uint16_t* ySrc, const int yPitchBytes, const uint16_t* uvSrc, const int uvPitchBytes, uint8_t* yDst, const int width, const int height,
    const video_utils::MobiusParams& mobius, const cudaStream_t stream) {
    constexpr dim3 blockSize(32, 8);
    const dim3 gridSize((width + 31) / 32, (height + 31) / 32);
    p010HdrYToSdrU8<<<gridSize, blockSize, 0, stream>>>(ySrc, yPitchBytes, uvSrc, uvPitchBytes, yDst, width, height, mobius.a, mobius.b, mobius.k);
    CUDA_CHECK(cudaGetLastError());
}

void launchGenerateWatermarkKernel(const std::array<uint32_t, 16>& baseState, __half* watermark, const int64_t numElements, const cudaStream_t stream) {
    ChaChaState state;
    std::copy(baseState.begin(), baseState.end(), state.words);
    constexpr int blockSize = 256;
    const int64_t chachaBlocks = (numElements + 7) / 8;
    const unsigned int gridSize = static_cast<unsigned int>((chachaBlocks + blockSize - 1) / blockSize);
    generate_watermark<<<gridSize, blockSize, 0, stream>>>(state, watermark, numElements);
    CUDA_CHECK(cudaGetLastError());
}

// used by TESTS only to verify watermark generation internal stuff, not used in production code
void launchBoxMullerKernel(const uint32_t* randomPairs, float* normals, const int pairs, const cudaStream_t stream) {
    constexpr int blockSize = 256;
    box_muller_pairs<<<(pairs + blockSize - 1) / blockSize, blockSize, 0, stream>>>(randomPairs, normals, pairs);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace cuda_utils
