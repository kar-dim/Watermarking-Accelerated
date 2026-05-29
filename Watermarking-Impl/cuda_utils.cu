#include "cuda_utils.hpp"
#include "kernels/kernels.cuh"
#include <algorithm>
#include <cstdint>
#include <cuda_runtime.h>

namespace cuda_utils {
// Mobius tonemap constants (a, b, K) depend only on hdrPeak, so compute them once on the host
// and give them to the HDR kernels instead of recomputing per pixel (matches FFmpeg vf_tonemap.c)
namespace {
struct MobiusParams {
    float a, b, k;
};
MobiusParams computeMobiusParams(const float hdrPeak) {
    constexpr float j = 0.3f;
    const float a = -j * j * (hdrPeak - 1.0f) / (j * j - 2.0f * j + hdrPeak);
    const float b = (j * j - 2.0f * j * hdrPeak + hdrPeak) / std::max(hdrPeak - 1.0f, 1e-6f);
    const float k = (b * b + 2.0f * b * j + j * j) / (b - a);
    return {a, b, k};
}
} // namespace

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

// uint8 col-major to float col-major grayscale
void launchU8ToFloatGrayKernel(const uint8_t* input, float* output, const int planeSize, const int numChannels, const cudaStream_t stream) {
    constexpr int blockSize = 768;
    const int gridSize = gridSize1DStridedCalculate(planeSize, blockSize);
    u8ToFloatGray<<<gridSize, blockSize, 0, stream>>>(input, output, planeSize, numChannels);
}

// transpose column-major uint8 to row-major uint8
void launchColMajorToRowMajorU8Kernel(const uint8_t* src, uint8_t* dst, const int width, const int height, const int channels, const cudaStream_t stream) {
    constexpr dim3 blockSize(32, 8);
    const dim3 gridSize((width + 31) / 32, (height + 31) / 32, channels);
    colMajorToRowMajorU8<<<gridSize, blockSize, 0, stream>>>(src, dst, width, height);
}
// transpose row-major float (CImg) to column-major float (CudaArray)
void launchRowMajorToColMajorFloatKernel(const float* src, float* dst, const int width, const int height, const int channels, const cudaStream_t stream) {
    constexpr dim3 blockSize(32, 8);
    const dim3 gridSize((width + 31) / 32, (height + 31) / 32, channels);
    rowMajorToColMajorFloat<<<gridSize, blockSize, 0, stream>>>(src, dst, width, height);
}
// fused row-major 3-channel RGB to col-major grayscale with luma weights
void launchRowMajorRGBToColMajorGrayKernel(const float* src, float* dst, const int width, const int height, const cudaStream_t stream) {
    constexpr dim3 blockSize(32, 8);
    const dim3 gridSize((width + 31) / 32, (height + 31) / 32);
    rowMajorRGBToColMajorGray<<<gridSize, blockSize, 0, stream>>>(src, dst, width, height);
}

void launchP010HdrYToSdrFloatKernel(const uint16_t* ySrc, const int yPitchBytes, const uint16_t* uvSrc, const int uvPitchBytes, float* yDst, const int width, const int height, const float hdrPeak,
                                    const cudaStream_t stream) {
    constexpr dim3 blockSize(32, 8);
    const dim3 gridSize((width + 31) / 32, (height + 31) / 32);
    const MobiusParams mob = computeMobiusParams(hdrPeak);
    p010HdrYToSdrFloat<<<gridSize, blockSize, 0, stream>>>(ySrc, yPitchBytes, uvSrc, uvPitchBytes, yDst, width, height, mob.a, mob.b, mob.k);
}

void launchP010HdrUVToSdrNV12Kernel(const uint16_t* ySrc, const int yPitchBytes, const uint16_t* uvSrc, const int uvPitchBytes, uint8_t* uvDst, const int width, const int height, const float hdrPeak,
                                    const cudaStream_t stream) {
    constexpr dim3 blockSize(32, 8);
    const dim3 gridSize((width / 2 + 31) / 32, (height / 2 + 7) / 8);
    const MobiusParams mob = computeMobiusParams(hdrPeak);
    p010HdrUVToSdrNV12<<<gridSize, blockSize, 0, stream>>>(ySrc, yPitchBytes, uvSrc, uvPitchBytes, uvDst, width, height, mob.a, mob.b, mob.k);
}

void launchP010HdrYToSdrU8Kernel(const uint16_t* ySrc, const int yPitchBytes, const uint16_t* uvSrc, const int uvPitchBytes, uint8_t* yDst, const int width, const int height, const float hdrPeak,
                                 const cudaStream_t stream) {
    constexpr dim3 blockSize(32, 8);
    const dim3 gridSize((width + 31) / 32, (height + 31) / 32);
    const MobiusParams mob = computeMobiusParams(hdrPeak);
    p010HdrYToSdrU8<<<gridSize, blockSize, 0, stream>>>(ySrc, yPitchBytes, uvSrc, uvPitchBytes, yDst, width, height, mob.a, mob.b, mob.k);
}
} // namespace cuda_utils