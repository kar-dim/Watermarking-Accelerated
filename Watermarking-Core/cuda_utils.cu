#include "cuda_utils.hpp"
#include "kernels/utility_kernels.cuh"
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

namespace {
__global__ void videoPreviewKernel(const uint8_t* yPlane, const uint8_t* uPlane, const uint8_t* vPlane, uchar4* output, const VideoPreviewParams p) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= p.outWidth || y >= p.outHeight)
        return;
    const int sx = min(p.width - 1, static_cast<int>((x + .5f) * p.width / p.outWidth));
    const int sy = min(p.height - 1, static_cast<int>((y + .5f) * p.height / p.outHeight));
    const int cx = sx >> p.shiftX, cy = sy >> p.shiftY;
    const float yy = (yPlane[sy * p.yPitch + sx] - p.yOffset) * p.yScale;
    const float u = p.gray ? 0.f : (uPlane[cy * p.uPitch + cx * (1 + p.interleaved)] - 128.f) * p.uvScale;
    const float v = p.gray ? 0.f : (vPlane[cy * p.vPitch + cx * (1 + p.interleaved)] - 128.f) * p.uvScale;
    output[y * p.outWidth + x] = make_uchar4(static_cast<uint8_t>(__float2int_rn(fminf(255.f, fmaxf(0.f, yy + p.redV * v)))),
        static_cast<uint8_t>(__float2int_rn(fminf(255.f, fmaxf(0.f, yy + p.greenU * u + p.greenV * v)))), static_cast<uint8_t>(__float2int_rn(fminf(255.f, fmaxf(0.f, yy + p.blueU * u)))), 255);
}
} // namespace

void launchVideoPreview(const uint8_t* y, const uint8_t* u, const uint8_t* v, uint8_t* output, const VideoPreviewParams params, const cudaStream_t stream) {
    const dim3 block(16, 16);
    videoPreviewKernel<<<dim3((params.outWidth + 15) / 16, (params.outHeight + 15) / 16), block, 0, stream>>>(y, u, v, reinterpret_cast<uchar4*>(output), params);
    CUDA_CHECK(cudaGetLastError());
}

// convert NV12 UV plane to YUV420P format
void launchNV12ToYUV420pKernel(const uint8_t* uvSrc, const int uvPitch, uint8_t* uvDst, const int uvWidth, const int uvHeight, const cudaStream_t stream) {
    constexpr int blockSize = 256;
    const int totalPixels = uvWidth * uvHeight;
    const int gridSize = (totalPixels + blockSize - 1) / blockSize;
    nV12ToYUV420p<<<gridSize, blockSize, 0, stream>>>(uvSrc, uvPitch, uvDst, uvWidth, uvHeight);
    CUDA_CHECK(cudaGetLastError());
}

// interleave packed U and V planes into pitched NV12 UV plane
void launchYUV420pToNV12Kernel(const uint8_t* uvSrc, uint8_t* uvDst, const int uvPitch, const int uvWidth, const int uvHeight, const cudaStream_t stream) {
    constexpr dim3 blockSize(32, 8);
    const dim3 gridSize((uvWidth + blockSize.x - 1) / blockSize.x, (uvHeight + blockSize.y - 1) / blockSize.y);
    yuv420pToNV12<<<gridSize, blockSize, 0, stream>>>(uvSrc, uvDst, uvPitch, uvWidth, uvHeight);
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

// transpose column-major planar uint8 to row-major interleaved uint8
void launchColMajorToInterleavedU8Kernel(const uint8_t* src, uint8_t* dst, const int width, const int height, const int channels, const cudaStream_t stream) {
    constexpr dim3 blockSize(32, 8);
    const dim3 gridSize((width + 31) / 32, (height + 31) / 32);
    colMajorToInterleavedU8<<<gridSize, blockSize, 0, stream>>>(src, dst, width, height, channels);
    CUDA_CHECK(cudaGetLastError());
}
// row-major planar 8-bit image to displayed column-major RGB and float luma
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

void launchYUV420p10ToP010Kernel(const uint16_t* planar, uint16_t* yDst, uint16_t* uvDst, const int width, const int height, const cudaStream_t stream) {
    constexpr dim3 blockSize(32, 8);
    const dim3 gridSize((width / 2 + 31) / 32, (height / 2 + 7) / 8);
    yuv420p10ToP010<<<gridSize, blockSize, 0, stream>>>(planar, yDst, uvDst, width, height);
    CUDA_CHECK(cudaGetLastError());
}

void launchP010HdrToSdrKernel(const uint16_t* ySrc, const int yPitchBytes, const uint16_t* uvSrc, const int uvPitchBytes, uint8_t* yDst, const int yDstPitch, float* lumaDst, uint8_t* uvDst,
    const int uvDstPitch, const int width, const int height, const video_utils::ChromaSiting& siting, const video_utils::MobiusParams& mobius, const cudaStream_t stream) {
    // one 32x8 block per 32x8 chroma pairs
    constexpr dim3 blockSize(32, 8);
    const dim3 gridSize((width / 2 + 31) / 32, (height / 2 + 7) / 8);
    p010HdrToSdr<<<gridSize, blockSize, 0, stream>>>(
        ySrc, yPitchBytes, uvSrc, uvPitchBytes, yDst, yDstPitch, lumaDst, uvDst, uvDstPitch, width, height, siting.x, siting.y, mobius.a, mobius.b, mobius.k);
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

// test-only verification for watermark generation
void launchBoxMullerKernel(const uint32_t* randomPairs, float* normals, const int pairs, const cudaStream_t stream) {
    constexpr int blockSize = 256;
    box_muller_pairs<<<(pairs + blockSize - 1) / blockSize, blockSize, 0, stream>>>(randomPairs, normals, pairs);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace cuda_utils
