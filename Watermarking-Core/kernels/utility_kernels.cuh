#pragma once
#include "common_utils.hpp"
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/*!
 * \brief CUDA image and video kernels used by the host launch wrappers
 * \author Dimitris Karatzas
 */

__global__ void nV12ToYUV420p(const uint8_t* __restrict__ uvSrc, const int uvPitch, uint8_t* __restrict__ uvDst, const int uvWidth, const int uvHeight);

__global__ void yuv420pToNV12(const uint8_t* __restrict__ uvSrc, uint8_t* __restrict__ uvDst, const int uvPitch, const int uvWidth, const int uvHeight);

__global__ void pitchedToFloat(const uint8_t* __restrict__ input, float* __restrict__ output, const int width, const int height, const int pitch);

__global__ void u8ToFloatGray(const uint8_t* __restrict__ input, float* __restrict__ output, const int planeSize, const int numChannels);

__global__ void colMajorToRowMajorU8(const uint8_t* __restrict__ src, uint8_t* __restrict__ dst, const int width, const int height);

__global__ void colMajorToInterleavedU8(const uint8_t* __restrict__ src, uint8_t* __restrict__ dst, const int width, const int height, const int channels);

// EXIF mapping must match the CPU rotation in utils.cpp
__device__ __forceinline__ int2 orientedSource(const int x, const int y, const int srcWidth, const int srcHeight, const int orientation) {
    switch (orientation) {
    case 2: return make_int2(srcWidth - 1 - x, y);
    case 3: return make_int2(srcWidth - 1 - x, srcHeight - 1 - y);
    case 4: return make_int2(x, srcHeight - 1 - y);
    case 5: return make_int2(y, x);
    case 6: return make_int2(y, srcHeight - 1 - x);
    case 7: return make_int2(srcWidth - 1 - y, srcHeight - 1 - x);
    case 8: return make_int2(srcWidth - 1 - y, x);
    default: return make_int2(x, y);
    }
}

// EXIF transforms keep each tile square, only its axes and direction change
struct TileOrientation {
    bool swapAxes;
    int mirrorX;
    int mirrorY;
    __device__ explicit TileOrientation(const int orientation)
        : swapAxes(orientation >= 5), mirrorX(orientation == 2 || orientation == 3 || orientation == 7 || orientation == 8 ? 31 : 0),
          mirrorY(orientation == 3 || orientation == 4 || orientation == 6 || orientation == 7 ? 31 : 0) {}
    __device__ int2 operator()(const int x, const int y) const { return make_int2((swapAxes ? y : x) ^ mirrorX, (swapAxes ? x : y) ^ mirrorY); }
};

// The preview uses interleaved rows, watermark buffers use planar columns
template <int CHANNELS>
__global__ void orientRowMajorToColMajor(
    const uint8_t* __restrict__ src, uint8_t* __restrict__ rgbDst, float* __restrict__ grayDst, uint8_t* __restrict__ display, const int srcWidth, const int srcHeight, const int orientation) {
    __shared__ uint8_t tile[CHANNELS][32][33];
    const bool swapAxes = orientation >= 5;
    const int width = swapAxes ? srcHeight : srcWidth;
    const int height = swapAxes ? srcWidth : srcHeight;
    const int tileCol = blockIdx.x * 32;
    const int tileRow = blockIdx.y * 32;
    // the source pixels of a 32x32 displayed tile are a 32x32 square too, loaded with coalesced reads
    const int2 first = orientedSource(tileCol, tileRow, srcWidth, srcHeight, orientation);
    const int2 last = orientedSource(tileCol + 31, tileRow + 31, srcWidth, srcHeight, orientation);
    const int srcCol0 = min(first.x, last.x);
    const int srcRow0 = min(first.y, last.y);
    const int srcCol = srcCol0 + threadIdx.x;
    const int srcPlaneSize = srcWidth * srcHeight;
#pragma unroll
    for (int i = 0; i < 32; i += 8) {
        const int srcRow = srcRow0 + threadIdx.y + i;
        if (srcCol >= 0 && srcCol < srcWidth && srcRow >= 0 && srcRow < srcHeight) {
#pragma unroll
            for (int c = 0; c < CHANNELS; c++)
                tile[c][threadIdx.y + i][threadIdx.x] = src[c * srcPlaneSize + srcRow * srcWidth + srcCol];
        }
    }
    __syncthreads();
    // column-major planes and luma: coalesced writes along the displayed columns
    const TileOrientation tileOrientation(orientation);
    const int planeSize = width * height;
    const int row = tileRow + threadIdx.x;
#pragma unroll
    for (int i = 0; i < 32; i += 8) {
        const int col = tileCol + threadIdx.y + i;
        if (row < height && col < width) {
            const int2 source = tileOrientation(threadIdx.y + i, threadIdx.x);
            const int idx = col * height + row;
            if constexpr (CHANNELS == 3) {
                const uint8_t r = tile[0][source.y][source.x];
                const uint8_t g = tile[1][source.y][source.x];
                const uint8_t b = tile[2][source.y][source.x];
                rgbDst[idx] = r;
                rgbDst[planeSize + idx] = g;
                rgbDst[2 * planeSize + idx] = b;
                grayDst[idx] = static_cast<float>(r) * CommonUtils::kLumaR + static_cast<float>(g) * CommonUtils::kLumaG + static_cast<float>(b) * CommonUtils::kLumaB;
            } else {
                grayDst[idx] = static_cast<float>(tile[0][source.y][source.x]);
            }
        }
    }
    // the displayed pixels, row-major interleaved: coalesced writes along the displayed rows
    if (display) {
        const int rowBytes = min(32, width - tileCol) * CHANNELS;
        for (int i = threadIdx.y; i < 32 && (tileRow + i) < height; i += 8) {
            uint8_t* output = display + ((tileRow + i) * width + tileCol) * CHANNELS;
            for (int byte = threadIdx.x; byte < rowBytes; byte += 32) {
                const int2 source = tileOrientation(byte / CHANNELS, i);
                output[byte] = tile[byte % CHANNELS][source.y][source.x];
            }
        }
    }
}

// The ChaCha20 start state of the watermark password (kernel argument)
struct ChaChaState {
    uint32_t words[16];
};

// Generates the watermark: one ChaCha20 block per thread, 8 standard normal values (Box-Muller) rounded to half
__global__ void generate_watermark(const ChaChaState baseState, __half* __restrict__ watermark, const int64_t numElements);

// Box-Muller transform of (x1, x2) pairs of 24-bit random values, the transform of generate_watermark (tested against the CPU)
__global__ void box_muller_pairs(const uint32_t* __restrict__ randomPairs, float* __restrict__ normals, const int pairs);

// planar 10-bit YUV420 (samples in the low bits, packed Y, U then V) to P010 (high bits, interleaved chroma): one thread per chroma pair and its 2x2 luma
__global__ void yuv420p10ToP010(const uint16_t* __restrict__ planar, uint16_t* __restrict__ yDst, uint16_t* __restrict__ uvDst, const int width, const int height);

// HDR uses an apron for the chroma filter, output planes can be omitted with null pointers
__global__ void p010HdrToSdr(const uint16_t* __restrict__ ySrc, const int yPitchBytes, const uint16_t* __restrict__ uvSrc, const int uvPitchBytes, uint8_t* __restrict__ yDst, const int yDstPitch,
    float* __restrict__ lumaDst, uint8_t* __restrict__ uvDst, const int uvDstPitch, const int width, const int height, const float siteX, const float siteY, const float mobA, const float mobB,
    const float mobK);
