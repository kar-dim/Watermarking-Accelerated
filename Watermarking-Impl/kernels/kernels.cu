#include "kernels.cuh"
#include <cstdint>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void apply_watermark_row_major(const float* __restrict__ input, const __half* __restrict__ u, const uint64_t* __restrict__ sumSqPtr, const uint64_t* __restrict__ maxAbsBits,
    uint8_t* __restrict__ output, const float strengthNumerator, const int width, const int height) {
    __shared__ uint8_t tile[32][36];
    const float strength = embedStrength(sumSqPtr, maxAbsBits, strengthNumerator);
    const int row = blockIdx.x * 32 + threadIdx.x;
    const int col = blockIdx.y * 32 + threadIdx.y;
#pragma unroll
    for (int i = 0; i < 32; i += 8) {
        if (row < height && (col + i) < width) {
            const int idx = (col + i) * height + row;
            tile[threadIdx.y + i][threadIdx.x] = toPixel(input[idx] + __half2float(u[idx]) * strength);
        }
    }
    __syncthreads();
    const int outCol = blockIdx.y * 32 + threadIdx.x;
    const int outRow = blockIdx.x * 32 + threadIdx.y;
#pragma unroll
    for (int i = 0; i < 32; i += 8) {
        if ((outRow + i) < height && outCol < width)
            output[(outRow + i) * width + outCol] = tile[threadIdx.x][threadIdx.y + i];
    }
}

__global__ void nV12ToYUV420p(const uint8_t* __restrict__ uvSrc, const int uvPitch, uint8_t* __restrict__ uvDst, const int uvWidth, const int uvHeight) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= uvWidth * uvHeight)
        return;
    const int y = idx / uvWidth;
    const int x = idx % uvWidth;
    const uint8_t* src = uvSrc + y * uvPitch + 2 * x;
    uvDst[idx] = src[0];
    uvDst[uvWidth * uvHeight + idx] = src[1];
}

__global__ void u8ToFloatGray(const uint8_t* __restrict__ input, float* __restrict__ output, const int planeSize, const int numChannels) {
    const int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < planeSize; i += stride) {
        if (numChannels == 3)
            output[i] = static_cast<float>(input[i]) * CommonUtils::kLumaR + static_cast<float>(input[i + planeSize]) * CommonUtils::kLumaG +
                        static_cast<float>(input[i + 2 * planeSize]) * CommonUtils::kLumaB;
        else
            output[i] = static_cast<float>(input[i]);
    }
}

__global__ void colMajorToRowMajorU8(const uint8_t* __restrict__ src, uint8_t* __restrict__ dst, const int width, const int height) {
    __shared__ uint8_t tile[32][33];
    const int planeOffset = blockIdx.z * width * height;
    const int col = blockIdx.x * 32 + threadIdx.y;
    const int row = blockIdx.y * 32 + threadIdx.x;
#pragma unroll
    for (int i = 0; i < 32; i += 8) {
        if ((col + i) < width && row < height)
            tile[threadIdx.x][threadIdx.y + i] = src[planeOffset + (col + i) * height + row];
    }
    __syncthreads();
    const int outRow = blockIdx.y * 32 + threadIdx.y;
    const int outCol = blockIdx.x * 32 + threadIdx.x;
#pragma unroll
    for (int i = 0; i < 32; i += 8) {
        if ((outRow + i) < height && outCol < width)
            dst[planeOffset + (outRow + i) * width + outCol] = tile[threadIdx.y + i][threadIdx.x];
    }
}

__global__ void pitchedToFloat(const uint8_t* __restrict__ input, float* __restrict__ output, const int width, const int height, const int pitch) {

    __shared__ float tile[32][33]; // 32x32 tile, +1 to avoid bank conflicts

    // x and y are the coordinates in the original input image
    const int x = blockIdx.x * 32 + threadIdx.x;
    const int y = blockIdx.y * 32 + threadIdx.y;
    // loop 4 times to process 32 rows
#pragma unroll
    for (int i = 0; i < 32; i += 8) {
        float val = 0.0f;
        if (x < width && (y + i) < height)
            val = static_cast<float>(input[(y + i) * pitch + x]);
        tile[threadIdx.y + i][threadIdx.x] = val;
    }
    __syncthreads();

    // swap the grid coords to calculate the transposed destination
    const int dstX = blockIdx.y * 32 + threadIdx.x;
    const int dstY = blockIdx.x * 32 + threadIdx.y;
#pragma unroll
    for (int i = 0; i < 32; i += 8) {
        // read from transposed shared memory coordinates (+1 in the stride to avoid bank conflicts) and write to global memory transposed
        if (dstX < height && (dstY + i) < width)
            output[(dstY + i) * height + dstX] = tile[threadIdx.x][threadIdx.y + i];
    }
}

// HDR Y -> col-major float: 32x8 / 32x32 tile-transpose like pitchedToFloat, with hue preserving tonemap
// Each thread reads its Y sample AND the colocated UV at (x/2, y/2), runs the full RGB pipeline,
// extracts Y_709 in limited range float [16,235], then transposes via shared memory
__global__ void p010HdrYToSdrFloat(const uint16_t* __restrict__ ySrc, const int yPitchBytes, const uint16_t* __restrict__ uvSrc, const int uvPitchBytes, float* __restrict__ output, const int width,
    const int height, const float mobA, const float mobB, const float mobK) {
    __shared__ float tile[32][33];
    const int yPitchU16 = yPitchBytes / 2;
    const int uvPitchU16 = uvPitchBytes / 2;
    const int x = blockIdx.x * 32 + threadIdx.x;
    const int y = blockIdx.y * 32 + threadIdx.y;
#pragma unroll
    for (int i = 0; i < 32; i += 8) {
        float val = 16.0f;
        if (x < width && (y + i) < height) {
            const uint16_t yRaw = ySrc[(y + i) * yPitchU16 + x];
            const int uvRow = (y + i) >> 1;
            const int uvCol = x >> 1;
            const uint16_t cbRaw = uvSrc[uvRow * uvPitchU16 + uvCol * 2];
            const uint16_t crRaw = uvSrc[uvRow * uvPitchU16 + uvCol * 2 + 1];
            val = rgbToYLimited(hdrPixelToSdrRgb(yRaw, cbRaw, crRaw, mobA, mobB, mobK));
        }
        tile[threadIdx.y + i][threadIdx.x] = val;
    }
    __syncthreads();
    const int dstX = blockIdx.y * 32 + threadIdx.x;
    const int dstY = blockIdx.x * 32 + threadIdx.y;
#pragma unroll
    for (int i = 0; i < 32; i += 8) {
        if (dstX < height && (dstY + i) < width)
            output[(dstY + i) * height + dstX] = tile[threadIdx.x][threadIdx.y + i];
    }
}

// HDR UV: one thread per UV sample, full BT.2020 YCbCr -> linear RGB -> tonemap -> BT.709 YCbCr -> NV12 uint8_t
// memory access: pack ALL loads/stores into wider types so consecutive threads for full 32-byte sector utilization
// Y load: uint32_t[uvX] on row 2*uvY, reads Y[2*uvX] (lo16) in one coalesced transaction
// UV load: uint32_t[uvX] on row uvY, reads U[uvX](lo16) + V[uvX](hi16) in one transaction
// UV store: uint16_t[uvX] on row uvY, writes Cb(lo8) + Cr(hi8) in one transaction
__global__ void p010HdrUVToSdrNV12(const uint16_t* __restrict__ ySrc, const int yPitchBytes, const uint16_t* __restrict__ uvSrc, const int uvPitchBytes, uint8_t* __restrict__ uvDst, const int width,
    const int height, const float mobA, const float mobB, const float mobK) {
    const int uvX = blockIdx.x * blockDim.x + threadIdx.x;
    const int uvY = blockIdx.y * blockDim.y + threadIdx.y;
    if (uvX >= width / 2 || uvY >= height / 2)
        return;

    // packed Y read (top-left of 2x2 block, lo16 of uint32_t)
    const uint32_t yPair = reinterpret_cast<const uint32_t*>(reinterpret_cast<const uint8_t*>(ySrc) + 2 * uvY * yPitchBytes)[uvX];
    // packed UV read (U in lo16, V in hi16)
    const uint32_t uvRaw = reinterpret_cast<const uint32_t*>(reinterpret_cast<const uint8_t*>(uvSrc) + uvY * uvPitchBytes)[uvX];
    // hue preserving HDR -> SDR pipeline -> display referred R'G'B' in [0,1]
    const float3 rgb = hdrPixelToSdrRgb(static_cast<uint16_t>(yPair & 0xFFFF), static_cast<uint16_t>(uvRaw & 0xFFFF), static_cast<uint16_t>((uvRaw >> 16) & 0xFFFF), mobA, mobB, mobK);
    // RGB -> Cb/Cr (BT.709, Kr=0.2126, Kg=0.7152, Kb=0.0722) -> limited range [16, 240]
    // Y7 as nested FMAs, divisions replaced with reciprocal multiplications
    constexpr float CB_SCALE = 1.0f / 1.8556f; // 1 / (2*(1-Kb))
    constexpr float CR_SCALE = 1.0f / 1.5748f; // 1 / (2*(1-Kr))
    const float Y7 = fmaf(0.2126f, rgb.x, fmaf(0.7152f, rgb.y, 0.0722f * rgb.z));
    const float Cb = (rgb.z - Y7) * CB_SCALE;
    const float Cr = (rgb.x - Y7) * CR_SCALE;
    // packed uint16_t store: Cb in low 8, Cr in high 8 with stride 1 in the warp
    const uint8_t cb8 = static_cast<uint8_t>(clamp(fmaf(Cb, 224.0f, 128.0f), 16.0f, 240.0f));
    const uint8_t cr8 = static_cast<uint8_t>(clamp(fmaf(Cr, 224.0f, 128.0f), 16.0f, 240.0f));
    reinterpret_cast<uint16_t*>(uvDst)[uvY * (width / 2) + uvX] = static_cast<uint16_t>(cb8) | (static_cast<uint16_t>(cr8) << 8);
}

// HDR Y passthrough: P010LE Y + UV -> uint8_t row-major limited range [16,235] (no transpose/watermark)
__global__ void p010HdrYToSdrU8(const uint16_t* __restrict__ ySrc, const int yPitchBytes, const uint16_t* __restrict__ uvSrc, const int uvPitchBytes, uint8_t* __restrict__ output, const int width,
    const int height, const float mobA, const float mobB, const float mobK) {
    const int yPitchU16 = yPitchBytes / 2;
    const int uvPitchU16 = uvPitchBytes / 2;
    const int x = blockIdx.x * 32 + threadIdx.x;
    const int y = blockIdx.y * 32 + threadIdx.y;
#pragma unroll
    for (int i = 0; i < 32; i += 8) {
        if (x < width && (y + i) < height) {
            const uint16_t yRaw = ySrc[(y + i) * yPitchU16 + x];
            const int uvRow = (y + i) >> 1;
            const int uvCol = x >> 1;
            const uint16_t cbRaw = uvSrc[uvRow * uvPitchU16 + uvCol * 2];
            const uint16_t crRaw = uvSrc[uvRow * uvPitchU16 + uvCol * 2 + 1];
            output[(y + i) * width + x] = static_cast<uint8_t>(rgbToYLimited(hdrPixelToSdrRgb(yRaw, cbRaw, crRaw, mobA, mobB, mobK)));
        }
    }
}

__global__ void rowMajorRgbToColMajor(const uint8_t* __restrict__ src, uint8_t* __restrict__ rgbDst, float* __restrict__ grayDst, const int width, const int height) {
    __shared__ uint8_t tile[3][32][36];
    const int planeSize = width * height;
    const int col = blockIdx.x * 32 + threadIdx.x;
    const int row = blockIdx.y * 32 + threadIdx.y;
#pragma unroll
    for (int i = 0; i < 32; i += 8) {
        if (col < width && (row + i) < height) {
            const int rmIdx = (row + i) * width + col;
#pragma unroll
            for (int c = 0; c < 3; c++)
                tile[c][threadIdx.y + i][threadIdx.x] = src[c * planeSize + rmIdx];
        }
    }
    __syncthreads();
    const int outRow = blockIdx.y * 32 + threadIdx.x;
    const int outCol = blockIdx.x * 32 + threadIdx.y;
#pragma unroll
    for (int i = 0; i < 32; i += 8) {
        if (outRow < height && (outCol + i) < width) {
            const int cmIdx = outRow + (outCol + i) * height;
            const uint8_t r = tile[0][threadIdx.x][threadIdx.y + i];
            const uint8_t g = tile[1][threadIdx.x][threadIdx.y + i];
            const uint8_t b = tile[2][threadIdx.x][threadIdx.y + i];
            rgbDst[cmIdx] = r;
            rgbDst[planeSize + cmIdx] = g;
            rgbDst[2 * planeSize + cmIdx] = b;
            grayDst[cmIdx] = static_cast<float>(r) * CommonUtils::kLumaR + static_cast<float>(g) * CommonUtils::kLumaG + static_cast<float>(b) * CommonUtils::kLumaB;
        }
    }
}
