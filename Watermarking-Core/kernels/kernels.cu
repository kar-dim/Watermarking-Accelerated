#include "kernels.cuh"
#include <cstdint>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/*!
 *  \brief  CUDA device kernels for watermark embedding, prediction error filtering, and transforms
 *  \author Dimitris Karatzas
 */

__global__ void apply_watermark_row_major(
    const float* __restrict__ input, const __half* __restrict__ u, const uint64_t* __restrict__ sumSq, uint8_t* __restrict__ output, const float strengthNumerator, const int width, const int height) {
    __shared__ uint8_t tile[32][36];
    const float strength = embedStrength(sumSq, strengthNumerator);
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

__global__ void yuv420pToNV12(const uint8_t* __restrict__ uvSrc, uint8_t* __restrict__ uvDst, const int uvPitch, const int uvWidth, const int uvHeight) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= uvWidth || y >= uvHeight)
        return;
    const int idx = y * uvWidth + x;
    // one vector store per U,V pair
    reinterpret_cast<uchar2*>(uvDst + y * uvPitch)[x] = make_uchar2(uvSrc[idx], uvSrc[uvWidth * uvHeight + idx]);
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

__global__ void colMajorToInterleavedU8(const uint8_t* __restrict__ src, uint8_t* __restrict__ dst, const int width, const int height, const int channels) {
    __shared__ uint8_t tile[3][32][33];
    const int planeSize = width * height;
    const int tileCol = blockIdx.x * 32;
    const int tileRow = blockIdx.y * 32;
    const int col = tileCol + threadIdx.y;
    const int row = tileRow + threadIdx.x;
    for (int channel = 0; channel < channels; channel++) {
#pragma unroll
        for (int i = 0; i < 32; i += 8) {
            if ((col + i) < width && row < height)
                tile[channel][threadIdx.y + i][threadIdx.x] = src[channel * planeSize + (col + i) * height + row];
        }
    }
    __syncthreads();
    const int rowBytes = min(32, width - tileCol) * channels;
    for (int i = threadIdx.y; i < 32 && (tileRow + i) < height; i += 8) {
        uint8_t* output = dst + ((tileRow + i) * width + tileCol) * channels;
        for (int byte = threadIdx.x; byte < rowBytes; byte += 32)
            output[byte] = tile[byte % channels][byte / channels][i];
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

namespace {
// the 10-bit chroma of a luma pixel: bilinear between the nearest samples of the interleaved P010 chroma plane at the chroma siting, edges replicated
__device__ __forceinline__ float2 hdrChromaAt(
    const uint16_t* __restrict__ uvSrc, const int uvPitchBytes, const int chromaWidth, const int chromaHeight, const int x, const int y, const float siteX, const float siteY) {
    const float u = (static_cast<float>(x) - siteX) * 0.5f;
    const float v = (static_cast<float>(y) - siteY) * 0.5f;
    const float u0 = floorf(u), v0 = floorf(v);
    const int x0 = clamp(static_cast<int>(u0), 0, chromaWidth - 1), x1 = clamp(static_cast<int>(u0) + 1, 0, chromaWidth - 1);
    const int y0 = clamp(static_cast<int>(v0), 0, chromaHeight - 1), y1 = clamp(static_cast<int>(v0) + 1, 0, chromaHeight - 1);
    // each 32-bit word holds a Cb (low) and Cr (high) sample
    const auto* row0 = reinterpret_cast<const uint32_t*>(reinterpret_cast<const uint8_t*>(uvSrc) + y0 * uvPitchBytes);
    const auto* row1 = reinterpret_cast<const uint32_t*>(reinterpret_cast<const uint8_t*>(uvSrc) + y1 * uvPitchBytes);
    const auto sample = [](const uint32_t pair) { return make_float2(static_cast<float>((pair & 0xFFFF) >> 6), static_cast<float>(pair >> 22)); };
    const float2 c00 = sample(__ldg(row0 + x0)), c10 = sample(__ldg(row0 + x1)), c01 = sample(__ldg(row1 + x0)), c11 = sample(__ldg(row1 + x1));
    const float fu = u - u0, fv = v - v0;
    const float2 top = make_float2(fmaf(fu, c10.x - c00.x, c00.x), fmaf(fu, c10.y - c00.y, c00.y));
    const float2 bottom = make_float2(fmaf(fu, c11.x - c01.x, c01.x), fmaf(fu, c11.y - c01.y, c01.y));
    return make_float2(fmaf(fv, bottom.x - top.x, top.x), fmaf(fv, bottom.y - top.y, top.y));
}

// the 2:1 bilinear (triangle) filter weight of a luma tap at offset from the chroma pair's first pixel
__device__ __forceinline__ float decimationWeight(const int offset, const float site) { return fmaxf(0.0f, 1.0f - fabsf(static_cast<float>(offset) - site) * 0.5f) * 0.5f; }

constexpr int HDR_TILE_WIDTH = 66;
constexpr int HDR_TILE_HEIGHT = 18;
} // namespace

__global__ void yuv420p10ToP010(const uint16_t* __restrict__ planar, uint16_t* __restrict__ yDst, uint16_t* __restrict__ uvDst, const int width, const int height) {
    const int cx = blockIdx.x * blockDim.x + threadIdx.x;
    const int cy = blockIdx.y * blockDim.y + threadIdx.y;
    const int chromaWidth = width / 2, chromaHeight = height / 2;
    if (cx >= chromaWidth || cy >= chromaHeight)
        return;
    // two luma samples per 32-bit word
    for (int row = 2 * cy; row < 2 * cy + 2; ++row) {
        const uint32_t pair = reinterpret_cast<const uint32_t*>(planar + row * width)[cx];
        reinterpret_cast<uint32_t*>(yDst + row * width)[cx] = (pair << 6) & 0xFFC0FFC0u;
    }
    const int chroma = cy * chromaWidth + cx, pixels = width * height;
    reinterpret_cast<ushort2*>(uvDst + cy * width)[cx] = make_ushort2(static_cast<uint16_t>(planar[pixels + chroma] << 6), static_cast<uint16_t>(planar[pixels + pixels / 4 + chroma] << 6));
}

__global__ void __launch_bounds__(256) p010HdrToSdr(const uint16_t* __restrict__ ySrc, const int yPitchBytes, const uint16_t* __restrict__ uvSrc, const int uvPitchBytes, uint8_t* __restrict__ yDst,
    const int yDstPitch, float* __restrict__ lumaDst, uint8_t* __restrict__ uvDst, const int uvDstPitch, const int width, const int height, const float siteX, const float siteY, const float mobA,
    const float mobB, const float mobK) {
    __shared__ float2 chroma[HDR_TILE_HEIGHT][HDR_TILE_WIDTH];
    // +1 column: the column-major write below reads down the columns without bank conflicts
    __shared__ float luma[16][65];
    constexpr float CB_SCALE = 1.0f / 1.8556f; // 1 / (2*(1-Kb))
    constexpr float CR_SCALE = 1.0f / 1.5748f; // 1 / (2*(1-Kr))
    const int thread = threadIdx.y * 32 + threadIdx.x;
    const int left = blockIdx.x * 64 - 1, top = blockIdx.y * 16 - 1;
    const int yPitchU16 = yPitchBytes / 2;
    const int chromaWidth = width / 2, chromaHeight = height / 2;
    for (int index = thread; index < HDR_TILE_WIDTH * HDR_TILE_HEIGHT; index += 256) {
        const int tileX = index % HDR_TILE_WIDTH, tileY = index / HDR_TILE_WIDTH;
        const int x = clamp(left + tileX, 0, width - 1), y = clamp(top + tileY, 0, height - 1);
        const float3 rgb = hdrPixelToSdrRgb(__ldg(&ySrc[y * yPitchU16 + x]), hdrChromaAt(uvSrc, uvPitchBytes, chromaWidth, chromaHeight, x, y, siteX, siteY), mobA, mobB, mobK);
        const float Y = fmaf(0.2126f, rgb.x, fmaf(0.7152f, rgb.y, 0.0722f * rgb.z));
        chroma[tileY][tileX] = make_float2((rgb.z - Y) * CB_SCALE, (rgb.x - Y) * CR_SCALE);
        // the block's own pixels give the luma outputs
        if (tileX >= 1 && tileX <= 64 && tileY >= 1 && tileY <= 16 && left + tileX < width && top + tileY < height) {
            const float scaled = clamp(fmaf(Y, 219.0f, 16.0f), 0.0f, 255.0f);
            if (yDst)
                yDst[y * yDstPitch + x] = static_cast<uint8_t>(__float2int_rn(scaled));
            luma[tileY - 1][tileX - 1] = scaled;
        }
    }
    __syncthreads();
    // column-major float luma: consecutive threads write consecutive rows of a column
    if (lumaDst)
        for (int index = thread; index < 64 * 16; index += 256) {
            const int column = index / 16, row = index % 16;
            const int x = left + 1 + column, y = top + 1 + row;
            if (x < width && y < height)
                lumaDst[static_cast<size_t>(x) * height + y] = luma[row][column];
        }
    const int cx = blockIdx.x * 32 + threadIdx.x, cy = blockIdx.y * 8 + threadIdx.y;
    if (!uvDst || cx >= chromaWidth || cy >= chromaHeight)
        return;
    const int tileX = 2 * threadIdx.x + 1, tileY = 2 * threadIdx.y + 1;
    float2 sum = make_float2(0.0f, 0.0f);
#pragma unroll
    for (int dy = -1; dy <= 2; ++dy) {
        const float rowWeight = decimationWeight(dy, siteY);
        float2 row = make_float2(0.0f, 0.0f);
#pragma unroll
        for (int dx = -1; dx <= 2; ++dx) {
            const float weight = decimationWeight(dx, siteX);
            const float2 value = chroma[tileY + dy][tileX + dx];
            row = make_float2(fmaf(weight, value.x, row.x), fmaf(weight, value.y, row.y));
        }
        sum = make_float2(fmaf(rowWeight, row.x, sum.x), fmaf(rowWeight, row.y, sum.y));
    }
    // Cb and Cr in one 16-bit store, consecutive in the warp
    const uint8_t cb8 = static_cast<uint8_t>(__float2int_rn(clamp(fmaf(sum.x, 224.0f, 128.0f), 0.0f, 255.0f)));
    const uint8_t cr8 = static_cast<uint8_t>(__float2int_rn(clamp(fmaf(sum.y, 224.0f, 128.0f), 0.0f, 255.0f)));
    reinterpret_cast<uchar2*>(uvDst + cy * uvDstPitch)[cx] = make_uchar2(cb8, cr8);
}

// // The CUDA watermark generation. For bit-exactness we use fma(), correctly rounded sqrt and explicit fmaf
namespace {
__device__ __forceinline__ uint32_t rotl(const uint32_t value, const int shift) { return __funnelshift_l(value, value, shift); }

// ChaCha20 quarter round: add, rotate, xor
__device__ __forceinline__ void quarterRound(uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d) {
    a += b;
    d = rotl(d ^ a, 16);
    c += d;
    b = rotl(b ^ c, 12);
    a += b;
    d = rotl(d ^ a, 8);
    c += d;
    b = rotl(b ^ c, 7);
}

// natural logarithm (Cephes polynomial)
__device__ __forceinline__ float logReference(const float x) {
    // frexp: mantissa in [0.5, 1) plus the matching exponent
    const uint32_t bits = __float_as_uint(x);
    float m = __uint_as_float((bits & 0x807FFFFFu) | 0x3F000000u);
    float e = static_cast<float>(static_cast<int>((bits & 0x7F800000u) >> 23) - 126);

    // keep mantissa near 1: when m < sqrt(0.5) decrease exponent and scale m, otherwise subtract 1
    const bool belowSqrtHalf = m < 0.707106781186547524f;
    e = belowSqrtHalf ? e - 1.0f : e;
    m = (belowSqrtHalf ? m + m : m) - 1.0f;

    const float z = m * m;
    float y = 7.0376836292E-2f;
    y = fmaf(y, m, -1.1514610310E-1f);
    y = fmaf(y, m, 1.1676998740E-1f);
    y = fmaf(y, m, -1.2420140846E-1f);
    y = fmaf(y, m, 1.4249322787E-1f);
    y = fmaf(y, m, -1.6668057665E-1f);
    y = fmaf(y, m, 2.0000714765E-1f);
    y = fmaf(y, m, -2.4999993993E-1f);
    y = fmaf(y, m, 3.3333331174E-1f);
    y = (y * m) * z;
    // recombine, using the hi/lo split of ln(2) for the exponent term
    y = fmaf(e, -2.12194440e-4f, y);
    y = fmaf(z, -0.5f, y);
    return fmaf(e, 0.693359375f, m + y);
}

// sine and cosine (Cephes minimax polynomials)
__device__ __forceinline__ void sinCosReference(const float x, float& sinOut, float& cosOut) {
    // reduce into an octant: j = ((int)(x * 4/pi) + 1) & ~1
    const int j = (static_cast<int>(x * 1.27323954473516f) + 1) & ~1;
    const float y = static_cast<float>(j);

    // extended precision modular arithmetic: r = ((x - y*DP1) - y*DP2) - y*DP3
    float r = fmaf(y, -0.78515625f, x);
    r = fmaf(y, -2.4187564849853515625e-4f, r);
    r = fmaf(y, -3.77489497744594108e-8f, r);

    // cosine and sine polynomials of the reduced argument
    const float z = r * r;
    float cosPoly = 2.443315711809948E-005f;
    cosPoly = fmaf(cosPoly, z, -1.388731625493765E-003f);
    cosPoly = fmaf(cosPoly, z, 4.166664568298827E-002f);
    cosPoly = (cosPoly * z) * z;
    cosPoly = fmaf(z, -0.5f, cosPoly);
    cosPoly = cosPoly + 1.0f;
    float sinPoly = -1.9515295891E-4f;
    sinPoly = fmaf(sinPoly, z, 8.3321608736E-3f);
    sinPoly = fmaf(sinPoly, z, -1.6666654611E-1f);
    sinPoly = fmaf(sinPoly * z, r, r);

    // the octant bits decide which polynomial is sin and which is cos, and the sign of each
    const bool sinFromSinPoly = (j & 2) == 0;
    sinOut = __uint_as_float(__float_as_uint(sinFromSinPoly ? sinPoly : cosPoly) ^ (static_cast<uint32_t>(j & 4) << 29));
    cosOut = __uint_as_float(__float_as_uint(sinFromSinPoly ? cosPoly : sinPoly) ^ (static_cast<uint32_t>(~(j - 2) & 4) << 29));
}

// two 24-bit random values -> two normally distributed values (Box-Muller transform)
__device__ __forceinline__ float2 boxMullerPair(const uint32_t random1, const uint32_t random2) {
    // uniform in (0, 1]: (random + 1) * 2^-24, exact
    const float u1 = fmaf(static_cast<float>(random1), 0x1.0p-24f, 0x1.0p-24f);
    const float u2 = fmaf(static_cast<float>(random2), 0x1.0p-24f, 0x1.0p-24f);
    const float radius = __fsqrt_rn(-2.0f * logReference(u1));
    float sinTheta, cosTheta;
    sinCosReference(CommonUtils::kTwoPi * u2, sinTheta, cosTheta);
    return make_float2(radius * cosTheta, radius * sinTheta);
}
} // namespace

__global__ void generate_watermark(const ChaChaState baseState, __half* __restrict__ watermark, const int64_t numElements) {
    const int64_t block = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t first = block * 8;
    if (first >= numElements)
        return;
    // ChaCha20 block "block": the 64-bit block counter in words 12-13, 20 rounds, then the final addition
    uint32_t x[16];
#pragma unroll
    for (int i = 0; i < 16; i++)
        x[i] = baseState.words[i];
    x[12] = static_cast<uint32_t>(block);
    x[13] = static_cast<uint32_t>(block >> 32);
#pragma unroll
    for (int i = 0; i < 10; i++) {
        quarterRound(x[0], x[4], x[8], x[12]);
        quarterRound(x[1], x[5], x[9], x[13]);
        quarterRound(x[2], x[6], x[10], x[14]);
        quarterRound(x[3], x[7], x[11], x[15]);
        quarterRound(x[0], x[5], x[10], x[15]);
        quarterRound(x[1], x[6], x[11], x[12]);
        quarterRound(x[2], x[7], x[8], x[13]);
        quarterRound(x[3], x[4], x[9], x[14]);
    }
    const uint32_t counter[2] = {static_cast<uint32_t>(block), static_cast<uint32_t>(block >> 32)};
#pragma unroll
    for (int i = 0; i < 16; i++)
        x[i] += (i == 12 || i == 13) ? counter[i - 12] : baseState.words[i];

    // the block is eight 64-bit values (little endian word pairs), pair j uses top 24 bits of values 2j and 2j + 1: high words >> 8
    // The 8 values as half, two per word (the first one in the low bits, the memory order)
    uint32_t packed[4];
#pragma unroll
    for (int j = 0; j < 4; j++) {
        const float2 normals = boxMullerPair(x[(4 * j) + 1] >> 8, x[(4 * j) + 3] >> 8);
        const __half2 pair = __floats2half2_rn(normals.x, normals.y);
        packed[j] = static_cast<uint32_t>(__half_as_ushort(pair.x)) | (static_cast<uint32_t>(__half_as_ushort(pair.y)) << 16);
    }
    if (first + 8 <= numElements) {
        *reinterpret_cast<uint4*>(watermark + first) = make_uint4(packed[0], packed[1], packed[2], packed[3]);
        return;
    }
    // the partial last block (at most 7 values): unrolled, every index is a constant and the values stay in registers
    const int64_t remaining = numElements - first;
#pragma unroll
    for (int i = 0; i < 7; i++) {
        if (i < remaining)
            watermark[first + i] = __ushort_as_half(static_cast<unsigned short>(packed[i / 2] >> (16 * (i & 1))));
    }
}

__global__ void box_muller_pairs(const uint32_t* __restrict__ randomPairs, float* __restrict__ normals, const int pairs) {
    const int pair = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair < pairs)
        reinterpret_cast<float2*>(normals)[pair] = boxMullerPair(randomPairs[2 * pair], randomPairs[(2 * pair) + 1]);
}
