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

    // keep the mantissa near 1: if (m < sqrt(0.5)) { e -= 1; m = 2m - 1; } else { m -= 1; }
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

    // the block is eight 64-bit values (little endian word pairs), pair j uses the top 24 bits of values 2j and 2j + 1: the high words >> 8.
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
