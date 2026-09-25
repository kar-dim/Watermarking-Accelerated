#pragma once
#include "luma_coefficients.hpp"
#include <cstdint>
#include <cub/cub.cuh>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <type_traits>

// Convert float to fixed point uint64 for deterministic atomic additions
__device__ inline uint64_t toScaledUint64(float value) { return static_cast<uint64_t>(value * 1000000000.0f); }
__device__ inline float toUnscaledFloat(uint64_t value) { return static_cast<float>(value * 1.0e-9f); }

inline constexpr float kMeMaskPrescale = 1.0f / 255.0f;

// Max functor for CUB block reductions
struct MaxOp {
    __device__ __forceinline__ float operator()(const float a, const float b) const { return fmaxf(a, b); }
};

// Stores the three correlation sums reduced together by CUB
struct CorrelationData {
    float dot;
    float normU;
    float normZ;
    __device__ __forceinline__ CorrelationData operator+(const CorrelationData& other) const { return {dot + other.dot, normU + other.normU, normZ + other.normZ}; }
};

// Converts a packed lower triangular index to row and column coordinates
__device__ inline int2 packedToRowCol(const int k) {
    // Inverse triangular formula to find row coordinate
    const int r = __float2int_rd(0.5f * (sqrtf(1.0f + 8.0f * k) - 1.0f));
    const int c = k - (r * (r + 1)) / 2;
    return make_int2(r, c);
}

// Clamps a value between lower and upper bounds
inline __device__ float clamp(float f, float a, float b) { return fmaxf(a, fminf(f, b)); }
inline __device__ int clamp(int f, int a, int b) { return max(a, min(f, b)); }

// Fills shared memory with the image tile around block outputs with clamped edges
// Loads all values into registers first before writing to shared memory
template <bool FUSED, int p, int shDimFast, int shDimSlow, int THREADS, bool ABS_A = false>
__device__ __forceinline__ void fillBlock(
    const float* __restrict__ inputA, const __half* __restrict__ inputB, float* __restrict__ sharedMem, const int tileSlow0, const int tileFast0, const int width, const int height) {
    constexpr int pad = p / 2;
    constexpr int totalElements = shDimFast * shDimSlow;
    constexpr int iterations = (totalElements + THREADS - 1) / THREADS;

    const int baseGlobalX = tileSlow0 - pad;
    const int baseGlobalY = tileFast0 - pad;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    float values[iterations];
#pragma unroll
    for (int it = 0; it < iterations; it++) {
        const int i = tid + (it * THREADS);
        if (i < totalElements) {
            const int r = i % shDimFast;
            const int c = i / shDimFast;
            const int globalX = clamp(baseGlobalX + c, 0, width - 1);
            const int globalY = clamp(baseGlobalY + r, 0, height - 1);
            const int idx = globalX * height + globalY;
            float val = inputA[idx];
            // Evaluated at compile time to avoid branches
            if constexpr (ABS_A)
                val = fabsf(val);
            if constexpr (FUSED)
                val *= __half2float(inputB[idx]);
            values[it] = val;
        }
    }
#pragma unroll
    for (int it = 0; it < iterations; it++) {
        const int i = tid + (it * THREADS);
        if (i < totalElements)
            sharedMem[i] = values[it];
    }
}

// Computes the NVF mask for one pixel from its shared memory window
template <int p, int shDimFast, int shDimSlow>
__device__ __forceinline__ float compute_nvf_mask(const float (&region)[shDimSlow][shDimFast], const int shSlow, const int shFast) {
    constexpr int pad = p / 2;
    constexpr float nPixels = static_cast<float>(p * p);
    constexpr float nPixelsSq = nPixels * nPixels;

    float sum = 0.0f, sumSq = 0.0f;

#pragma unroll
    for (int i = -pad; i <= pad; i++) {
#pragma unroll
        for (int j = -pad; j <= pad; j++) {
            const float pixelValue = region[shSlow + i][shFast + j];
            sum += pixelValue;
            sumSq += pixelValue * pixelValue;
        }
    }

    // NVF formula using variance with a single division
    const float numerator = (nPixels * sumSq) - (sum * sum);
    const float output = __fdividef(numerator, nPixelsSq + numerator);
    return __saturatef(output);
}

// Computes the NVF mask for every pixel during detection
template <int p>
__global__ void nvf(const float* __restrict__ input, float* __restrict__ nvf, const int width, const int height) {
    constexpr int pad = p / 2;
    constexpr int shDimFast = 32 + (2 * pad);
    constexpr int shDimSlow = 8 + (2 * pad);

    const int x = blockIdx.y * blockDim.y + threadIdx.y;
    const int y = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ alignas(16) float region[shDimSlow][shDimFast];

    fillBlock<false, p, shDimFast, shDimSlow, 32 * 8>(input, nullptr, &region[0][0], blockIdx.y * blockDim.y, blockIdx.x * blockDim.x, width, height);
    __syncthreads();

    if (x >= width || y >= height)
        return;

    const int shSlow = threadIdx.y + pad;
    const int shFast = threadIdx.x + pad;
    nvf[x * height + y] = compute_nvf_mask<p, shDimFast, shDimSlow>(region, shSlow, shFast);
}

// Fused kernel computing NVF mask, u = mask * w, and sum of u squared
template <int p>
__global__ void nvf_u_and_sumsq_fused(const float* __restrict__ input, const __half* __restrict__ w, __half* __restrict__ u, uint64_t* __restrict__ globalSumSq, const int width, const int height) {
    constexpr int pad = p / 2;
    constexpr int shDimFast = 32 + (2 * pad);
    constexpr int shDimSlow = 8 + (2 * pad);

    const int x = blockIdx.y * blockDim.y + threadIdx.y;
    const int y = blockIdx.x * blockDim.x + threadIdx.x;
    const int linearTid = threadIdx.y * blockDim.x + threadIdx.x;

    using BlockReduceT = cub::BlockReduce<float, 32, cub::BLOCK_REDUCE_WARP_REDUCTIONS, 8>; // Block size is 32 by 8
    __shared__ alignas(16) float region[shDimSlow][shDimFast];
    __shared__ typename BlockReduceT::TempStorage temp_storage;

    fillBlock<false, p, shDimFast, shDimSlow, 32 * 8>(input, nullptr, &region[0][0], blockIdx.y * blockDim.y, blockIdx.x * blockDim.x, width, height);
    __syncthreads();

    // Threads outside image bounds contribute zero
    float threadSumSq = 0.0f;
    if (x < width && y < height) {
        const int shSlow = threadIdx.y + pad;
        const int shFast = threadIdx.x + pad;
        // Accumulate squared u values in half precision
        const float maskVal = compute_nvf_mask<p, shDimFast, shDimSlow>(region, shSlow, shFast);
        const int idx = x * height + y;
        const __half uHalf = __float2half_rn(maskVal * __half2float(w[idx]));
        u[idx] = uHalf;
        const float uVal = __half2float(uHalf);
        threadSumSq = uVal * uVal;
    }

    // Reduce block sum and update global total with atomic addition
    const float blockTotalSq = BlockReduceT(temp_storage).Sum(threadSumSq);
    if (linearTid == 0)
        atomicAdd(globalSumSq, toScaledUint64(blockTotalSq));
}

// Prediction error tile where each thread processes 4 rows by 2 columns
// Reuses loaded window columns across outputs while keeping coefficients in registers
template <int p>
struct ErrorTile {
    static constexpr int pad = p / 2;
    static constexpr int coeffsSize = (p * p) - 1;
    static constexpr int threads = 256;
    static constexpr int rows = 4; // Rows per thread along fast dimension
    static constexpr int cols = 2; // Columns per thread along slow dimension
    // Warp threads cover tile rows while warps step across columns
    static constexpr int tileFast = 32 * rows;
    static constexpr int tileSlow = (threads / 32) * cols;
    // Per-thread column window rounded up to float4 alignment
    static constexpr int windowFast = ((rows + (2 * pad) + 3) / 4) * 4;
    // Shared memory tile sized to cover padded windows across all threads
    static constexpr int shFast = tileFast - rows + windowFast;
    static constexpr int shSlow = tileSlow + (2 * pad);

    __host__ static dim3 grid(const int width, const int height) { return dim3((height + tileFast - 1) / tileFast, (width + tileSlow - 1) / tileSlow); }
};

// Loads prediction coefficients into registers using float4 loads
template <int p>
__device__ __forceinline__ void loadCoefficients(const float* __restrict__ coeffs, float (&coef)[ErrorTile<p>::coeffsSize]) {
#pragma unroll
    for (int v = 0; v < ErrorTile<p>::coeffsSize / 4; v++) {
        const float4 c = __ldg(reinterpret_cast<const float4*>(coeffs) + v);
        coef[(4 * v) + 0] = c.x;
        coef[(4 * v) + 1] = c.y;
        coef[(4 * v) + 2] = c.z;
        coef[(4 * v) + 3] = c.w;
    }
}

// Computes prediction errors for all outputs assigned to this thread
template <int p>
__device__ __forceinline__ void predictionErrorTile(
    const float* __restrict__ region, const float (&coef)[ErrorTile<p>::coeffsSize], const int slow0, const int fast0, float (&error)[ErrorTile<p>::cols][ErrorTile<p>::rows]) {
    using T = ErrorTile<p>;
    constexpr int center = (p * p) / 2;
    float dot[T::cols][T::rows] = {};
    float pixel[T::cols][T::rows];
    // Maps window column wc to output column c
#pragma unroll
    for (int wc = 0; wc < p + T::cols - 1; wc++) {
        float window[T::windowFast];
        const float4* src = reinterpret_cast<const float4*>(region + ((slow0 + wc) * T::shFast) + fast0);
#pragma unroll
        for (int v = 0; v < T::windowFast / 4; v++) {
            const float4 value = src[v];
            window[(4 * v) + 0] = value.x;
            window[(4 * v) + 1] = value.y;
            window[(4 * v) + 2] = value.z;
            window[(4 * v) + 3] = value.w;
        }
#pragma unroll
        for (int c = 0; c < T::cols; c++) {
            const int i = wc - c;
            if (i < 0 || i >= p)
                continue;
#pragma unroll
            for (int j = 0; j < p; j++) {
                const int k = (i * p) + j;
                if (k == center) {
#pragma unroll
                    for (int r = 0; r < T::rows; r++)
                        pixel[c][r] = window[r + j];
                    continue;
                }
#pragma unroll
                for (int r = 0; r < T::rows; r++)
                    dot[c][r] += coef[k - (k > center)] * window[r + j];
            }
        }
    }
#pragma unroll
    for (int c = 0; c < T::cols; c++)
#pragma unroll
        for (int r = 0; r < T::rows; r++)
            error[c][r] = pixel[c][r] - dot[c][r];
}

// Loads 4 consecutive rows using float4 when aligned or scalar reads otherwise
__device__ __forceinline__ float4 loadRows4(const float* __restrict__ input, const int x, const int y0, const int height) {
    const int idx = (x * height) + y0;
    if ((height & 3) == 0)
        return y0 < height ? __ldg(reinterpret_cast<const float4*>(input + idx)) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    return make_float4(y0 < height ? input[idx] : 0.0f, y0 + 1 < height ? input[idx + 1] : 0.0f, y0 + 2 < height ? input[idx + 2] : 0.0f, y0 + 3 < height ? input[idx + 3] : 0.0f);
}
__device__ __forceinline__ void storeRows4(float* __restrict__ output, const int x, const int y0, const int height, const float4 value) {
    const int idx = (x * height) + y0;
    if ((height & 3) == 0) {
        if (y0 < height)
            *reinterpret_cast<float4*>(output + idx) = value;
        return;
    }
    const float v[4] = {value.x, value.y, value.z, value.w};
#pragma unroll
    for (int r = 0; r < 4; r++)
        if (y0 + r < height)
            output[idx + r] = v[r];
}

// Stores 4 half values packed into an 8-byte structure
struct alignas(8) Half4 {
    __half2 lo, hi;
};
__device__ __forceinline__ float4 toFloat4(const Half4 v) {
    const float2 lo = __half22float2(v.lo);
    const float2 hi = __half22float2(v.hi);
    return make_float4(lo.x, lo.y, hi.x, hi.y);
}
__device__ __forceinline__ float roundToHalf(const float value) { return __half2float(__float2half_rn(value)); }
__device__ __forceinline__ float4 loadRows4(const __half* __restrict__ input, const int x, const int y0, const int height) {
    const int idx = (x * height) + y0;
    if ((height & 3) == 0)
        return y0 < height ? toFloat4(*reinterpret_cast<const Half4*>(input + idx)) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    return make_float4(y0 < height ? __half2float(input[idx]) : 0.0f, y0 + 1 < height ? __half2float(input[idx + 1]) : 0.0f, y0 + 2 < height ? __half2float(input[idx + 2]) : 0.0f,
        y0 + 3 < height ? __half2float(input[idx + 3]) : 0.0f);
}
__device__ __forceinline__ void storeRows4(__half* __restrict__ output, const int x, const int y0, const int height, const float4 value) {
    const int idx = (x * height) + y0;
    if ((height & 3) == 0) {
        if (y0 < height)
            *reinterpret_cast<Half4*>(output + idx) = {__floats2half2_rn(value.x, value.y), __floats2half2_rn(value.z, value.w)};
        return;
    }
    const float v[4] = {value.x, value.y, value.z, value.w};
#pragma unroll
    for (int r = 0; r < 4; r++)
        if (y0 + r < height)
            output[idx + r] = __float2half_rn(v[r]);
}

// Prepares shared memory tile and coefficients, returning thread start coordinates
template <int p, bool FUSED, bool ABS_A = false>
__device__ __forceinline__ int2 setupErrorTile(const float* __restrict__ inputA, const __half* __restrict__ inputB, const float* __restrict__ coeffs, float* __restrict__ region,
    float (&coef)[ErrorTile<p>::coeffsSize], const int width, const int height) {
    using T = ErrorTile<p>;
    const int tileSlow0 = blockIdx.y * T::tileSlow;
    const int tileFast0 = blockIdx.x * T::tileFast;
    fillBlock<FUSED, p, T::shFast, T::shSlow, T::threads, ABS_A>(inputA, inputB, region, tileSlow0, tileFast0, width, height);
    loadCoefficients<p>(coeffs, coef);
    __syncthreads();
    return make_int2(tileSlow0 + ((threadIdx.x / 32) * T::cols), tileFast0 + ((threadIdx.x % 32) * T::rows));
}

// Computes prediction error across the image for detection
template <int p>
__global__ void __launch_bounds__(ErrorTile<p>::threads)
    calculate_error_sequence(const float* __restrict__ input, float* __restrict__ errorOut, const float* __restrict__ coeffs, const int width, const int height, const int* __restrict__ stopFlag) {
    using T = ErrorTile<p>;
    __shared__ alignas(16) float region[T::shSlow * T::shFast];
    float coef[T::coeffsSize];
    const int2 origin = setupErrorTile<p, false>(input, nullptr, coeffs, region, coef, width, height);

    float error[T::cols][T::rows] = {};
    if (!(*stopFlag))
        predictionErrorTile<p>(region, coef, (threadIdx.x / 32) * T::cols, (threadIdx.x % 32) * T::rows, error);
#pragma unroll
    for (int c = 0; c < T::cols; c++)
        if (origin.x + c < width)
            storeRows4(errorOut, origin.x + c, origin.y, height, make_float4(error[c][0], error[c][1], error[c][2], error[c][3]));
}

// Computes prediction error, creates u = (|e| / 255) * w, and tracks energy sums
template <int p>
__global__ void __launch_bounds__(ErrorTile<p>::threads) me_error_sequence_u_sumsq_fused(const float* __restrict__ input, const __half* __restrict__ w, __half* __restrict__ u,
    const float* __restrict__ coeffs, uint64_t* __restrict__ globalSumSqMax, const int width, const int height, const int* __restrict__ stopFlag) {
    using T = ErrorTile<p>;
    using BlockReduceT = cub::BlockReduce<float, T::threads, cub::BLOCK_REDUCE_WARP_REDUCTIONS>;
    __shared__ typename BlockReduceT::TempStorage sumStorage;
    __shared__ typename BlockReduceT::TempStorage maxStorage;
    __shared__ alignas(16) float region[T::shSlow * T::shFast];
    float coef[T::coeffsSize];
    const int2 origin = setupErrorTile<p, false>(input, nullptr, coeffs, region, coef, width, height);

    float error[T::cols][T::rows] = {};
    if (!(*stopFlag)) // Skip computation if system solve failed
        predictionErrorTile<p>(region, coef, (threadIdx.x / 32) * T::cols, (threadIdx.x % 32) * T::rows, error);
    // Outputs outside image bounds contribute zero
    float threadSumSq = 0.0f;
    float threadMax = 0.0f;
#pragma unroll
    for (int c = 0; c < T::cols; c++) {
        if (origin.x + c >= width)
            continue;
        const float4 wv = loadRows4(w, origin.x + c, origin.y, height);
        const float wr[4] = {wv.x, wv.y, wv.z, wv.w};
        float uv[4];
#pragma unroll
        for (int r = 0; r < T::rows; r++) {
            const float absError = origin.y + r < height ? fabsf(error[c][r]) : 0.0f;
            uv[r] = roundToHalf(absError * kMeMaskPrescale * wr[r]);
            threadSumSq += uv[r] * uv[r];
            threadMax = fmaxf(threadMax, absError);
        }
        storeRows4(u, origin.x + c, origin.y, height, make_float4(uv[0], uv[1], uv[2], uv[3]));
    }

    // Block reductions for sum and maximum, followed by atomic updates
    const float blockTotalSq = BlockReduceT(sumStorage).Sum(threadSumSq);
    const float blockMax = BlockReduceT(maxStorage).Reduce(threadMax, MaxOp{});
    if (threadIdx.x == 0) {
        atomicAdd(globalSumSqMax, toScaledUint64(blockTotalSq));
        atomicMax(reinterpret_cast<unsigned long long*>(globalSumSqMax + 1), static_cast<unsigned long long>(__float_as_uint(blockMax)));
    }
}

// Computes partial correlation sums and writes final result from the last finished block
template <int p, bool ABS_MASK>
__global__ void __launch_bounds__(ErrorTile<p>::threads) calculate_error_sequence_and_partial_corr_fused(const float* __restrict__ mask, const __half* __restrict__ w, const float* __restrict__ e_u,
    const float* __restrict__ coeffs, float* __restrict__ partialDots, float* __restrict__ partialNormU, float* __restrict__ partialNormZ, unsigned int* __restrict__ blockCounter,
    float* __restrict__ correlation, const int width, const int height, const int* __restrict__ stopFlag) {
    using T = ErrorTile<p>;
    using BlockReduceT = cub::BlockReduce<CorrelationData, T::threads, cub::BLOCK_REDUCE_WARP_REDUCTIONS>;
    __shared__ typename BlockReduceT::TempStorage temp_storage;
    __shared__ alignas(16) float region[T::shSlow * T::shFast];
    float coef[T::coeffsSize];
    const int2 origin = setupErrorTile<p, true, ABS_MASK>(mask, w, coeffs, region, coef, width, height);

    // Out-of-bounds pixels or failed solves contribute zero
    CorrelationData threadData = {0.0f, 0.0f, 0.0f};
    if (!(*stopFlag)) {
        float ez[T::cols][T::rows];
        predictionErrorTile<p>(region, coef, (threadIdx.x / 32) * T::cols, (threadIdx.x % 32) * T::rows, ez);
#pragma unroll
        for (int c = 0; c < T::cols; c++) {
            if (origin.x + c >= width)
                continue;
            // e_u reads zero outside image bounds
            const float4 euv = loadRows4(e_u, origin.x + c, origin.y, height);
            const float eu[4] = {euv.x, euv.y, euv.z, euv.w};
#pragma unroll
            for (int r = 0; r < T::rows; r++) {
                const float z = origin.y + r < height ? ez[c][r] : 0.0f;
                threadData.dot += eu[r] * z;
                threadData.normU += eu[r] * eu[r];
                threadData.normZ += z * z;
            }
        }
    }

    // Thread 0 writes block reduction totals
    __shared__ bool isLastBlock;
    const int numBlocks = gridDim.x * gridDim.y;
    const CorrelationData blockSum = BlockReduceT(temp_storage).Sum(threadData);
    if (threadIdx.x == 0) {
        const int blockIdxFlat = blockIdx.y * gridDim.x + blockIdx.x;
        partialDots[blockIdxFlat] = blockSum.dot;
        partialNormU[blockIdxFlat] = blockSum.normU;
        partialNormZ[blockIdxFlat] = blockSum.normZ;
        // Ensure block results are visible before atomic counter increment
        __threadfence();
        isLastBlock = atomicAdd(blockCounter, 1u) == static_cast<unsigned int>(numBlocks - 1);
    }
    __syncthreads();
    if (!isLastBlock)
        return;

    // Final block sums all block outputs and writes normalized correlation
    CorrelationData total = {0.0f, 0.0f, 0.0f};
    for (int i = threadIdx.x; i < numBlocks; i += T::threads) {
        total.dot += __ldcg(partialDots + i);
        total.normU += __ldcg(partialNormU + i);
        total.normZ += __ldcg(partialNormZ + i);
    }
    const CorrelationData sum = BlockReduceT(temp_storage).Sum(total);
    if (threadIdx.x == 0) {
        const float normU = sqrtf(sum.normU);
        const float normZ = sqrtf(sum.normZ);
        *correlation = (normU > 1e-12f && normZ > 1e-12f) ? (sum.dot / (normU * normZ)) : 0.0f;
        *blockCounter = 0; // Reset counter for next launch
    }
}

// Converts solver variable index to row-major coefficient index
template <int p>
__device__ __forceinline__ int coefficientIndex(const int k) {
    constexpr int center = (p * p) / 2;
    constexpr int p2_minus_1 = ((p * p) - 1);

    const int kPixel = k + (k >= center);
    const int r = kPixel / p;
    const int originalPixel = (kPixel * p) - (r * p2_minus_1);
    return originalPixel - (originalPixel > center);
}

// Emulated double precision using two floats (hi + lo) for roughly 48 bits of precision
struct FloatFloat {
    float hi, lo;

    FloatFloat() = default;
    __device__ __forceinline__ FloatFloat(const float value) : hi(value), lo(0.0f) {}
    __device__ __forceinline__ FloatFloat(const float h, const float l) : hi(h), lo(l) {}

    // Exact sum of two floats using Dekker addition
    __device__ static __forceinline__ FloatFloat twoSum(const float a, const float b) {
        const float s = a + b;
        const float v = s - a;
        return {s, (a - (s - v)) + (b - v)};
    }
    // Fast exact sum when magnitude of a is greater than or equal to b
    __device__ static __forceinline__ FloatFloat quickTwoSum(const float a, const float b) {
        const float s = a + b;
        return {s, b - (s - a)};
    }
    // Exact product of two floats using FMA
    __device__ static __forceinline__ FloatFloat twoProd(const float a, const float b) {
        const float p = a * b;
        return {p, fmaf(a, b, -p)};
    }
    // Exact conversion from uint64 to float-float
    __device__ static __forceinline__ FloatFloat fromUint64(const uint64_t value) {
        const float h = __ull2float_rn(value);
        return quickTwoSum(h, static_cast<float>(static_cast<long long>(value) - static_cast<long long>(h)));
    }

    __device__ __forceinline__ explicit operator float() const { return hi + lo; }
    __device__ __forceinline__ friend FloatFloat operator*(const FloatFloat a, const FloatFloat b) {
        FloatFloat p = twoProd(a.hi, b.hi);
        p.lo += (a.hi * b.lo) + (a.lo * b.hi);
        return quickTwoSum(p.hi, p.lo);
    }
    // Computes c - a * b with fast error compensation for Cholesky updates
    __device__ __forceinline__ friend FloatFloat fnma(const FloatFloat a, const FloatFloat b, const FloatFloat c) {
        const float ph = a.hi * b.hi;
        const float pl = fmaf(a.hi, b.lo, fmaf(a.lo, b.hi, fmaf(a.hi, b.hi, -ph)));
        const FloatFloat s = twoSum(c.hi, -ph);
        return quickTwoSum(s.hi, s.lo + (c.lo - pl));
    }
    // Fast reciprocal square root refined with one float-float Newton step
    __device__ __forceinline__ friend FloatFloat rsqrt(const FloatFloat v) {
        const float h = rsqrtf(v.hi);
        const FloatFloat vh2 = v * twoProd(h, h);
        const float r = (1.0f - vh2.hi) - vh2.lo;
        return quickTwoSum(h, 0.5f * h * r);
    }
    __device__ __forceinline__ friend FloatFloat shfl(const FloatFloat v, const int srcLane) { return {__shfl_sync(0xffffffffu, v.hi, srcLane), __shfl_sync(0xffffffffu, v.lo, srcLane)}; }
};

// Autocorrelation shift layout for fast prediction system assembly
// Sums unique spatial shifts over the inner area, then adds border lines and corners
template <int p>
struct MeShiftLayout {
    static constexpr int pad = p / 2;
    static constexpr int windowSize = p * p;
    static constexpr int windowCenter = windowSize / 2;
    static constexpr int maxShift = p - 1;
    static constexpr int shiftSpan = (2 * maxShift) + 1;
    // Keep only unique shift directions (dc > 0 or dc == 0 with dr >= 0)
    static constexpr int numShifts = ((shiftSpan * shiftSpan) + 1) / 2;
    // Border rows and columns outside the inner image area
    static constexpr int borderSize = 4 * pad;
    // Layout of fixed point sums for inner area, border rows, and border columns
    static constexpr int borderRowsOffset = numShifts;
    static constexpr int borderColsOffset = numShifts + (numShifts * borderSize);
    static constexpr int shiftSumsSize = numShifts * (1 + (2 * borderSize));
    // Scale products by 1 / 255^2 so fixed point sums do not overflow
    static constexpr float productScale = 1.0f / (255.0f * 255.0f);

    __host__ __device__ static constexpr int shiftIndex(const int dr, const int dc) { return dc == 0 ? dr : (maxShift + 1) + ((dc - 1) * shiftSpan) + (dr + maxShift); }
    // First shift index for column offset dc
    __host__ __device__ static constexpr int firstShift(const int dc) { return dc == 0 ? 0 : shiftIndex(-maxShift, dc); }
    // Number of column shift groups to keep register usage in check
    static constexpr int shiftGroups = p >= 9 ? 3 : (p >= 7 ? 2 : 1);
    // Number of neighboring inner columns processed per task
    static constexpr int interiorCols = p >= 7 ? 2 : 1;
    // Number of rows processed per task
    static constexpr int interiorRun = 8;
    __host__ __device__ static constexpr int groupFirstDc(const int group) { return (group * (maxShift + 1)) / shiftGroups; }
    // One past the last shift index before column offset dc
    __host__ __device__ static constexpr int shiftEnd(const int dc) { return dc > maxShift ? numShifts : firstShift(dc); }
    // Maps a border index to the unclamped row or column coordinate
    __device__ static int borderToCoord(const int border, const int size) { return border < 2 * pad ? border - pad : size - (3 * pad) + border; }
    // Copy border rows to row-major format for p >= 7 to speed up memory access
    static constexpr bool copyBorderRows = p >= 7;
    static constexpr int borderCopyRows = 6 * pad;
    __device__ static int borderCopyRow(const int row, const int height) { return row < borderCopyRows / 2 ? row : row - height + borderCopyRows; }
};

// Reads an image pixel with edge clamping
__device__ __forceinline__ float clampedPixel(const float* __restrict__ input, const int r, const int c, const int width, const int height) {
    return input[(static_cast<size_t>(clamp(c, 0, width - 1)) * height) + clamp(r, 0, height - 1)];
}

// Reduces thread sums in the block and updates global output with atomic adds
template <int NUM>
__device__ __forceinline__ void reduceShiftSums(const float (&sums)[NUM], uint64_t* __restrict__ output, const int stride, float* __restrict__ warpSums) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
#pragma unroll
    for (int i = 0; i < NUM; i++) {
        float v = sums[i];
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            v += __shfl_xor_sync(0xFFFFFFFF, v, offset);
        if (lane == 0)
            warpSums[(warp * NUM) + i] = v;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < NUM; i += blockDim.x) {
        float v = 0.0f;
#pragma unroll
        for (int w = 0; w < 8; w++)
            v += warpSums[(w * NUM) + i];
        atomicAdd(output + (i * stride), toScaledUint64(v));
    }
}

// Accumulates shift sums for RUN rows across COLS neighboring columns
template <int p, int RUN, int DC0, int DC1, int NUM, int COLS = 1>
__device__ __forceinline__ void sumColumnRun(const float* __restrict__ input, const int c, const int r0, const int width, const int height, float (&sums)[NUM], const int validCols = COLS) {
    using L = MeShiftLayout<p>;
    constexpr int HALO = 8; // >= maxShift, a multiple of 4 for aligned float4 loads
    constexpr int WINDOW = RUN + (2 * HALO);
    static_assert(HALO >= L::maxShift, "the halo must cover the largest shift");
    const bool alignedColumns = (height & 3) == 0;
    const bool interiorRun = alignedColumns && r0 >= L::pad && r0 + RUN <= height - L::pad;

    // Scale task values while zeroing pixels outside inner image rows
    float a[COLS][RUN];
#pragma unroll
    for (int j = 0; j < COLS; j++) {
        const float* colA = input + (static_cast<size_t>(clamp(c + j, 0, width - 1)) * height);
        const float scale = j < validCols ? L::productScale : 0.0f;
        if (interiorRun) {
#pragma unroll
            for (int v = 0; v < RUN; v += 4) {
                const float4 f = *reinterpret_cast<const float4*>(colA + r0 + v);
                a[j][v + 0] = f.x * scale;
                a[j][v + 1] = f.y * scale;
                a[j][v + 2] = f.z * scale;
                a[j][v + 3] = f.w * scale;
            }
        } else {
#pragma unroll
            for (int t = 0; t < RUN; t++) {
                const int r = r0 + t;
                a[j][t] = (r >= L::pad && r < height - L::pad) ? colA[r] * scale : 0.0f;
            }
        }
    }

    const bool fastWindow = alignedColumns && r0 - HALO >= 0 && r0 + RUN + HALO <= height;
#pragma unroll
    for (int k = DC0; k < DC1 + COLS - 1; k++) {
        const float* colB = input + (static_cast<size_t>(clamp(c + k, 0, width - 1)) * height);
        float w[WINDOW];
        if (fastWindow) {
#pragma unroll
            for (int v = 0; v < WINDOW; v += 4) {
                const float4 f = *reinterpret_cast<const float4*>(colB + r0 - HALO + v);
                w[v + 0] = f.x;
                w[v + 1] = f.y;
                w[v + 2] = f.z;
                w[v + 3] = f.w;
            }
        } else {
#pragma unroll
            for (int v = 0; v < WINDOW; v++)
                w[v] = colB[clamp(r0 - HALO + v, 0, height - 1)];
        }
#pragma unroll
        for (int j = 0; j < COLS; j++) {
            const int dc = k - j;
            if (dc < DC0 || dc >= DC1)
                continue; // Resolved at compile time
#pragma unroll
            for (int dr = (dc == 0 ? 0 : -L::maxShift); dr <= L::maxShift; dr++) {
                float sum = 0.0f;
#pragma unroll
                for (int t = 0; t < RUN; t++)
                    sum = fmaf(a[j][t], w[t + HALO + dr], sum);
                sums[L::shiftIndex(dr, dc) - L::firstShift(DC0)] += sum;
            }
        }
    }
}

// Accumulates shift sums along a border row for columns in range
template <int p, int RUN, int DC0, int DC1, int NUM>
__device__ __forceinline__ void sumRowRun(const float* __restrict__ source, const int r, const int c0, const int width, const int height, float (&sums)[NUM]) {
    using L = MeShiftLayout<p>;
    constexpr int WINDOW = RUN + (DC1 - 1 - DC0);
    const size_t columnStride = L::copyBorderRows ? 1 : height;
    const auto row = [&](const int imageRow) {
        const int clamped = clamp(imageRow, 0, height - 1);
        return L::copyBorderRows ? source + (static_cast<size_t>(L::borderCopyRow(clamped, height)) * width) : source + clamped;
    };
    const float* rowA = row(r);
    float a[RUN];
#pragma unroll
    for (int t = 0; t < RUN; t++) {
        const int c = c0 + t;
        a[t] = (c >= L::pad && c < width - L::pad) ? rowA[c * columnStride] * L::productScale : 0.0f;
    }
#pragma unroll
    for (int dr = -L::maxShift; dr <= L::maxShift; dr++) {
        const float* rowB = row(r + dr);
        float w[WINDOW];
#pragma unroll
        for (int v = 0; v < WINDOW; v++)
            w[v] = rowB[clamp(c0 + DC0 + v, 0, width - 1) * columnStride];
#pragma unroll
        for (int dc = DC0; dc < DC1; dc++) {
            if (dc == 0 && dr < 0)
                continue; // Skip opposite shift already covered
            float sum = 0.0f;
#pragma unroll
            for (int t = 0; t < RUN; t++)
                sum = fmaf(a[t], w[t + dc - DC0], sum);
            sums[L::shiftIndex(dr, dc) - L::firstShift(DC0)] += sum;
        }
    }
}

// Copies top and bottom border rows into a contiguous row-major buffer
template <int p>
__global__ void me_copy_border_rows(const float* __restrict__ input, float* __restrict__ borderCopy, const int width, const int height) {
    using L = MeShiftLayout<p>;
    const int total = L::borderCopyRows * width;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += gridDim.x * blockDim.x) {
        const int row = i % L::borderCopyRows;
        const int col = i / L::borderCopyRows;
        const int imageRow = row < L::borderCopyRows / 2 ? row : height - L::borderCopyRows + row;
        borderCopy[(static_cast<size_t>(row) * width) + col] = input[(static_cast<size_t>(col) * height) + imageRow];
    }
}

template <int V>
using IntConstant = std::integral_constant<int, V>;

// Dispatches the column shift range for the block's assigned group at compile time
template <int p, int G = 0, typename Work>
__device__ __forceinline__ void forColumnShiftGroup(const int group, Work&& work) {
    using L = MeShiftLayout<p>;
    if constexpr (G + 1 == L::shiftGroups)
        work(IntConstant<L::groupFirstDc(G)>{}, IntConstant<L::groupFirstDc(G + 1)>{});
    else if (group == G)
        work(IntConstant<L::groupFirstDc(G)>{}, IntConstant<L::groupFirstDc(G + 1)>{});
    else
        forColumnShiftGroup<p, G + 1>(group, work);
}

// Single-launch kernel computing all inner and border shift sums
template <int p>
__global__ void __launch_bounds__(256, 2) me_shift_sums(const float* __restrict__ input, const float* __restrict__ borderRowsCopy, uint64_t* __restrict__ shiftSums, const int width, const int height,
    const int interiorBlocks, const int borderBlocksPerLine) {
    using L = MeShiftLayout<p>;
    constexpr int RUN = L::interiorRun;
    constexpr int G = L::shiftGroups;
    __shared__ float warpSums[8 * L::numShifts];

    const bool isBorder = blockIdx.x >= interiorBlocks;
    const int localBlock = isBorder ? blockIdx.x - interiorBlocks : blockIdx.x;
    const int group = localBlock % G;
    const int groupBlock = localBlock / G;
    forColumnShiftGroup<p>(group, [&](auto dc0, auto dc1) {
        constexpr int DC0 = decltype(dc0)::value;
        constexpr int DC1 = decltype(dc1)::value;
        constexpr int SHIFT0 = L::firstShift(DC0);
        constexpr int NUM = L::shiftEnd(DC1) - SHIFT0;
        float sums[NUM];
#pragma unroll
        for (int i = 0; i < NUM; i++)
            sums[i] = 0.0f;
        if (!isBorder) {
            const int runsPerColumn = (height + RUN - 1) / RUN;
            const int interiorColumns = width - (2 * L::pad);
            constexpr int COLS = L::interiorCols;
            const int totalTasks = runsPerColumn * ((interiorColumns + COLS - 1) / COLS);
            const int stride = (interiorBlocks / G) * blockDim.x;
            for (int task = groupBlock * blockDim.x + threadIdx.x; task < totalTasks; task += stride) {
                const int c = COLS * (task / runsPerColumn);
                sumColumnRun<p, RUN, DC0, DC1, NUM, COLS>(input, L::pad + c, (task % runsPerColumn) * RUN, width, height, sums, min(COLS, interiorColumns - c));
            }
            reduceShiftSums<NUM>(sums, shiftSums + SHIFT0, 1, warpSums);
        } else {
            const int line = groupBlock / borderBlocksPerLine;
            const int lineBlock = groupBlock % borderBlocksPerLine;
            const bool isBorderRow = line < L::borderSize;
            const int border = isBorderRow ? line : line - L::borderSize;
            const int fixedCoord = L::borderToCoord(border, isBorderRow ? height : width);
            const int runs = ((isBorderRow ? width : height) + RUN - 1) / RUN;
            for (int run = lineBlock * blockDim.x + threadIdx.x; run < runs; run += borderBlocksPerLine * blockDim.x) {
                if (isBorderRow)
                    sumRowRun<p, RUN, DC0, DC1>(L::copyBorderRows ? borderRowsCopy : input, fixedCoord, run * RUN, width, height, sums);
                else
                    sumColumnRun<p, RUN, DC0, DC1>(input, fixedCoord, run * RUN, width, height, sums);
            }
            uint64_t* output = shiftSums + (isBorderRow ? L::borderRowsOffset : L::borderColsOffset) + (SHIFT0 * L::borderSize) + border;
            reduceShiftSums<NUM>(sums, output, L::borderSize, warpSums);
        }
    });
}

// Builds one entry of the solver system from inner shift sums, borders, and corners
template <int p>
__device__ uint64_t buildSystemEntry(const float* __restrict__ input, const uint64_t* __restrict__ shiftSums, const int entry, const int width, const int height) {
    using L = MeShiftLayout<p>;
    constexpr int N = L::windowSize - 1;
    constexpr int RxSize = (N * (N + 1)) / 2;
    // Maps solver variable to window pixel index
    auto toWindow = [](const int k) {
        const int e = coefficientIndex<p>(k);
        return e + (e >= L::windowCenter);
    };
    int wa, wb;
    if (entry < RxSize) {
        const int2 coords = packedToRowCol(entry);
        wa = toWindow(coords.x);
        wb = toWindow(coords.y);
    } else {
        wa = toWindow(entry - RxSize);
        wb = L::windowCenter;
    }
    // Map window coordinates to shift offset, swapping direction if needed
    int baseRow = (wa % p) - L::pad, baseCol = (wa / p) - L::pad;
    int dr = (wb % p) - (wa % p), dc = (wb / p) - (wa / p);
    if (dc < 0 || (dc == 0 && dr < 0)) {
        baseRow += dr;
        baseCol += dc;
        dr = -dr;
        dc = -dc;
    }
    const int shift = L::shiftIndex(dr, dc);
    // Accumulate border sums across the contiguous window range
    const int rowLo = baseRow + L::pad;
    const int colLo = baseCol + L::pad;
    uint64_t fixedSum = shiftSums[shift];
#pragma unroll
    for (int i = 0; i < 2 * L::pad; i++)
        fixedSum += shiftSums[L::borderRowsOffset + (shift * L::borderSize) + rowLo + i] + shiftSums[L::borderColsOffset + (shift * L::borderSize) + colLo + i];
    float corners = 0.0f;
#pragma unroll
    for (int i = 0; i < 2 * L::pad; i++) {
        const int r = L::borderToCoord(rowLo + i, height);
#pragma unroll
        for (int j = 0; j < 2 * L::pad; j++) {
            const int c = L::borderToCoord(colLo + j, width);
            corners += clampedPixel(input, r, c, width, height) * clampedPixel(input, r + dr, c + dc, width, height);
        }
    }
    return fixedSum + toScaledUint64(corners * L::productScale);
}

// Blocked Cholesky solver in float-float arithmetic using panel updates
// Panels factor 8 columns at a time with warp shuffles, followed by a trailing submatrix update
template <int N>
struct CholeskyLayout {
    static constexpr int NB = 8;
    static_assert(N % NB == 0, "N = p^2 - 1 is a multiple of 8 for odd p");
    static constexpr int packedSize = (N * (N + 1)) / 2;
    // Rows below the diagonal block assigned per warp
    static constexpr int panelRowsPerWarp = 32 - NB;
    // Dense panel buffer for trailing submatrix update
    static constexpr int panelStride = (2 * NB) + 4;
    static constexpr int panelSize = (N - NB) * panelStride;
    // Packed index for lower triangular element (r >= c)
    __device__ static __forceinline__ int idx(const int r, const int c) { return ((r * (r + 1)) / 2) + c; }
};

// Factors an 8-column panel and updates the right-hand side using warp shuffles
template <int N>
__device__ __forceinline__ void choleskyPanel(FloatFloat* __restrict__ sA, FloatFloat* __restrict__ sB, FloatFloat* __restrict__ sInv, float* __restrict__ sPanel, const int k0, int& sAbort) {
    using C = CholeskyLayout<N>;
    constexpr int NB = C::NB;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const bool diagonal = lane < NB;
    const int row = diagonal ? k0 + lane : k0 + NB + (warp * C::panelRowsPerWarp) + (lane - NB);
    const bool valid = row < N;

    FloatFloat a[NB];
    FloatFloat b = valid ? sB[row] : FloatFloat(0.0f);
#pragma unroll
    for (int j = 0; j < NB; j++)
        a[j] = valid && (!diagonal || j <= lane) ? sA[C::idx(row, k0 + j)] : FloatFloat(0.0f);

    bool positiveDefinite = true;
    FloatFloat pivot = shfl(a[0], 0);
#pragma unroll
    for (int k = 0; k < NB; k++) {
        positiveDefinite &= pivot.hi > 1e-3f; // Check positive definiteness threshold
        const FloatFloat inv = rsqrt(pivot);
        const FloatFloat l = a[k] * inv; // Scale panel entries by inverse pivot square root
        // Compute next pivot along the critical path
        if (k + 1 < NB)
            pivot = shfl(fnma(l, l, a[k + 1]), k + 1);
        const FloatFloat y = shfl(b, k) * inv;
        // Update panel rows using warp shuffles without branching
#pragma unroll
        for (int j = k + 1; j < NB; j++)
            a[j] = fnma(l, shfl(l, j), a[j]);
        a[k] = l;
        const FloatFloat updatedB = fnma(l, y, b);
        b = lane == k ? y : (!diagonal || lane > k ? updatedB : b);
        if (warp == 0 && lane == k)
            sInv[k0 + k] = inv;
    }

    if (valid && (warp == 0 || !diagonal)) {
#pragma unroll
        for (int j = 0; j < NB; j++)
            if (!diagonal || j <= lane)
                sA[C::idx(row, k0 + j)] = a[j];
        sB[row] = b;
        if (!diagonal) {
            float* panelRow = sPanel + ((row - k0 - NB) * C::panelStride);
#pragma unroll
            for (int j = 0; j < NB; j++) {
                panelRow[j] = a[j].hi;
                panelRow[NB + j] = a[j].lo;
            }
        }
    }
    if (warp == 0 && lane == 0 && !positiveDefinite)
        sAbort = 1;
}

// Updates trailing submatrix A22 -= L21 * L21^T
template <int N, int BLOCK>
__device__ __forceinline__ void choleskyTrailingUpdate(FloatFloat* __restrict__ sA, const float* __restrict__ sPanel, const int k0) {
    using C = CholeskyLayout<N>;
    constexpr int NB = C::NB;
    const int first = k0 + NB;
    const int rows = N - first;
    const int trailing = (rows * (rows + 1)) / 2;
    for (int e = threadIdx.x; e < trailing; e += BLOCK) {
        const int2 coords = packedToRowCol(e);
        float pi[2 * NB], pj[2 * NB];
#pragma unroll
        for (int v = 0; v < (2 * NB) / 4; v++) {
            reinterpret_cast<float4*>(pi)[v] = reinterpret_cast<const float4*>(sPanel + (coords.x * C::panelStride))[v];
            reinterpret_cast<float4*>(pj)[v] = reinterpret_cast<const float4*>(sPanel + (coords.y * C::panelStride))[v];
        }
        const int index = C::idx(first + coords.x, first + coords.y);
        // Accumulate low parts first and normalize once at the end
        const FloatFloat acc = sA[index];
        float hi = acc.hi, lo = acc.lo;
#pragma unroll
        for (int m = 0; m < NB; m++) {
            const float ph = pi[m] * pj[m];
            const float pl = fmaf(pi[m], pj[NB + m], fmaf(pi[NB + m], pj[m], fmaf(pi[m], pj[m], -ph)));
            const FloatFloat sum = FloatFloat::twoSum(hi, -ph);
            hi = sum.hi;
            lo += sum.lo - pl;
        }
        sA[index] = FloatFloat::quickTwoSum(hi, lo);
    }
}

// Backward substitution L^T * x = y within a single warp using shuffles
template <int p>
__device__ __forceinline__ void choleskyBackSubstitution(const FloatFloat* __restrict__ sA, const FloatFloat* __restrict__ sB, const FloatFloat* __restrict__ sInv, float* __restrict__ X) {
    constexpr int N = (p * p) - 1;
    constexpr int SLOTS = (N + 31) / 32;
    using C = CholeskyLayout<N>;
    const int lane = threadIdx.x & 31;
    FloatFloat y[SLOTS];
#pragma unroll
    for (int s = 0; s < SLOTS; s++)
        y[s] = lane + (32 * s) < N ? sB[lane + (32 * s)] : FloatFloat(0.0f);
#pragma unroll
    for (int s = SLOTS - 1; s >= 0; s--) {
        for (int k = min(N, 32 * (s + 1)) - 1; k >= 32 * s; k--) {
            const FloatFloat x = shfl(y[s], k - (32 * s)) * sInv[k];
            if (lane == k - (32 * s))
                y[s] = x;
#pragma unroll
            for (int t = 0; t <= s; t++) {
                const int i = lane + (32 * t);
                const FloatFloat updated = fnma(sA[C::idx(k, min(i, k))], x, y[t]);
                y[t] = i < k ? updated : y[t];
            }
        }
    }
#pragma unroll
    for (int s = 0; s < SLOTS; s++)
        if (lane + (32 * s) < N)
            X[coefficientIndex<p>(lane + (32 * s))] = static_cast<float>(y[s]);
}

// Assembles the solver matrix entries in parallel across grid blocks
template <int p>
__global__ void me_build_system(const float* __restrict__ input, const uint64_t* __restrict__ shiftSums, uint64_t* __restrict__ system, const int width, const int height) {
    constexpr int N = (p * p) - 1;
    const int entry = blockIdx.x * blockDim.x + threadIdx.x;
    if (entry < ((N * (N + 1)) / 2) + N)
        system[entry] = buildSystemEntry<p>(input, shiftSums, entry, width, height);
}

// Solves Rx * x = rx using blocked Cholesky decomposition in shared memory
template <int p, int BLOCK>
__global__ void __launch_bounds__(BLOCK)
    me_solve_system(const uint64_t* __restrict__ system, uint64_t* __restrict__ shiftSums, uint64_t* __restrict__ embedSums, float* __restrict__ X, int* __restrict__ stopFlag) {
    constexpr int N = (p * p) - 1;
    using C = CholeskyLayout<N>;
    static_assert(BLOCK >= 32 * ((N - C::NB + C::panelRowsPerWarp - 1) / C::panelRowsPerWarp), "not enough warps for the first panel");
    // Packed Rx followed by rx vector
    constexpr int systemSize = C::packedSize + N;
    __shared__ FloatFloat sSystem[systemSize];
    FloatFloat* sA = sSystem;
    FloatFloat* sB = sSystem + C::packedSize;
    __shared__ FloatFloat sInv[N];
    __shared__ __align__(16) float sPanel[C::panelSize > 0 ? C::panelSize : 1];
    __shared__ int sAbort;
    // Load all system values into registers before converting
    constexpr int loadsPerThread = (systemSize + BLOCK - 1) / BLOCK;
    uint64_t values[loadsPerThread];
#pragma unroll
    for (int i = 0; i < loadsPerThread; i++)
        values[i] = (i * BLOCK) + threadIdx.x < systemSize ? system[(i * BLOCK) + threadIdx.x] : 0;
#pragma unroll
    for (int i = 0; i < loadsPerThread; i++)
        if ((i * BLOCK) + threadIdx.x < systemSize)
            sSystem[(i * BLOCK) + threadIdx.x] = FloatFloat::fromUint64(values[i]);
    for (int i = threadIdx.x; i < MeShiftLayout<p>::shiftSumsSize; i += BLOCK)
        shiftSums[i] = 0;
    if (threadIdx.x < 2)
        embedSums[threadIdx.x] = 0;
    if (threadIdx.x == 0)
        sAbort = 0;
    __syncthreads();

    for (int k0 = 0; k0 < N; k0 += C::NB) {
        // Only activate warps needed for the current panel rows
        if ((threadIdx.x >> 5) * C::panelRowsPerWarp < max(N - k0 - C::NB, 1))
            choleskyPanel<N>(sA, sB, sInv, sPanel, k0, sAbort);
        __syncthreads();
        if (sAbort) { // Exit early if matrix is not positive definite
            if (threadIdx.x == 0)
                *stopFlag = 1;
            return;
        }
        if (k0 + C::NB < N) {
            choleskyTrailingUpdate<N, BLOCK>(sA, sPanel, k0);
            __syncthreads();
        }
    }
    if (threadIdx.x < 32) {
        choleskyBackSubstitution<p>(sA, sB, sInv, X);
        if (threadIdx.x == 0)
            *stopFlag = 0;
    }
}

// Computes embedding strength and checks whether the image is flat
__device__ __forceinline__ float embedStrength(const uint64_t* __restrict__ sumSqPtr, const uint64_t* __restrict__ maxAbsBits, const float strengthNumerator) {
    const float uSumSquared = toUnscaledFloat(*sumSqPtr);
    float normalizedSumSquared = uSumSquared;
    if (maxAbsBits) {
        const float normFactor = 1.0f / ((__uint_as_float(static_cast<unsigned int>(*maxAbsBits)) + 1.0e-6f) * kMeMaskPrescale);
        normalizedSumSquared *= normFactor * normFactor;
    }
    return normalizedSumSquared > 1e-3f ? strengthNumerator * rsqrtf(uSumSquared) : 0.0f;
}

// Rounds and clamps a floating point value to an 8-bit pixel
__device__ __forceinline__ uint8_t toPixel(const float value) { return static_cast<uint8_t>(clamp(value + 0.5f, 0.0f, 255.0f)); }

// Loads 4 consecutive pixels as float4
__device__ __forceinline__ float4 loadPixels4(const uint8_t* __restrict__ input, const int vectorIndex) {
    const uchar4 bytes = reinterpret_cast<const uchar4*>(input)[vectorIndex];
    return make_float4(bytes.x, bytes.y, bytes.z, bytes.w);
}
__device__ __forceinline__ float4 loadPixels4(const float* __restrict__ input, const int vectorIndex) { return reinterpret_cast<const float4*>(input)[vectorIndex]; }

// Adds watermark to image channels: output = input + strength * u
template <typename T>
__global__ void apply_watermark_fused(const T* __restrict__ input, const __half* __restrict__ u, const uint64_t* __restrict__ sumSqPtr, const uint64_t* __restrict__ maxAbsBits,
    uint8_t* __restrict__ output, const float strengthNumerator, const int planeElements, const int numChannels) {
    const float strength = embedStrength(sumSqPtr, maxAbsBits, strengthNumerator);
    const int stride = blockDim.x * gridDim.x;
    const int first = blockIdx.x * blockDim.x + threadIdx.x;
    const int planeVectors = planeElements % 4 == 0 ? planeElements / 4 : 0;
    for (int v = first; v < planeVectors; v += stride) {
        const float4 uv = toFloat4(reinterpret_cast<const Half4*>(u)[v]);
        const float4 us = make_float4(uv.x * strength, uv.y * strength, uv.z * strength, uv.w * strength);
        for (int c = 0; c < numChannels; c++) {
            const int vectorIndex = v + c * planeVectors;
            const float4 in = loadPixels4(input, vectorIndex);
            reinterpret_cast<uchar4*>(output)[vectorIndex] = make_uchar4(toPixel(in.x + us.x), toPixel(in.y + us.y), toPixel(in.z + us.z), toPixel(in.w + us.w));
        }
    }
    for (int i = planeVectors * 4 + first; i < planeElements; i += stride) {
        const float us = __half2float(u[i]) * strength;
        for (int c = 0; c < numChannels; c++) {
            const int pixelIdx = i + c * planeElements;
            output[pixelIdx] = toPixel(static_cast<float>(input[pixelIdx]) + us);
        }
    }
}

// Grayscale embedding transposing column-major input to row-major output via shared memory
__global__ void apply_watermark_row_major(const float* __restrict__ input, const __half* __restrict__ u, const uint64_t* __restrict__ sumSqPtr, const uint64_t* __restrict__ maxAbsBits,
    uint8_t* __restrict__ output, const float strengthNumerator, const int width, const int height);

// Converts NV12 to YUV420p format for video decoding
__global__ void nV12ToYUV420p(const uint8_t* __restrict__ uvSrc, const int uvPitch, uint8_t* __restrict__ uvDst, const int uvWidth, const int uvHeight);

// Converts pitched uint8 memory to non-pitched float buffer
__global__ void pitchedToFloat(const uint8_t* __restrict__ input, float* __restrict__ output, const int width, const int height, const int pitch);

// Converts column-major uint8 input to float grayscale
__global__ void u8ToFloatGray(const uint8_t* __restrict__ input, float* __restrict__ output, const int planeSize, const int numChannels);

// Transposes column-major uint8 image back to row-major format
__global__ void colMajorToRowMajorU8(const uint8_t* __restrict__ src, uint8_t* __restrict__ dst, const int width, const int height);

// Converts row-major planar RGB to column-major RGB and float luma
__global__ void rowMajorRgbToColMajor(const uint8_t* __restrict__ src, uint8_t* __restrict__ rgbDst, float* __restrict__ grayDst, const int width, const int height);

// Precomputed lookup tables for PQ EOTF and BT.1886 gamma to avoid costly powf calls
static __device__ const float pqEotfLUT[1024] = {0.00000000e+00f, 4.04227176e-07f, 1.31113719e-06f, 2.62368259e-06f, 4.31514955e-06f, 6.37468853e-06f, 8.79823827e-06f, 1.15853619e-05f,
    1.47378191e-05f, 1.82588184e-05f, 2.21525860e-05f, 2.64240984e-05f, 3.10789073e-05f, 3.61230211e-05f, 4.15628212e-05f, 4.74050009e-05f, 5.36565210e-05f, 6.03245762e-05f, 6.74165685e-05f,
    7.49400881e-05f, 8.29028967e-05f, 9.13129157e-05f, 1.00178216e-04f, 1.09507010e-04f, 1.19307645e-04f, 1.29588598e-04f, 1.40358472e-04f, 1.51625993e-04f, 1.63400004e-04f, 1.75689469e-04f,
    1.88503463e-04f, 2.01851179e-04f, 2.15741921e-04f, 2.30185106e-04f, 2.45190261e-04f, 2.60767028e-04f, 2.76925156e-04f, 2.93674508e-04f, 3.11025056e-04f, 3.28986884e-04f, 3.47570188e-04f,
    3.66785276e-04f, 3.86642568e-04f, 4.07152596e-04f, 4.28326007e-04f, 4.50173561e-04f, 4.72706132e-04f, 4.95934711e-04f, 5.19870404e-04f, 5.44524435e-04f, 5.69908144e-04f, 5.96032991e-04f,
    6.22910555e-04f, 6.50552534e-04f, 6.78970751e-04f, 7.08177146e-04f, 7.38183787e-04f, 7.69002863e-04f, 8.00646690e-04f, 8.33127708e-04f, 8.66458488e-04f, 9.00651724e-04f, 9.35720245e-04f,
    9.71677006e-04f, 1.00853510e-03f, 1.04630774e-03f, 1.08500828e-03f, 1.12465022e-03f, 1.16524719e-03f, 1.20681293e-03f, 1.24936136e-03f, 1.29290652e-03f, 1.33746260e-03f, 1.38304390e-03f,
    1.42966491e-03f, 1.47734024e-03f, 1.52608464e-03f, 1.57591302e-03f, 1.62684044e-03f, 1.67888210e-03f, 1.73205336e-03f, 1.78636971e-03f, 1.84184684e-03f, 1.89850054e-03f, 1.95634680e-03f,
    2.01540174e-03f, 2.07568166e-03f, 2.13720301e-03f, 2.19998240e-03f, 2.26403661e-03f, 2.32938257e-03f, 2.39603741e-03f, 2.46401838e-03f, 2.53334295e-03f, 2.60402872e-03f, 2.67609348e-03f,
    2.74955520e-03f, 2.82443200e-03f, 2.90074222e-03f, 2.97850434e-03f, 3.05773702e-03f, 3.13845914e-03f, 3.22068972e-03f, 3.30444799e-03f, 3.38975335e-03f, 3.47662541e-03f, 3.56508395e-03f,
    3.65514894e-03f, 3.74684057e-03f, 3.84017918e-03f, 3.93518536e-03f, 4.03187985e-03f, 4.13028361e-03f, 4.23041782e-03f, 4.33230383e-03f, 4.43596321e-03f, 4.54141774e-03f, 4.64868941e-03f,
    4.75780041e-03f, 4.86877315e-03f, 4.98163025e-03f, 5.09639456e-03f, 5.21308912e-03f, 5.33173722e-03f, 5.45236236e-03f, 5.57498825e-03f, 5.69963885e-03f, 5.82633832e-03f, 5.95511108e-03f,
    6.08598176e-03f, 6.21897523e-03f, 6.35411660e-03f, 6.49143121e-03f, 6.63094464e-03f, 6.77268271e-03f, 6.91667151e-03f, 7.06293733e-03f, 7.21150675e-03f, 7.36240657e-03f, 7.51566386e-03f,
    7.67130594e-03f, 7.82936040e-03f, 7.98985505e-03f, 8.15281800e-03f, 8.31827761e-03f, 8.48626251e-03f, 8.65680159e-03f, 8.82992402e-03f, 9.00565922e-03f, 9.18403692e-03f, 9.36508710e-03f,
    9.54884004e-03f, 9.73532628e-03f, 9.92457666e-03f, 1.01166223e-02f, 1.03114946e-02f, 1.05092253e-02f, 1.07098464e-02f, 1.09133902e-02f, 1.11198892e-02f, 1.13293765e-02f, 1.15418851e-02f,
    1.17574485e-02f, 1.19761007e-02f, 1.21978758e-02f, 1.24228081e-02f, 1.26509325e-02f, 1.28822840e-02f, 1.31168981e-02f, 1.33548105e-02f, 1.35960573e-02f, 1.38406748e-02f, 1.40886999e-02f,
    1.43401695e-02f, 1.45951210e-02f, 1.48535924e-02f, 1.51156215e-02f, 1.53812470e-02f, 1.56505076e-02f, 1.59234424e-02f, 1.62000910e-02f, 1.64804933e-02f, 1.67646895e-02f, 1.70527202e-02f,
    1.73446264e-02f, 1.76404495e-02f, 1.79402312e-02f, 1.82440135e-02f, 1.85518391e-02f, 1.88637508e-02f, 1.91797919e-02f, 1.95000060e-02f, 1.98244373e-02f, 2.01531301e-02f, 2.04861294e-02f,
    2.08234804e-02f, 2.11652288e-02f, 2.15114208e-02f, 2.18621028e-02f, 2.22173218e-02f, 2.25771252e-02f, 2.29415608e-02f, 2.33106767e-02f, 2.36845218e-02f, 2.40631449e-02f, 2.44465958e-02f,
    2.48349245e-02f, 2.52281813e-02f, 2.56264171e-02f, 2.60296834e-02f, 2.64380319e-02f, 2.68515150e-02f, 2.72701854e-02f, 2.76940963e-02f, 2.81233015e-02f, 2.85578551e-02f, 2.89978119e-02f,
    2.94432270e-02f, 2.98941560e-02f, 3.03506553e-02f, 3.08127813e-02f, 3.12805914e-02f, 3.17541431e-02f, 3.22334948e-02f, 3.27187051e-02f, 3.32098334e-02f, 3.37069393e-02f, 3.42100832e-02f,
    3.47193261e-02f, 3.52347292e-02f, 3.57563545e-02f, 3.62842645e-02f, 3.68185224e-02f, 3.73591917e-02f, 3.79063366e-02f, 3.84600219e-02f, 3.90203129e-02f, 3.95872756e-02f, 4.01609764e-02f,
    4.07414825e-02f, 4.13288615e-02f, 4.19231818e-02f, 4.25245123e-02f, 4.31329224e-02f, 4.37484824e-02f, 4.43712629e-02f, 4.50013355e-02f, 4.56387720e-02f, 4.62836451e-02f, 4.69360282e-02f,
    4.75959952e-02f, 4.82636206e-02f, 4.89389799e-02f, 4.96221488e-02f, 5.03132041e-02f, 5.10122230e-02f, 5.17192834e-02f, 5.24344640e-02f, 5.31578442e-02f, 5.38895040e-02f, 5.46295242e-02f,
    5.53779862e-02f, 5.61349722e-02f, 5.69005652e-02f, 5.76748488e-02f, 5.84579073e-02f, 5.92498260e-02f, 6.00506906e-02f, 6.08605878e-02f, 6.16796050e-02f, 6.25078304e-02f, 6.33453529e-02f,
    6.41922623e-02f, 6.50486489e-02f, 6.59146043e-02f, 6.67902204e-02f, 6.76755901e-02f, 6.85708073e-02f, 6.94759664e-02f, 7.03911630e-02f, 7.13164931e-02f, 7.22520538e-02f, 7.31979432e-02f,
    7.41542599e-02f, 7.51211036e-02f, 7.60985748e-02f, 7.70867750e-02f, 7.80858064e-02f, 7.90957722e-02f, 8.01167765e-02f, 8.11489243e-02f, 8.21923215e-02f, 8.32470749e-02f, 8.43132924e-02f,
    8.53910827e-02f, 8.64805554e-02f, 8.75818211e-02f, 8.86949916e-02f, 8.98201792e-02f, 9.09574977e-02f, 9.21070615e-02f, 9.32689862e-02f, 9.44433884e-02f, 9.56303856e-02f, 9.68300965e-02f,
    9.80426406e-02f, 9.92681386e-02f, 1.00506712e-01f, 1.01758485e-01f, 1.03023579e-01f, 1.04302121e-01f, 1.05594236e-01f, 1.06900052e-01f, 1.08219696e-01f, 1.09553298e-01f, 1.10900989e-01f,
    1.12262900e-01f, 1.13639164e-01f, 1.15029915e-01f, 1.16435287e-01f, 1.17855418e-01f, 1.19290444e-01f, 1.20740505e-01f, 1.22205740e-01f, 1.23686289e-01f, 1.25182296e-01f, 1.26693905e-01f,
    1.28221258e-01f, 1.29764504e-01f, 1.31323789e-01f, 1.32899261e-01f, 1.34491070e-01f, 1.36099369e-01f, 1.37724308e-01f, 1.39366042e-01f, 1.41024727e-01f, 1.42700518e-01f, 1.44393573e-01f,
    1.46104052e-01f, 1.47832114e-01f, 1.49577924e-01f, 1.51341642e-01f, 1.53123435e-01f, 1.54923468e-01f, 1.56741910e-01f, 1.58578929e-01f, 1.60434696e-01f, 1.62309383e-01f, 1.64203164e-01f,
    1.66116213e-01f, 1.68048707e-01f, 1.70000825e-01f, 1.71972746e-01f, 1.73964651e-01f, 1.75976722e-01f, 1.78009146e-01f, 1.80062106e-01f, 1.82135791e-01f, 1.84230390e-01f, 1.86346094e-01f,
    1.88483096e-01f, 1.90641589e-01f, 1.92821769e-01f, 1.95023833e-01f, 1.97247982e-01f, 1.99494416e-01f, 2.01763337e-01f, 2.04054951e-01f, 2.06369463e-01f, 2.08707081e-01f, 2.11068015e-01f,
    2.13452476e-01f, 2.15860679e-01f, 2.18292838e-01f, 2.20749170e-01f, 2.23229895e-01f, 2.25735233e-01f, 2.28265408e-01f, 2.30820643e-01f, 2.33401166e-01f, 2.36007205e-01f, 2.38638991e-01f,
    2.41296757e-01f, 2.43980736e-01f, 2.46691166e-01f, 2.49428286e-01f, 2.52192336e-01f, 2.54983558e-01f, 2.57802199e-01f, 2.60648504e-01f, 2.63522723e-01f, 2.66425108e-01f, 2.69355912e-01f,
    2.72315390e-01f, 2.75303801e-01f, 2.78321404e-01f, 2.81368462e-01f, 2.84445240e-01f, 2.87552004e-01f, 2.90689024e-01f, 2.93856571e-01f, 2.97054920e-01f, 3.00284345e-01f, 3.03545127e-01f,
    3.06837546e-01f, 3.10161885e-01f, 3.13518430e-01f, 3.16907471e-01f, 3.20329297e-01f, 3.23784202e-01f, 3.27272482e-01f, 3.30794436e-01f, 3.34350364e-01f, 3.37940570e-01f, 3.41565361e-01f,
    3.45225045e-01f, 3.48919934e-01f, 3.52650342e-01f, 3.56416586e-01f, 3.60218986e-01f, 3.64057864e-01f, 3.67933546e-01f, 3.71846360e-01f, 3.75796636e-01f, 3.79784709e-01f, 3.83810915e-01f,
    3.87875593e-01f, 3.91979086e-01f, 3.96121740e-01f, 4.00303903e-01f, 4.04525926e-01f, 4.08788163e-01f, 4.13090973e-01f, 4.17434716e-01f, 4.21819756e-01f, 4.26246459e-01f, 4.30715196e-01f,
    4.35226340e-01f, 4.39780267e-01f, 4.44377357e-01f, 4.49017992e-01f, 4.53702561e-01f, 4.58431451e-01f, 4.63205056e-01f, 4.68023772e-01f, 4.72887999e-01f, 4.77798141e-01f, 4.82754605e-01f,
    4.87757799e-01f, 4.92808139e-01f, 4.97906042e-01f, 5.03051929e-01f, 5.08246224e-01f, 5.13489355e-01f, 5.18781756e-01f, 5.24123861e-01f, 5.29516110e-01f, 5.34958946e-01f, 5.40452817e-01f,
    5.45998174e-01f, 5.51595471e-01f, 5.57245168e-01f, 5.62947727e-01f, 5.68703615e-01f, 5.74513303e-01f, 5.80377267e-01f, 5.86295984e-01f, 5.92269938e-01f, 5.98299617e-01f, 6.04385513e-01f,
    6.10528120e-01f, 6.16727940e-01f, 6.22985477e-01f, 6.29301240e-01f, 6.35675742e-01f, 6.42109500e-01f, 6.48603037e-01f, 6.55156880e-01f, 6.61771560e-01f, 6.68447613e-01f, 6.75185579e-01f,
    6.81986004e-01f, 6.88849438e-01f, 6.95776435e-01f, 7.02767555e-01f, 7.09823362e-01f, 7.16944425e-01f, 7.24131320e-01f, 7.31384624e-01f, 7.38704922e-01f, 7.46092804e-01f, 7.53548864e-01f,
    7.61073702e-01f, 7.68667921e-01f, 7.76332133e-01f, 7.84066952e-01f, 7.91873000e-01f, 7.99750901e-01f, 8.07701289e-01f, 8.15724799e-01f, 8.23822074e-01f, 8.31993763e-01f, 8.40240518e-01f,
    8.48563000e-01f, 8.56961874e-01f, 8.65437810e-01f, 8.73991485e-01f, 8.82623582e-01f, 8.91334789e-01f, 9.00125801e-01f, 9.08997318e-01f, 9.17950048e-01f, 9.26984703e-01f, 9.36102002e-01f,
    9.45302670e-01f, 9.54587439e-01f, 9.63957047e-01f, 9.73412239e-01f, 9.82953764e-01f, 9.92582382e-01f, 1.00229886e+00f, 1.01210396e+00f, 1.02199846e+00f, 1.03198315e+00f, 1.04205882e+00f,
    1.05222627e+00f, 1.06248630e+00f, 1.07283972e+00f, 1.08328736e+00f, 1.09383004e+00f, 1.10446858e+00f, 1.11520384e+00f, 1.12603667e+00f, 1.13696790e+00f, 1.14799842e+00f, 1.15912909e+00f,
    1.17036078e+00f, 1.18169439e+00f, 1.19313080e+00f, 1.20467092e+00f, 1.21631566e+00f, 1.22806594e+00f, 1.23992267e+00f, 1.25188679e+00f, 1.26395925e+00f, 1.27614099e+00f, 1.28843297e+00f,
    1.30083615e+00f, 1.31335152e+00f, 1.32598006e+00f, 1.33872275e+00f, 1.35158060e+00f, 1.36455461e+00f, 1.37764581e+00f, 1.39085523e+00f, 1.40418389e+00f, 1.41763284e+00f, 1.43120314e+00f,
    1.44489586e+00f, 1.45871205e+00f, 1.47265282e+00f, 1.48671924e+00f, 1.50091242e+00f, 1.51523348e+00f, 1.52968352e+00f, 1.54426369e+00f, 1.55897513e+00f, 1.57381897e+00f, 1.58879639e+00f,
    1.60390856e+00f, 1.61915666e+00f, 1.63454187e+00f, 1.65006541e+00f, 1.66572848e+00f, 1.68153231e+00f, 1.69747813e+00f, 1.71356719e+00f, 1.72980074e+00f, 1.74618005e+00f, 1.76270640e+00f,
    1.77938108e+00f, 1.79620539e+00f, 1.81318065e+00f, 1.83030816e+00f, 1.84758928e+00f, 1.86502536e+00f, 1.88261773e+00f, 1.90036779e+00f, 1.91827692e+00f, 1.93634650e+00f, 1.95457796e+00f,
    1.97297270e+00f, 1.99153216e+00f, 2.01025780e+00f, 2.02915105e+00f, 2.04821341e+00f, 2.06744635e+00f, 2.08685137e+00f, 2.10642998e+00f, 2.12618371e+00f, 2.14611409e+00f, 2.16622268e+00f,
    2.18651104e+00f, 2.20698074e+00f, 2.22763339e+00f, 2.24847060e+00f, 2.26949397e+00f, 2.29070515e+00f, 2.31210579e+00f, 2.33369755e+00f, 2.35548212e+00f, 2.37746119e+00f, 2.39963646e+00f,
    2.42200967e+00f, 2.44458256e+00f, 2.46735687e+00f, 2.49033439e+00f, 2.51351690e+00f, 2.53690620e+00f, 2.56050411e+00f, 2.58431247e+00f, 2.60833313e+00f, 2.63256796e+00f, 2.65701884e+00f,
    2.68168768e+00f, 2.70657640e+00f, 2.73168692e+00f, 2.75702121e+00f, 2.78258124e+00f, 2.80836899e+00f, 2.83438647e+00f, 2.86063570e+00f, 2.88711873e+00f, 2.91383762e+00f, 2.94079445e+00f,
    2.96799130e+00f, 2.99543031e+00f, 3.02311360e+00f, 3.05104334e+00f, 3.07922168e+00f, 3.10765084e+00f, 3.13633301e+00f, 3.16527044e+00f, 3.19446537e+00f, 3.22392008e+00f, 3.25363686e+00f,
    3.28361803e+00f, 3.31386591e+00f, 3.34438288e+00f, 3.37517130e+00f, 3.40623357e+00f, 3.43757211e+00f, 3.46918937e+00f, 3.50108780e+00f, 3.53326990e+00f, 3.56573816e+00f, 3.59849513e+00f,
    3.63154336e+00f, 3.66488542e+00f, 3.69852391e+00f, 3.73246145e+00f, 3.76670070e+00f, 3.80124433e+00f, 3.83609502e+00f, 3.87125550e+00f, 3.90672851e+00f, 3.94251683e+00f, 3.97862324e+00f,
    4.01505056e+00f, 4.05180165e+00f, 4.08887936e+00f, 4.12628660e+00f, 4.16402628e+00f, 4.20210137e+00f, 4.24051483e+00f, 4.27926966e+00f, 4.31836890e+00f, 4.35781560e+00f, 4.39761286e+00f,
    4.43776377e+00f, 4.47827149e+00f, 4.51913918e+00f, 4.56037005e+00f, 4.60196732e+00f, 4.64393424e+00f, 4.68627412e+00f, 4.72899025e+00f, 4.77208600e+00f, 4.81556473e+00f, 4.85942985e+00f,
    4.90368482e+00f, 4.94833309e+00f, 4.99337817e+00f, 5.03882359e+00f, 5.08467292e+00f, 5.13092977e+00f, 5.17759775e+00f, 5.22468054e+00f, 5.27218184e+00f, 5.32010538e+00f, 5.36845493e+00f,
    5.41723428e+00f, 5.46644727e+00f, 5.51609779e+00f, 5.56618972e+00f, 5.61672702e+00f, 5.66771366e+00f, 5.71915367e+00f, 5.77105108e+00f, 5.82341000e+00f, 5.87623454e+00f, 5.92952888e+00f,
    5.98329721e+00f, 6.03754378e+00f, 6.09227287e+00f, 6.14748879e+00f, 6.20319592e+00f, 6.25939864e+00f, 6.31610141e+00f, 6.37330870e+00f, 6.43102503e+00f, 6.48925497e+00f, 6.54800312e+00f,
    6.60727415e+00f, 6.66707273e+00f, 6.72740360e+00f, 6.78827154e+00f, 6.84968139e+00f, 6.91163799e+00f, 6.97414628e+00f, 7.03721120e+00f, 7.10083776e+00f, 7.16503102e+00f, 7.22979606e+00f,
    7.29513804e+00f, 7.36106215e+00f, 7.42757363e+00f, 7.49467777e+00f, 7.56237991e+00f, 7.63068543e+00f, 7.69959978e+00f, 7.76912844e+00f, 7.83927695e+00f, 7.91005092e+00f, 7.98145597e+00f,
    8.05349781e+00f, 8.12618219e+00f, 8.19951490e+00f, 8.27350182e+00f, 8.34814884e+00f, 8.42346194e+00f, 8.49944713e+00f, 8.57611051e+00f, 8.65345819e+00f, 8.73149639e+00f, 8.81023134e+00f,
    8.88966936e+00f, 8.96981682e+00f, 9.05068014e+00f, 9.13226582e+00f, 9.21458040e+00f, 9.29763050e+00f, 9.38142279e+00f, 9.46596400e+00f, 9.55126093e+00f, 9.63732045e+00f, 9.72414948e+00f,
    9.81175502e+00f, 9.90014412e+00f, 9.98932391e+00f, 1.00793016e+01f, 1.01700844e+01f, 1.02616797e+01f, 1.03540948e+01f, 1.04473373e+01f, 1.05414147e+01f, 1.06363345e+01f, 1.07321045e+01f,
    1.08287324e+01f, 1.09262260e+01f, 1.10245933e+01f, 1.11238422e+01f, 1.12239808e+01f, 1.13250172e+01f, 1.14269595e+01f, 1.15298162e+01f, 1.16335954e+01f, 1.17383058e+01f, 1.18439558e+01f,
    1.19505539e+01f, 1.20581090e+01f, 1.21666297e+01f, 1.22761249e+01f, 1.23866035e+01f, 1.24980746e+01f, 1.26105472e+01f, 1.27240306e+01f, 1.28385339e+01f, 1.29540667e+01f, 1.30706384e+01f,
    1.31882584e+01f, 1.33069364e+01f, 1.34266822e+01f, 1.35475056e+01f, 1.36694165e+01f, 1.37924249e+01f, 1.39165409e+01f, 1.40417748e+01f, 1.41681367e+01f, 1.42956372e+01f, 1.44242868e+01f,
    1.45540959e+01f, 1.46850754e+01f, 1.48172361e+01f, 1.49505887e+01f, 1.50851445e+01f, 1.52209145e+01f, 1.53579098e+01f, 1.54961419e+01f, 1.56356223e+01f, 1.57763623e+01f, 1.59183739e+01f,
    1.60616686e+01f, 1.62062584e+01f, 1.63521553e+01f, 1.64993715e+01f, 1.66479191e+01f, 1.67978106e+01f, 1.69490585e+01f, 1.71016752e+01f, 1.72556736e+01f, 1.74110665e+01f, 1.75678668e+01f,
    1.77260878e+01f, 1.78857425e+01f, 1.80468444e+01f, 1.82094069e+01f, 1.83734436e+01f, 1.85389684e+01f, 1.87059950e+01f, 1.88745375e+01f, 1.90446101e+01f, 1.92162270e+01f, 1.93894027e+01f,
    1.95641517e+01f, 1.97404888e+01f, 1.99184288e+01f, 2.00979867e+01f, 2.02791777e+01f, 2.04620170e+01f, 2.06465202e+01f, 2.08327028e+01f, 2.10205805e+01f, 2.12101694e+01f, 2.14014853e+01f,
    2.15945447e+01f, 2.17893638e+01f, 2.19859591e+01f, 2.21843475e+01f, 2.23845457e+01f, 2.25865708e+01f, 2.27904401e+01f, 2.29961707e+01f, 2.32037805e+01f, 2.34132869e+01f, 2.36247080e+01f,
    2.38380617e+01f, 2.40533664e+01f, 2.42706405e+01f, 2.44899026e+01f, 2.47111715e+01f, 2.49344661e+01f, 2.51598056e+01f, 2.53872094e+01f, 2.56166971e+01f, 2.58482884e+01f, 2.60820032e+01f,
    2.63178616e+01f, 2.65558841e+01f, 2.67960912e+01f, 2.70385035e+01f, 2.72831421e+01f, 2.75300282e+01f, 2.77791830e+01f, 2.80306282e+01f, 2.82843856e+01f, 2.85404772e+01f, 2.87989253e+01f,
    2.90597523e+01f, 2.93229810e+01f, 2.95886341e+01f, 2.98567349e+01f, 3.01273068e+01f, 3.04003734e+01f, 3.06759584e+01f, 3.09540861e+01f, 3.12347807e+01f, 3.15180669e+01f, 3.18039693e+01f,
    3.20925132e+01f, 3.23837239e+01f, 3.26776268e+01f, 3.29742480e+01f, 3.32736134e+01f, 3.35757494e+01f, 3.38806827e+01f, 3.41884402e+01f, 3.44990490e+01f, 3.48125366e+01f, 3.51289307e+01f,
    3.54482593e+01f, 3.57705507e+01f, 3.60958336e+01f, 3.64241366e+01f, 3.67554891e+01f, 3.70899205e+01f, 3.74274605e+01f, 3.77681392e+01f, 3.81119870e+01f, 3.84590345e+01f, 3.88093127e+01f,
    3.91628530e+01f, 3.95196869e+01f, 3.98798465e+01f, 4.02433639e+01f, 4.06102718e+01f, 4.09806031e+01f, 4.13543911e+01f, 4.17316695e+01f, 4.21124721e+01f, 4.24968333e+01f, 4.28847878e+01f,
    4.32763706e+01f, 4.36716170e+01f, 4.40705627e+01f, 4.44732440e+01f, 4.48796973e+01f, 4.52899594e+01f, 4.57040677e+01f, 4.61220596e+01f, 4.65439732e+01f, 4.69698470e+01f, 4.73997196e+01f,
    4.78336304e+01f, 4.82716190e+01f, 4.87137253e+01f, 4.91599897e+01f, 4.96104533e+01f, 5.00651571e+01f, 5.05241429e+01f, 5.09874530e+01f, 5.14551297e+01f, 5.19272162e+01f, 5.24037560e+01f,
    5.28847928e+01f, 5.33703712e+01f, 5.38605360e+01f, 5.43553325e+01f, 5.48548064e+01f, 5.53590040e+01f, 5.58679721e+01f, 5.63817579e+01f, 5.69004092e+01f, 5.74239741e+01f, 5.79525014e+01f,
    5.84860404e+01f, 5.90246408e+01f, 5.95683528e+01f, 6.01172274e+01f, 6.06713159e+01f, 6.12306701e+01f, 6.17953425e+01f, 6.23653861e+01f, 6.29408545e+01f, 6.35218017e+01f, 6.41082824e+01f,
    6.47003520e+01f, 6.52980662e+01f, 6.59014816e+01f, 6.65106550e+01f, 6.71256443e+01f, 6.77465076e+01f, 6.83733038e+01f, 6.90060925e+01f, 6.96449337e+01f, 7.02898882e+01f, 7.09410175e+01f,
    7.15983836e+01f, 7.22620492e+01f, 7.29320778e+01f, 7.36085334e+01f, 7.42914808e+01f, 7.49809855e+01f, 7.56771136e+01f, 7.63799320e+01f, 7.70895083e+01f, 7.78059107e+01f, 7.85292084e+01f,
    7.92594710e+01f, 7.99967692e+01f, 8.07411742e+01f, 8.14927580e+01f, 8.22515935e+01f, 8.30177544e+01f, 8.37913150e+01f, 8.45723505e+01f, 8.53609369e+01f, 8.61571512e+01f, 8.69610709e+01f,
    8.77727747e+01f, 8.85923417e+01f, 8.94198524e+01f, 9.02553878e+01f, 9.10990298e+01f, 9.19508613e+01f, 9.28109661e+01f, 9.36794288e+01f, 9.45563352e+01f, 9.54417716e+01f, 9.63358255e+01f,
    9.72385855e+01f, 9.81501408e+01f, 9.90705818e+01f, 1.00000000e+02f};

static __device__ const float bt1886LUT[1024] = {0.00000000e+00f, 5.57038423e-02f, 7.43557087e-02f, 8.80411437e-02f, 9.92529634e-02f, 1.08923767e-01f, 1.17520827e-01f, 1.25316811e-01f,
    1.32486811e-01f, 1.39150957e-01f, 1.45395786e-01f, 1.51286011e-01f, 1.56871484e-01f, 1.62191547e-01f, 1.67277874e-01f, 1.72156402e-01f, 1.76848676e-01f, 1.81372819e-01f, 1.85744243e-01f,
    1.89976181e-01f, 1.94080090e-01f, 1.98065967e-01f, 2.01942597e-01f, 2.05717743e-01f, 2.09398309e-01f, 2.12990462e-01f, 2.16499741e-01f, 2.19931138e-01f, 2.23289173e-01f, 2.26577954e-01f,
    2.29801226e-01f, 2.32962414e-01f, 2.36064661e-01f, 2.39110856e-01f, 2.42103667e-01f, 2.45045560e-01f, 2.47938819e-01f, 2.50785566e-01f, 2.53587778e-01f, 2.56347295e-01f, 2.59065839e-01f,
    2.61745019e-01f, 2.64386347e-01f, 2.66991238e-01f, 2.69561026e-01f, 2.72096966e-01f, 2.74600242e-01f, 2.77071971e-01f, 2.79513208e-01f, 2.81924955e-01f, 2.84308158e-01f, 2.86663716e-01f,
    2.88992483e-01f, 2.91295269e-01f, 2.93572848e-01f, 2.95825954e-01f, 2.98055287e-01f, 3.00261518e-01f, 3.02445284e-01f, 3.04607195e-01f, 3.06747836e-01f, 3.08867764e-01f, 3.10967515e-01f,
    3.13047602e-01f, 3.15108517e-01f, 3.17150732e-01f, 3.19174700e-01f, 3.21180858e-01f, 3.23169624e-01f, 3.25141402e-01f, 3.27096579e-01f, 3.29035531e-01f, 3.30958617e-01f, 3.32866184e-01f,
    3.34758569e-01f, 3.36636094e-01f, 3.38499072e-01f, 3.40347806e-01f, 3.42182586e-01f, 3.44003695e-01f, 3.45811406e-01f, 3.47605984e-01f, 3.49387683e-01f, 3.51156753e-01f, 3.52913432e-01f,
    3.54657955e-01f, 3.56390545e-01f, 3.58111423e-01f, 3.59820801e-01f, 3.61518885e-01f, 3.63205875e-01f, 3.64881966e-01f, 3.66547347e-01f, 3.68202202e-01f, 3.69846709e-01f, 3.71481042e-01f,
    3.73105370e-01f, 3.74719858e-01f, 3.76324665e-01f, 3.77919949e-01f, 3.79505860e-01f, 3.81082546e-01f, 3.82650153e-01f, 3.84208819e-01f, 3.85758683e-01f, 3.87299878e-01f, 3.88832535e-01f,
    3.90356780e-01f, 3.91872737e-01f, 3.93380529e-01f, 3.94880273e-01f, 3.96372084e-01f, 3.97856076e-01f, 3.99332359e-01f, 4.00801041e-01f, 4.02262226e-01f, 4.03716018e-01f, 4.05162518e-01f,
    4.06601824e-01f, 4.08034032e-01f, 4.09459236e-01f, 4.10877529e-01f, 4.12289001e-01f, 4.13693740e-01f, 4.15091832e-01f, 4.16483363e-01f, 4.17868416e-01f, 4.19247070e-01f, 4.20619407e-01f,
    4.21985504e-01f, 4.23345437e-01f, 4.24699281e-01f, 4.26047110e-01f, 4.27388996e-01f, 4.28725009e-01f, 4.30055219e-01f, 4.31379694e-01f, 4.32698499e-01f, 4.34011701e-01f, 4.35319364e-01f,
    4.36621550e-01f, 4.37918322e-01f, 4.39209740e-01f, 4.40495864e-01f, 4.41776752e-01f, 4.43052461e-01f, 4.44323049e-01f, 4.45588570e-01f, 4.46849079e-01f, 4.48104630e-01f, 4.49355275e-01f,
    4.50601065e-01f, 4.51842052e-01f, 4.53078286e-01f, 4.54309815e-01f, 4.55536688e-01f, 4.56758953e-01f, 4.57976655e-01f, 4.59189842e-01f, 4.60398558e-01f, 4.61602847e-01f, 4.62802754e-01f,
    4.63998321e-01f, 4.65189590e-01f, 4.66376604e-01f, 4.67559404e-01f, 4.68738029e-01f, 4.69912519e-01f, 4.71082915e-01f, 4.72249253e-01f, 4.73411572e-01f, 4.74569910e-01f, 4.75724303e-01f,
    4.76874788e-01f, 4.78021400e-01f, 4.79164174e-01f, 4.80303146e-01f, 4.81438348e-01f, 4.82569816e-01f, 4.83697581e-01f, 4.84821677e-01f, 4.85942137e-01f, 4.87058990e-01f, 4.88172270e-01f,
    4.89282007e-01f, 4.90388232e-01f, 4.91490973e-01f, 4.92590262e-01f, 4.93686127e-01f, 4.94778597e-01f, 4.95867700e-01f, 4.96953464e-01f, 4.98035918e-01f, 4.99115087e-01f, 5.00191000e-01f,
    5.01263683e-01f, 5.02333161e-01f, 5.03399462e-01f, 5.04462609e-01f, 5.05522629e-01f, 5.06579546e-01f, 5.07633385e-01f, 5.08684170e-01f, 5.09731925e-01f, 5.10776674e-01f, 5.11818439e-01f,
    5.12857244e-01f, 5.13893112e-01f, 5.14926065e-01f, 5.15956124e-01f, 5.16983313e-01f, 5.18007653e-01f, 5.19029164e-01f, 5.20047869e-01f, 5.21063787e-01f, 5.22076940e-01f, 5.23087348e-01f,
    5.24095031e-01f, 5.25100008e-01f, 5.26102300e-01f, 5.27101926e-01f, 5.28098905e-01f, 5.29093255e-01f, 5.30084997e-01f, 5.31074147e-01f, 5.32060725e-01f, 5.33044748e-01f, 5.34026235e-01f,
    5.35005203e-01f, 5.35981669e-01f, 5.36955651e-01f, 5.37927166e-01f, 5.38896231e-01f, 5.39862862e-01f, 5.40827077e-01f, 5.41788890e-01f, 5.42748319e-01f, 5.43705380e-01f, 5.44660087e-01f,
    5.45612458e-01f, 5.46562507e-01f, 5.47510250e-01f, 5.48455701e-01f, 5.49398876e-01f, 5.50339790e-01f, 5.51278457e-01f, 5.52214891e-01f, 5.53149108e-01f, 5.54081121e-01f, 5.55010944e-01f,
    5.55938592e-01f, 5.56864078e-01f, 5.57787415e-01f, 5.58708617e-01f, 5.59627698e-01f, 5.60544671e-01f, 5.61459548e-01f, 5.62372343e-01f, 5.63283068e-01f, 5.64191737e-01f, 5.65098361e-01f,
    5.66002953e-01f, 5.66905526e-01f, 5.67806092e-01f, 5.68704663e-01f, 5.69601250e-01f, 5.70495865e-01f, 5.71388521e-01f, 5.72279229e-01f, 5.73168000e-01f, 5.74054846e-01f, 5.74939778e-01f,
    5.75822807e-01f, 5.76703945e-01f, 5.77583202e-01f, 5.78460588e-01f, 5.79336116e-01f, 5.80209795e-01f, 5.81081636e-01f, 5.81951650e-01f, 5.82819847e-01f, 5.83686236e-01f, 5.84550829e-01f,
    5.85413636e-01f, 5.86274666e-01f, 5.87133929e-01f, 5.87991435e-01f, 5.88847194e-01f, 5.89701215e-01f, 5.90553508e-01f, 5.91404083e-01f, 5.92252948e-01f, 5.93100114e-01f, 5.93945589e-01f,
    5.94789382e-01f, 5.95631503e-01f, 5.96471960e-01f, 5.97310762e-01f, 5.98147919e-01f, 5.98983438e-01f, 5.99817329e-01f, 6.00649600e-01f, 6.01480260e-01f, 6.02309317e-01f, 6.03136779e-01f,
    6.03962655e-01f, 6.04786953e-01f, 6.05609681e-01f, 6.06430847e-01f, 6.07250460e-01f, 6.08068527e-01f, 6.08885055e-01f, 6.09700054e-01f, 6.10513530e-01f, 6.11325492e-01f, 6.12135947e-01f,
    6.12944902e-01f, 6.13752365e-01f, 6.14558344e-01f, 6.15362845e-01f, 6.16165877e-01f, 6.16967446e-01f, 6.17767560e-01f, 6.18566226e-01f, 6.19363451e-01f, 6.20159241e-01f, 6.20953605e-01f,
    6.21746548e-01f, 6.22538078e-01f, 6.23328202e-01f, 6.24116926e-01f, 6.24904257e-01f, 6.25690202e-01f, 6.26474767e-01f, 6.27257959e-01f, 6.28039784e-01f, 6.28820249e-01f, 6.29599360e-01f,
    6.30377124e-01f, 6.31153546e-01f, 6.31928634e-01f, 6.32702393e-01f, 6.33474830e-01f, 6.34245950e-01f, 6.35015760e-01f, 6.35784265e-01f, 6.36551473e-01f, 6.37317387e-01f, 6.38082016e-01f,
    6.38845363e-01f, 6.39607436e-01f, 6.40368240e-01f, 6.41127781e-01f, 6.41886064e-01f, 6.42643094e-01f, 6.43398879e-01f, 6.44153422e-01f, 6.44906730e-01f, 6.45658809e-01f, 6.46409662e-01f,
    6.47159297e-01f, 6.47907718e-01f, 6.48654931e-01f, 6.49400940e-01f, 6.50145752e-01f, 6.50889371e-01f, 6.51631802e-01f, 6.52373052e-01f, 6.53113123e-01f, 6.53852023e-01f, 6.54589756e-01f,
    6.55326326e-01f, 6.56061739e-01f, 6.56796000e-01f, 6.57529113e-01f, 6.58261084e-01f, 6.58991917e-01f, 6.59721618e-01f, 6.60450190e-01f, 6.61177638e-01f, 6.61903968e-01f, 6.62629183e-01f,
    6.63353290e-01f, 6.64076291e-01f, 6.64798192e-01f, 6.65518997e-01f, 6.66238710e-01f, 6.66957337e-01f, 6.67674882e-01f, 6.68391348e-01f, 6.69106741e-01f, 6.69821065e-01f, 6.70534324e-01f,
    6.71246522e-01f, 6.71957664e-01f, 6.72667754e-01f, 6.73376796e-01f, 6.74084794e-01f, 6.74791753e-01f, 6.75497676e-01f, 6.76202568e-01f, 6.76906433e-01f, 6.77609274e-01f, 6.78311097e-01f,
    6.79011904e-01f, 6.79711700e-01f, 6.80410489e-01f, 6.81108275e-01f, 6.81805061e-01f, 6.82500852e-01f, 6.83195651e-01f, 6.83889462e-01f, 6.84582289e-01f, 6.85274136e-01f, 6.85965007e-01f,
    6.86654904e-01f, 6.87343833e-01f, 6.88031796e-01f, 6.88718798e-01f, 6.89404841e-01f, 6.90089930e-01f, 6.90774068e-01f, 6.91457259e-01f, 6.92139507e-01f, 6.92820814e-01f, 6.93501184e-01f,
    6.94180621e-01f, 6.94859128e-01f, 6.95536709e-01f, 6.96213368e-01f, 6.96889106e-01f, 6.97563929e-01f, 6.98237839e-01f, 6.98910840e-01f, 6.99582934e-01f, 7.00254126e-01f, 7.00924418e-01f,
    7.01593814e-01f, 7.02262317e-01f, 7.02929931e-01f, 7.03596658e-01f, 7.04262501e-01f, 7.04927465e-01f, 7.05591551e-01f, 7.06254764e-01f, 7.06917105e-01f, 7.07578580e-01f, 7.08239189e-01f,
    7.08898937e-01f, 7.09557826e-01f, 7.10215860e-01f, 7.10873042e-01f, 7.11529374e-01f, 7.12184860e-01f, 7.12839502e-01f, 7.13493303e-01f, 7.14146267e-01f, 7.14798396e-01f, 7.15449693e-01f,
    7.16100161e-01f, 7.16749803e-01f, 7.17398622e-01f, 7.18046620e-01f, 7.18693801e-01f, 7.19340167e-01f, 7.19985720e-01f, 7.20630465e-01f, 7.21274402e-01f, 7.21917536e-01f, 7.22559869e-01f,
    7.23201403e-01f, 7.23842142e-01f, 7.24482087e-01f, 7.25121242e-01f, 7.25759610e-01f, 7.26397192e-01f, 7.27033992e-01f, 7.27670011e-01f, 7.28305254e-01f, 7.28939722e-01f, 7.29573417e-01f,
    7.30206343e-01f, 7.30838502e-01f, 7.31469896e-01f, 7.32100528e-01f, 7.32730400e-01f, 7.33359516e-01f, 7.33987876e-01f, 7.34615485e-01f, 7.35242343e-01f, 7.35868455e-01f, 7.36493821e-01f,
    7.37118445e-01f, 7.37742329e-01f, 7.38365475e-01f, 7.38987885e-01f, 7.39609563e-01f, 7.40230510e-01f, 7.40850728e-01f, 7.41470221e-01f, 7.42088989e-01f, 7.42707036e-01f, 7.43324364e-01f,
    7.43940975e-01f, 7.44556872e-01f, 7.45172056e-01f, 7.45786529e-01f, 7.46400295e-01f, 7.47013355e-01f, 7.47625711e-01f, 7.48237366e-01f, 7.48848322e-01f, 7.49458581e-01f, 7.50068145e-01f,
    7.50677016e-01f, 7.51285196e-01f, 7.51892688e-01f, 7.52499494e-01f, 7.53105615e-01f, 7.53711055e-01f, 7.54315814e-01f, 7.54919895e-01f, 7.55523300e-01f, 7.56126031e-01f, 7.56728090e-01f,
    7.57329480e-01f, 7.57930201e-01f, 7.58530257e-01f, 7.59129649e-01f, 7.59728379e-01f, 7.60326449e-01f, 7.60923861e-01f, 7.61520618e-01f, 7.62116720e-01f, 7.62712170e-01f, 7.63306971e-01f,
    7.63901123e-01f, 7.64494629e-01f, 7.65087490e-01f, 7.65679709e-01f, 7.66271287e-01f, 7.66862227e-01f, 7.67452530e-01f, 7.68042198e-01f, 7.68631232e-01f, 7.69219636e-01f, 7.69807410e-01f,
    7.70394556e-01f, 7.70981077e-01f, 7.71566973e-01f, 7.72152248e-01f, 7.72736901e-01f, 7.73320937e-01f, 7.73904355e-01f, 7.74487158e-01f, 7.75069348e-01f, 7.75650927e-01f, 7.76231895e-01f,
    7.76812256e-01f, 7.77392010e-01f, 7.77971159e-01f, 7.78549706e-01f, 7.79127651e-01f, 7.79704996e-01f, 7.80281744e-01f, 7.80857895e-01f, 7.81433452e-01f, 7.82008416e-01f, 7.82582789e-01f,
    7.83156572e-01f, 7.83729768e-01f, 7.84302377e-01f, 7.84874401e-01f, 7.85445842e-01f, 7.86016702e-01f, 7.86586982e-01f, 7.87156684e-01f, 7.87725809e-01f, 7.88294359e-01f, 7.88862335e-01f,
    7.89429740e-01f, 7.89996574e-01f, 7.90562839e-01f, 7.91128537e-01f, 7.91693669e-01f, 7.92258238e-01f, 7.92822243e-01f, 7.93385687e-01f, 7.93948572e-01f, 7.94510898e-01f, 7.95072668e-01f,
    7.95633883e-01f, 7.96194544e-01f, 7.96754653e-01f, 7.97314211e-01f, 7.97873220e-01f, 7.98431681e-01f, 7.98989595e-01f, 7.99546965e-01f, 8.00103792e-01f, 8.00660076e-01f, 8.01215820e-01f,
    8.01771025e-01f, 8.02325692e-01f, 8.02879823e-01f, 8.03433418e-01f, 8.03986481e-01f, 8.04539011e-01f, 8.05091010e-01f, 8.05642480e-01f, 8.06193422e-01f, 8.06743838e-01f, 8.07293728e-01f,
    8.07843095e-01f, 8.08391938e-01f, 8.08940261e-01f, 8.09488064e-01f, 8.10035348e-01f, 8.10582115e-01f, 8.11128367e-01f, 8.11674103e-01f, 8.12219327e-01f, 8.12764038e-01f, 8.13308239e-01f,
    8.13851931e-01f, 8.14395115e-01f, 8.14937792e-01f, 8.15479963e-01f, 8.16021630e-01f, 8.16562795e-01f, 8.17103458e-01f, 8.17643620e-01f, 8.18183283e-01f, 8.18722449e-01f, 8.19261117e-01f,
    8.19799291e-01f, 8.20336970e-01f, 8.20874156e-01f, 8.21410851e-01f, 8.21947055e-01f, 8.22482769e-01f, 8.23017996e-01f, 8.23552736e-01f, 8.24086990e-01f, 8.24620760e-01f, 8.25154046e-01f,
    8.25686851e-01f, 8.26219174e-01f, 8.26751018e-01f, 8.27282383e-01f, 8.27813271e-01f, 8.28343683e-01f, 8.28873619e-01f, 8.29403082e-01f, 8.29932072e-01f, 8.30460590e-01f, 8.30988638e-01f,
    8.31516216e-01f, 8.32043326e-01f, 8.32569970e-01f, 8.33096147e-01f, 8.33621859e-01f, 8.34147108e-01f, 8.34671894e-01f, 8.35196218e-01f, 8.35720082e-01f, 8.36243487e-01f, 8.36766433e-01f,
    8.37288922e-01f, 8.37810955e-01f, 8.38332534e-01f, 8.38853658e-01f, 8.39374329e-01f, 8.39894549e-01f, 8.40414318e-01f, 8.40933637e-01f, 8.41452508e-01f, 8.41970931e-01f, 8.42488908e-01f,
    8.43006439e-01f, 8.43523526e-01f, 8.44040169e-01f, 8.44556370e-01f, 8.45072130e-01f, 8.45587449e-01f, 8.46102329e-01f, 8.46616771e-01f, 8.47130776e-01f, 8.47644344e-01f, 8.48157477e-01f,
    8.48670176e-01f, 8.49182441e-01f, 8.49694275e-01f, 8.50205677e-01f, 8.50716648e-01f, 8.51227191e-01f, 8.51737305e-01f, 8.52246992e-01f, 8.52756252e-01f, 8.53265087e-01f, 8.53773497e-01f,
    8.54281484e-01f, 8.54789048e-01f, 8.55296191e-01f, 8.55802913e-01f, 8.56309216e-01f, 8.56815100e-01f, 8.57320565e-01f, 8.57825614e-01f, 8.58330247e-01f, 8.58834465e-01f, 8.59338269e-01f,
    8.59841660e-01f, 8.60344639e-01f, 8.60847206e-01f, 8.61349363e-01f, 8.61851110e-01f, 8.62352449e-01f, 8.62853379e-01f, 8.63353904e-01f, 8.63854022e-01f, 8.64353735e-01f, 8.64853044e-01f,
    8.65351950e-01f, 8.65850453e-01f, 8.66348555e-01f, 8.66846256e-01f, 8.67343558e-01f, 8.67840461e-01f, 8.68336965e-01f, 8.68833073e-01f, 8.69328784e-01f, 8.69824100e-01f, 8.70319021e-01f,
    8.70813549e-01f, 8.71307684e-01f, 8.71801426e-01f, 8.72294778e-01f, 8.72787739e-01f, 8.73280311e-01f, 8.73772494e-01f, 8.74264289e-01f, 8.74755697e-01f, 8.75246720e-01f, 8.75737356e-01f,
    8.76227608e-01f, 8.76717477e-01f, 8.77206962e-01f, 8.77696066e-01f, 8.78184788e-01f, 8.78673130e-01f, 8.79161092e-01f, 8.79648675e-01f, 8.80135880e-01f, 8.80622708e-01f, 8.81109159e-01f,
    8.81595235e-01f, 8.82080936e-01f, 8.82566262e-01f, 8.83051215e-01f, 8.83535796e-01f, 8.84020005e-01f, 8.84503843e-01f, 8.84987310e-01f, 8.85470408e-01f, 8.85953138e-01f, 8.86435499e-01f,
    8.86917493e-01f, 8.87399121e-01f, 8.87880383e-01f, 8.88361281e-01f, 8.88841814e-01f, 8.89321983e-01f, 8.89801790e-01f, 8.90281235e-01f, 8.90760319e-01f, 8.91239042e-01f, 8.91717406e-01f,
    8.92195410e-01f, 8.92673056e-01f, 8.93150345e-01f, 8.93627277e-01f, 8.94103853e-01f, 8.94580073e-01f, 8.95055939e-01f, 8.95531451e-01f, 8.96006610e-01f, 8.96481416e-01f, 8.96955870e-01f,
    8.97429974e-01f, 8.97903727e-01f, 8.98377130e-01f, 8.98850184e-01f, 8.99322890e-01f, 8.99795248e-01f, 9.00267260e-01f, 9.00738925e-01f, 9.01210244e-01f, 9.01681219e-01f, 9.02151850e-01f,
    9.02622137e-01f, 9.03092082e-01f, 9.03561684e-01f, 9.04030945e-01f, 9.04499865e-01f, 9.04968445e-01f, 9.05436685e-01f, 9.05904587e-01f, 9.06372151e-01f, 9.06839377e-01f, 9.07306267e-01f,
    9.07772820e-01f, 9.08239038e-01f, 9.08704921e-01f, 9.09170470e-01f, 9.09635685e-01f, 9.10100568e-01f, 9.10565118e-01f, 9.11029337e-01f, 9.11493225e-01f, 9.11956783e-01f, 9.12420011e-01f,
    9.12882910e-01f, 9.13345480e-01f, 9.13807723e-01f, 9.14269639e-01f, 9.14731228e-01f, 9.15192491e-01f, 9.15653429e-01f, 9.16114043e-01f, 9.16574332e-01f, 9.17034298e-01f, 9.17493942e-01f,
    9.17953263e-01f, 9.18412262e-01f, 9.18870941e-01f, 9.19329300e-01f, 9.19787338e-01f, 9.20245058e-01f, 9.20702459e-01f, 9.21159542e-01f, 9.21616308e-01f, 9.22072757e-01f, 9.22528890e-01f,
    9.22984707e-01f, 9.23440210e-01f, 9.23895398e-01f, 9.24350273e-01f, 9.24804834e-01f, 9.25259082e-01f, 9.25713019e-01f, 9.26166644e-01f, 9.26619959e-01f, 9.27072963e-01f, 9.27525658e-01f,
    9.27978043e-01f, 9.28430120e-01f, 9.28881889e-01f, 9.29333350e-01f, 9.29784505e-01f, 9.30235353e-01f, 9.30685896e-01f, 9.31136133e-01f, 9.31586066e-01f, 9.32035695e-01f, 9.32485021e-01f,
    9.32934043e-01f, 9.33382763e-01f, 9.33831182e-01f, 9.34279299e-01f, 9.34727115e-01f, 9.35174632e-01f, 9.35621848e-01f, 9.36068766e-01f, 9.36515385e-01f, 9.36961706e-01f, 9.37407729e-01f,
    9.37853456e-01f, 9.38298886e-01f, 9.38744021e-01f, 9.39188860e-01f, 9.39633404e-01f, 9.40077655e-01f, 9.40521611e-01f, 9.40965274e-01f, 9.41408645e-01f, 9.41851723e-01f, 9.42294510e-01f,
    9.42737006e-01f, 9.43179211e-01f, 9.43621126e-01f, 9.44062751e-01f, 9.44504088e-01f, 9.44945136e-01f, 9.45385896e-01f, 9.45826368e-01f, 9.46266554e-01f, 9.46706453e-01f, 9.47146066e-01f,
    9.47585393e-01f, 9.48024436e-01f, 9.48463194e-01f, 9.48901668e-01f, 9.49339858e-01f, 9.49777766e-01f, 9.50215391e-01f, 9.50652734e-01f, 9.51089796e-01f, 9.51526577e-01f, 9.51963077e-01f,
    9.52399297e-01f, 9.52835237e-01f, 9.53270899e-01f, 9.53706282e-01f, 9.54141387e-01f, 9.54576214e-01f, 9.55010764e-01f, 9.55445038e-01f, 9.55879035e-01f, 9.56312757e-01f, 9.56746203e-01f,
    9.57179375e-01f, 9.57612272e-01f, 9.58044896e-01f, 9.58477246e-01f, 9.58909323e-01f, 9.59341128e-01f, 9.59772662e-01f, 9.60203923e-01f, 9.60634914e-01f, 9.61065634e-01f, 9.61496084e-01f,
    9.61926264e-01f, 9.62356175e-01f, 9.62785818e-01f, 9.63215192e-01f, 9.63644299e-01f, 9.64073138e-01f, 9.64501710e-01f, 9.64930016e-01f, 9.65358056e-01f, 9.65785830e-01f, 9.66213339e-01f,
    9.66640583e-01f, 9.67067564e-01f, 9.67494280e-01f, 9.67920733e-01f, 9.68346924e-01f, 9.68772852e-01f, 9.69198517e-01f, 9.69623922e-01f, 9.70049065e-01f, 9.70473947e-01f, 9.70898570e-01f,
    9.71322932e-01f, 9.71747035e-01f, 9.72170879e-01f, 9.72594464e-01f, 9.73017792e-01f, 9.73440861e-01f, 9.73863674e-01f, 9.74286229e-01f, 9.74708528e-01f, 9.75130571e-01f, 9.75552359e-01f,
    9.75973891e-01f, 9.76395169e-01f, 9.76816192e-01f, 9.77236962e-01f, 9.77657478e-01f, 9.78077741e-01f, 9.78497751e-01f, 9.78917509e-01f, 9.79337015e-01f, 9.79756270e-01f, 9.80175273e-01f,
    9.80594026e-01f, 9.81012529e-01f, 9.81430782e-01f, 9.81848786e-01f, 9.82266541e-01f, 9.82684047e-01f, 9.83101305e-01f, 9.83518315e-01f, 9.83935078e-01f, 9.84351593e-01f, 9.84767863e-01f,
    9.85183886e-01f, 9.85599663e-01f, 9.86015194e-01f, 9.86430481e-01f, 9.86845523e-01f, 9.87260321e-01f, 9.87674875e-01f, 9.88089186e-01f, 9.88503253e-01f, 9.88917078e-01f, 9.89330661e-01f,
    9.89744001e-01f, 9.90157100e-01f, 9.90569958e-01f, 9.90982575e-01f, 9.91394952e-01f, 9.91807089e-01f, 9.92218986e-01f, 9.92630644e-01f, 9.93042063e-01f, 9.93453243e-01f, 9.93864186e-01f,
    9.94274891e-01f, 9.94685358e-01f, 9.95095588e-01f, 9.95505582e-01f, 9.95915339e-01f, 9.96324861e-01f, 9.96734147e-01f, 9.97143198e-01f, 9.97552014e-01f, 9.97960595e-01f, 9.98368943e-01f,
    9.98777057e-01f, 9.99184937e-01f, 9.99592585e-01f, 1.00000000e+00f};

// LUT lookup with linear interpolation between adjacent values
__device__ __forceinline__ float lutLerp1024(const float* __restrict__ lut, const float x01) {
    const float idx = __saturatef(x01) * 1023.0f;
    const int i0 = __float2int_rd(idx);
    const int i1 = min(i0 + 1, 1023);
    const float frac = idx - static_cast<float>(i0);
    const float v0 = __ldg(&lut[i0]);
    const float v1 = __ldg(&lut[i1]);
    return fmaf(v1 - v0, frac, v0);
}

// Mobius tonemapping: exact FFmpeg vf_tonemap.c formula: K * (x+a)/(x+b)
// a, b, K depend ONLY on the per-video hdrPeak so they are precomputed once on the host
__device__ __forceinline__ float mobiusTonemap(const float x, const float mobA, const float mobB, const float mobK) {
    constexpr float j = 0.3f; // transition point in npl=100 units (30 nits)
    if (x <= j)
        return x;
    return mobK * (x + mobA) * __frcp_rn(x + mobB);
}

// full HDR -> SDR pipeline for ONE pixel: P010LE YCbCr (BT.2020 PQ) -> display-referred BT.709 R'G'B' [0,1].
// this matches FFmpeg vf_tonemap: linear BT.2020 -> desat -> hue-preserving Mobius -> gamut -> gamma, caller extracts Y or Cb/Cr
// NOTE: I had to use fmaf() explicitly, compiler did not do it automatically, I wonder why..
__device__ __forceinline__ float3 hdrPixelToSdrRgb(const uint16_t yRaw, const uint16_t cbRaw, const uint16_t crRaw, const float mobA, const float mobB, const float mobK) {
    // P010LE limited range -> normalized [0,1] / [-0.5,0.5]
    const float Y2020 = __saturatef(fmaf(static_cast<float>(yRaw >> 6), 1.0f / 876.0f, -64.0f / 876.0f));
    const float Cb2020 = fmaf(static_cast<float>(cbRaw >> 6), 1.0f / 896.0f, -512.0f / 896.0f);
    const float Cr2020 = fmaf(static_cast<float>(crRaw >> 6), 1.0f / 896.0f, -512.0f / 896.0f);
    // BT.2020 YCbCr -> PQ-encoded R'G'B' (Kr=0.2627, Kb=0.0593), explicit FMAs! (lutLerp1024 clamps [0, 1] via __saturatef internally)
    const float Rp = fmaf(1.4746f, Cr2020, Y2020);
    const float Gp = fmaf(-0.16455f, Cb2020, fmaf(-0.57135f, Cr2020, Y2020));
    const float Bp = fmaf(1.8814f, Cb2020, Y2020);
    // PQ EOTF to linear RGB with peak luminance scaled inside the lookup table
    float R = lutLerp1024(pqEotfLUT, Rp);
    float G = lutLerp1024(pqEotfLUT, Gp);
    float B = lutLerp1024(pqEotfLUT, Bp);
    // Highlight desaturation
    constexpr float desat = 2.0f;
    const float luma2020 = fmaf(0.2627f, R, fmaf(0.6780f, G, 0.0593f * B));
    const float overbright = fmaxf(luma2020 - desat, 1e-6f) * __frcp_rn(fmaxf(luma2020, 1e-6f)); // Fast reciprocal
    R = fmaf(overbright, luma2020 - R, R);
    G = fmaf(overbright, luma2020 - G, G);
    B = fmaf(overbright, luma2020 - B, B);
    // Hue-preserving Mobius tone mapping applied uniformly across channels
    const float sig = fmaxf(fmaxf(R, G), B);
    if (sig > 1e-6f) {
        const float scale = __saturatef(mobiusTonemap(sig, mobA, mobB, mobK)) * __frcp_rn(sig);
        R *= scale;
        G *= scale;
        B *= scale;
    } else {
        R = G = B = 0.0f;
    }
    // Convert BT.2020 color gamut to BT.709
    const float R7 = fmaf(1.6605f, R, fmaf(-0.5876f, G, -0.0728f * B));
    const float G7 = fmaf(-0.1246f, R, fmaf(1.1329f, G, -0.0083f * B));
    const float B7 = fmaf(-0.0182f, R, fmaf(-0.1006f, G, 1.1187f * B));
    // Apply BT.1886 display gamma curve
    return make_float3(lutLerp1024(bt1886LUT, R7), lutLerp1024(bt1886LUT, G7), lutLerp1024(bt1886LUT, B7));
}

// Convert BT.709 RGB to limited range Y [16, 235]
__device__ __forceinline__ float rgbToYLimited(const float3 rgb) {
    const float Y = fmaf(0.2126f, rgb.x, fmaf(0.7152f, rgb.y, 0.0722f * rgb.z));
    return clamp(fmaf(Y, 219.0f, 16.0f), 16.0f, 235.0f);
}

// Converts HDR P010LE luma to column-major limited-range float with tone mapping
__global__ void p010HdrYToSdrFloat(const uint16_t* __restrict__ ySrc, const int yPitchBytes, const uint16_t* __restrict__ uvSrc, const int uvPitchBytes, float* __restrict__ output, const int width,
    const int height, const float mobA, const float mobB, const float mobK);

// Converts HDR P010LE chroma to SDR NV12 format with tone mapping
__global__ void p010HdrUVToSdrNV12(const uint16_t* __restrict__ ySrc, const int yPitchBytes, const uint16_t* __restrict__ uvSrc, const int uvPitchBytes, uint8_t* __restrict__ uvDst, const int width,
    const int height, const float mobA, const float mobB, const float mobK);

// Converts HDR P010LE luma to SDR uint8 without watermarking
__global__ void p010HdrYToSdrU8(const uint16_t* __restrict__ ySrc, const int yPitchBytes, const uint16_t* __restrict__ uvSrc, const int uvPitchBytes, uint8_t* __restrict__ output, const int width,
    const int height, const float mobA, const float mobB, const float mobK);
