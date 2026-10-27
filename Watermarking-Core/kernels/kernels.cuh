#pragma once
#include "common_utils.hpp"
#include <cstdint>
#include <cub/block/block_reduce.cuh>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "utility_kernels.cuh"
#include <device_launch_parameters.h>
#include <type_traits>

/*!
 *  \brief  CUDA device helper functions, atomic fixed-point conversions, and block reduction types
 *  \author Dimitris Karatzas
 */

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

// Fills shared memory with the image tile around block outputs with clamped edges, MASKED: |input| * w (the ME mask times the watermark)
// Loads all values into registers first before writing to shared memory
template <bool MASKED, int p, int shDimFast, int shDimSlow, int THREADS>
__device__ __forceinline__ void fillBlock(
    const float* __restrict__ input, const __half* __restrict__ w, float* __restrict__ sharedMem, const int tileSlow0, const int tileFast0, const int width, const int height) {
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
            // Evaluated at compile time to avoid branches
            if constexpr (MASKED)
                values[it] = fabsf(input[idx]) * __half2float(w[idx]);
            else
                values[it] = input[idx];
        }
    }
#pragma unroll
    for (int it = 0; it < iterations; it++) {
        const int i = tid + (it * THREADS);
        if (i < totalElements)
            sharedMem[i] = values[it];
    }
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

// Prepares shared memory tile (the input, or |input| * w when MASKED) and coefficients, returning thread start coordinates
template <int p, bool MASKED>
__device__ __forceinline__ int2 setupErrorTile(const float* __restrict__ input, const __half* __restrict__ w, const float* __restrict__ coeffs, float* __restrict__ region,
    float (&coef)[ErrorTile<p>::coeffsSize], const int width, const int height) {
    using T = ErrorTile<p>;
    const int tileSlow0 = blockIdx.y * T::tileSlow;
    const int tileFast0 = blockIdx.x * T::tileFast;
    fillBlock<MASKED, p, T::shFast, T::shSlow, T::threads>(input, w, region, tileSlow0, tileFast0, width, height);
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

// Correlation of the image prediction error e_u with the prediction error of |e_u| * w (the ME mask times the watermark, not divided by
// max|e_u|: the correlation does not depend on the scale). Computes partial sums and writes the final result from the last finished block
template <int p>
__global__ void __launch_bounds__(ErrorTile<p>::threads) calculate_error_sequence_and_partial_corr_fused(const float* __restrict__ e_u, const __half* __restrict__ w, const float* __restrict__ coeffs,
    float* __restrict__ partialDots, float* __restrict__ partialNormU, float* __restrict__ partialNormZ, unsigned int* __restrict__ blockCounter, float* __restrict__ correlation, const int width,
    const int height, const int* __restrict__ stopFlag) {
    using T = ErrorTile<p>;
    using BlockReduceT = cub::BlockReduce<CorrelationData, T::threads, cub::BLOCK_REDUCE_WARP_REDUCTIONS>;
    __shared__ typename BlockReduceT::TempStorage temp_storage;
    __shared__ alignas(16) float region[T::shSlow * T::shFast];
    float coef[T::coeffsSize];
    const int2 origin = setupErrorTile<p, true>(e_u, w, coeffs, region, coef, width, height);

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
        p.lo = fmaf(a.hi, b.lo, fmaf(a.lo, b.hi, p.lo));
        return quickTwoSum(p.hi, p.lo);
    }
    // Computes c - a * b with fast error compensation for Cholesky updates
    __device__ __forceinline__ friend FloatFloat fnma(const FloatFloat a, const FloatFloat b, const FloatFloat c) {
        const float ph = __fmul_rn(a.hi, b.hi); // fmul_rn on purpose
        const float pl = fmaf(a.hi, b.lo, fmaf(a.lo, b.hi, fmaf(a.hi, b.hi, -ph)));
        const FloatFloat s = twoSum(c.hi, -ph);
        return quickTwoSum(s.hi, s.lo + (c.lo - pl));
    }
    // Fast reciprocal square root refined with one float-float Newton step
    __device__ __forceinline__ friend FloatFloat rsqrt(const FloatFloat v) {
        const float h = rsqrtf(v.hi);
        const FloatFloat vh2 = v * twoProd(h, h);
        const float r = (1.0f - vh2.hi) - vh2.lo;
        return quickTwoSum(h, __fmul_rn(0.5f * h, r)); // fmul_rn on purpose
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

// Loads N consecutive floats with aligned float4 loads, OFF is the misalignment of "span" in floats (the 16-byte chunks also cover
// up to 3 floats before and after the span)
template <int OFF, int N>
__device__ __forceinline__ void loadSpanChunks(const float* __restrict__ span, float (&out)[N]) {
    const float4* chunks = reinterpret_cast<const float4*>(span - OFF);
#pragma unroll
    for (int q = 0; q < (OFF + N + 3) / 4; q++) {
        const float4 f = chunks[q];
        const float v[4] = {f.x, f.y, f.z, f.w};
#pragma unroll
        for (int i = 0; i < 4; i++)
            if ((4 * q) + i - OFF >= 0 && (4 * q) + i - OFF < N)
                out[(4 * q) + i - OFF] = v[i];
    }
}
// Same, for a column span at any float alignment (column heights that are not a multiple of 4)
template <int N>
__device__ __forceinline__ void loadSpan(const float* __restrict__ span, float (&out)[N]) {
    switch ((reinterpret_cast<uintptr_t>(span) / sizeof(float)) & 3) {
    case 0: loadSpanChunks<0>(span, out); break;
    case 1: loadSpanChunks<1>(span, out); break;
    case 2: loadSpanChunks<2>(span, out); break;
    default: loadSpanChunks<3>(span, out); break;
    }
}

// Accumulates shift sums for RUN rows across COLS neighboring columns, ALIGNED: the column height is a multiple of 4 (16-byte aligned columns)
template <int p, bool ALIGNED, int RUN, int DC0, int DC1, int NUM, int COLS = 1>
__device__ __forceinline__ void sumColumnRun(const float* __restrict__ input, const int c, const int r0, const int width, const int height, float (&sums)[NUM], const int validCols = COLS) {
    using L = MeShiftLayout<p>;
    constexpr int HALO = 8; // >= maxShift, a multiple of 4 for aligned float4 loads
    constexpr int WINDOW = RUN + (2 * HALO);
    static_assert(HALO >= L::maxShift, "the halo must cover the largest shift");
    // misaligned columns are read in whole 16-byte chunks, the vector loads need 3 more rows of the same column after the span (the last column ends the buffer)
    constexpr int chunkSlack = ALIGNED ? 0 : 3;
    const bool interiorRun = r0 >= L::pad && r0 + RUN <= height - L::pad && r0 + RUN + chunkSlack <= height;

    // Scale task values while zeroing pixels outside inner image rows
    float a[COLS][RUN];
#pragma unroll
    for (int j = 0; j < COLS; j++) {
        const float* colA = input + (static_cast<size_t>(clamp(c + j, 0, width - 1)) * height);
        const float scale = j < validCols ? L::productScale : 0.0f;
        if (interiorRun) {
            float run[RUN];
            if constexpr (ALIGNED)
                loadSpanChunks<0>(colA + r0, run);
            else
                loadSpan(colA + r0, run);
#pragma unroll
            for (int t = 0; t < RUN; t++)
                a[j][t] = run[t] * scale;
        } else {
#pragma unroll
            for (int t = 0; t < RUN; t++) {
                const int r = r0 + t;
                a[j][t] = (r >= L::pad && r < height - L::pad) ? colA[r] * scale : 0.0f;
            }
        }
    }

    const bool fastWindow = r0 - HALO >= 0 && r0 + RUN + HALO + chunkSlack <= height;
#pragma unroll
    for (int k = DC0; k < DC1 + COLS - 1; k++) {
        const float* colB = input + (static_cast<size_t>(clamp(c + k, 0, width - 1)) * height);
        float w[WINDOW];
        if (fastWindow) {
            if constexpr (ALIGNED)
                loadSpanChunks<0>(colB + r0 - HALO, w);
            else
                loadSpan(colB + r0 - HALO, w);
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

// Copies top and bottom border rows into a contiguous row-major buffer: slots [0, 3 * pad) hold the top image rows and slots
// [3 * pad, 6 * pad) the bottom ones (borderCopyRow maps an image row to its slot). For small images (height < 6 * pad) the two parts
// clamp source row inside image bounds
template <int p>
__global__ void me_copy_border_rows(const float* __restrict__ input, float* __restrict__ borderCopy, const int width, const int height) {
    using L = MeShiftLayout<p>;
    const int total = L::borderCopyRows * width;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += gridDim.x * blockDim.x) {
        const int row = i % L::borderCopyRows;
        const int col = i / L::borderCopyRows;
        const int imageRow = clamp(row < L::borderCopyRows / 2 ? row : height - L::borderCopyRows + row, 0, height - 1);
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

// Single-launch kernel computing all inner and border shift sums. Separate kernels for aligned and misaligned columns (see sumColumnRun): the
// misaligned loads would cost the aligned kernel registers
template <int p, bool ALIGNED>
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
                sumColumnRun<p, ALIGNED, RUN, DC0, DC1, NUM, COLS>(input, L::pad + c, (task % runsPerColumn) * RUN, width, height, sums, min(COLS, interiorColumns - c));
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
                    sumColumnRun<p, ALIGNED, RUN, DC0, DC1>(input, fixedCoord, run * RUN, width, height, sums);
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
            const float ph = __fmul_rn(pi[m], pj[m]); // fmul_rn on purpose
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

// Computes embedding strength and checks whether the image is flat: sumSq[0] = sum(u^2) (fixed point), sumSq[1] = max|e| (float bits)
__device__ __forceinline__ float embedStrength(const uint64_t* __restrict__ sumSq, const float strengthNumerator) {
    const float uSumSquared = toUnscaledFloat(sumSq[0]);
    const float normFactor = 1.0f / ((__uint_as_float(static_cast<unsigned int>(sumSq[1])) + 1.0e-6f) * kMeMaskPrescale);
    const float normalizedSumSquared = uSumSquared * (normFactor * normFactor);
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
__global__ void apply_watermark_fused(const T* __restrict__ input, const __half* __restrict__ u, const uint64_t* __restrict__ sumSq, uint8_t* __restrict__ output, const float strengthNumerator,
    const int planeElements, const int numChannels) {
    const float strength = embedStrength(sumSq, strengthNumerator);
    const int stride = blockDim.x * gridDim.x;
    const int first = blockIdx.x * blockDim.x + threadIdx.x;
    const int planeVectors = planeElements / 4;
    if (planeElements % 4 == 0) {
        for (int v = first; v < planeVectors; v += stride) {
            const float4 uv = toFloat4(reinterpret_cast<const Half4*>(u)[v]);
            const float4 us = make_float4(uv.x * strength, uv.y * strength, uv.z * strength, uv.w * strength);
            for (int c = 0; c < numChannels; c++) {
                const int vectorIndex = v + c * planeVectors;
                const float4 in = loadPixels4(input, vectorIndex);
                reinterpret_cast<uchar4*>(output)[vectorIndex] = make_uchar4(toPixel(in.x + us.x), toPixel(in.y + us.y), toPixel(in.z + us.z), toPixel(in.w + us.w));
            }
        }
    } else {
        // only the first plane starts 4-pixel aligned, the next ones are written per pixel
        for (int v = first; v < planeVectors; v += stride) {
            const float4 uv = toFloat4(reinterpret_cast<const Half4*>(u)[v]);
            const float us[4] = {uv.x * strength, uv.y * strength, uv.z * strength, uv.w * strength};
            const float4 in = loadPixels4(input, v);
            reinterpret_cast<uchar4*>(output)[v] = make_uchar4(toPixel(in.x + us[0]), toPixel(in.y + us[1]), toPixel(in.z + us[2]), toPixel(in.w + us[3]));
            for (int c = 1; c < numChannels; c++) {
                const int pixelIdx = 4 * v + c * planeElements;
#pragma unroll
                for (int i = 0; i < 4; i++)
                    output[pixelIdx + i] = toPixel(static_cast<float>(input[pixelIdx + i]) + us[i]);
            }
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
__global__ void apply_watermark_row_major(
    const float* __restrict__ input, const __half* __restrict__ u, const uint64_t* __restrict__ sumSq, uint8_t* __restrict__ output, const float strengthNumerator, const int width, const int height);
