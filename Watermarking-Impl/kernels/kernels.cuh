#pragma once
#include <cstdint>
#include <cub/cub.cuh>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// ITU-R 601 luma coefficients, shared across all kernels
static constexpr float kLumaR = 0.299f;
static constexpr float kLumaG = 0.587f;
static constexpr float kLumaB = 0.114f;

// convert FLOAT to UINT64 safely by multiplying with a very large power of 10 in order to not lose digits
// for converting back to float, we multiply with the inverse
__device__ inline uint64_t toScaledUint64(float value) { return static_cast<uint64_t>(value * 1000000000.0f); }
__device__ inline float toUnscaledFloat(uint64_t value) { return static_cast<float>(value * 1.0e-9f); }

// half8 struct for vectorized operations on 8 half values
struct alignas(16) half8 {
    half a, b, c, d, e, f, g, h;
};

// struct to hold correlation data for reduction (cub), used in correlation calculation kernels
struct CorrelationData {
    float dot;
    float normU;
    float normZ;
    __device__ __forceinline__ CorrelationData operator+(const CorrelationData& other) const { return {dot + other.dot, normU + other.normU, normZ + other.normZ}; }
};

// struct to hold rx vector values for reduction (cub) with vectorized addition with operator+
template <int N>
struct alignas(16) rxVecData {
    float vals[N];

    // it is better to initialize whenever we want instead of constructor
    // though for now we always initialize and immediately initialize to zero, but this may change in the future
    __device__ __forceinline__ void zero() {
#pragma unroll
        for (int i = 0; i < N; i++)
            vals[i] = 0.0f;
    }

    __device__ __forceinline__ rxVecData operator+(const rxVecData& other) const {
        rxVecData res;
#pragma unroll
        for (int i = 0; i < N; i++)
            res.vals[i] = vals[i] + other.vals[i];
        return res;
    }
};

// CUB Transform Functor for absolute value, used in error sequence calculation when we want absolute error sequence (for detection)
struct AbsTransformOp {
    __device__ __forceinline__ float operator()(const float& val) const { return fabsf(val); }
};

// maps a linear index k to(row, col) coordinates for a packed lower triangular matrix
__device__ inline int2 getPackedCoords(const int k) {
    // inverse triangular number formula: r = floor((sqrt(1 + 8k) - 1) / 2)
    const int r = __float2int_rd(0.5f * (sqrtf(1.0f + 8.0f * k) - 1.0f));
    const int c = k - (r * (r + 1)) / 2;
    return make_int2(r, c);
}

// helper methods to clamp a value between two limits
inline __device__ float clamp(float f, float a, float b) { return fmaxf(a, fminf(f, b)); }
inline __device__ int clamp(int f, int a, int b) { return max(a, min(f, b)); }

// helper function to fill block-wide shared memory cooperatively for error sequence and NVF kernels
// sharedMem must be rectangle with dimensions [shDimFast][shDimSlow + 1]
template <bool FUSED, int p, int shDimFast, int shDimSlow>
__device__ __forceinline__ void fillBlock(const float* __restrict__ inputA, const float* __restrict__ inputB, float* __restrict__ sharedMem, const int width, const int height) {
    constexpr int pad = p / 2;
    constexpr int totalElements = shDimFast * shDimSlow;

    const int baseGlobalX = (int)(blockIdx.y * blockDim.y) - pad;
    const int baseGlobalY = (int)(blockIdx.x * blockDim.x) - pad;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    // cooperatively fill 2D shared memory
    for (int i = tid; i < totalElements; i += blockDim.x * blockDim.y) {
        const int r = i % shDimFast;
        const int c = i / shDimFast;
        const int globalX = clamp(baseGlobalX + c, 0, width - 1);
        const int globalY = clamp(baseGlobalY + r, 0, height - 1);
        const int idx = globalX * height + globalY;
        float val = inputA[idx];
        // if we need to fuse (A*B), do it here, branch-free because it its known at compile time
        if constexpr (FUSED)
            val *= inputB[idx];
        sharedMem[i] = val;
    }
}

// helper function to fill block-wide shared memory cooperatively for ME kernels
// optimized to minimize uncoalesced reads and warp stalls
template <int p, int PixelsPerBlock, int StripHeight = PixelsPerBlock + p - 1>
__device__ __forceinline__ void fillBlockStripVertical(half blockValues[p][StripHeight], const float* __restrict__ input, const int width, const int height, const int bx, const int by) {
    constexpr float scaleFactor = 0.00392156862f;
    constexpr int radius = (p - 1) / 2;
    constexpr int totalPixels = p * StripHeight;
    constexpr int blockSize = p <= 5 ? 256 : 128; // note: blockDim.x is slower here! it is important to use constexpr for this critical hot loop!

    const int baseGlobalCol = (bx * 1) - radius;
    const int baseGlobalRow = (by * PixelsPerBlock) - radius;
    int idx = threadIdx.x;
    while (idx < totalPixels) {
        const int r = idx % StripHeight;
        const int c = idx / StripHeight;
        const int globalCol = clamp(baseGlobalCol + c, 0, width - 1);
        const int globalRow = clamp(baseGlobalRow + r, 0, height - 1);
        blockValues[c][r] = __float2half(input[(globalCol * height) + globalRow] * scaleFactor);
        idx += blockSize;
    }
}

// NVF mask calculation helper for a block
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

    // calculate NVF with optimized math (avoid divisions)
    const float numerator = (nPixels * sumSq) - (sum * sum);
    const float output = __fdividef(numerator, nPixelsSq + numerator);
    return clamp(output, 0.0f, 1.0f);
}

// NVF mask calculation and write to global memory, used for detection only
template <int p>
__global__ void nvf(const float* __restrict__ input, float* __restrict__ nvf, const int width, const int height) {
    constexpr int pad = p / 2;
    constexpr int shDimFast = 32 + (2 * pad);
    constexpr int shDimSlow = 8 + (2 * pad);

    const int x = blockIdx.y * blockDim.y + threadIdx.y;
    const int y = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ alignas(16) float region[shDimSlow][shDimFast];

    fillBlock<false, p, shDimFast, shDimSlow>(input, nullptr, &region[0][0], width, height);
    __syncthreads();

    if (x >= width || y >= height)
        return;

    const int shSlow = threadIdx.y + pad;
    const int shFast = threadIdx.x + pad;
    nvf[x * height + y] = compute_nvf_mask<p, shDimFast, shDimSlow>(region, shSlow, shFast);
}

// NVF mask calculation AND u calculation fused in one kernel to save global memory bandwidth and increase speed
// this is used ONLY for embedding, not for detection, because in detection we need the error sequence of the mask itself
template <int p>
__global__ void nvf_u_and_sumsq_fused(const float* __restrict__ input, const float* __restrict__ w, float* __restrict__ u, uint64_t* __restrict__ globalSumSq, const int width, const int height) {
    constexpr int pad = p / 2;
    constexpr int shDimFast = 32 + (2 * pad);
    constexpr int shDimSlow = 8 + (2 * pad);

    const int x = blockIdx.y * blockDim.y + threadIdx.y;
    const int y = blockIdx.x * blockDim.x + threadIdx.x;
    const int linearTid = threadIdx.y * blockDim.x + threadIdx.x;

    using BlockReduceT = cub::BlockReduce<float, 32, cub::BLOCK_REDUCE_WARP_REDUCTIONS, 8>; // block size is {32, 8}
    __shared__ alignas(16) float region[shDimSlow][shDimFast];
    __shared__ typename BlockReduceT::TempStorage temp_storage;

    fillBlock<false, p, shDimFast, shDimSlow>(input, nullptr, &region[0][0], width, height);
    __syncthreads();

    // default to 0 so out of bounds threads don't corrupt the CUB reduction
    float threadSumSq = 0.0f;
    if (x < width && y < height) {
        const int shSlow = threadIdx.y + pad;
        const int shFast = threadIdx.x + pad;
        // calculate the mask value and fuse u calculation with local sum of squares of u
        const float maskVal = compute_nvf_mask<p, shDimFast, shDimSlow>(region, shSlow, shFast);
        const int idx = x * height + y;
        const float uVal = maskVal * w[idx];
        u[idx] = uVal;
        // local sum for CUB
        threadSumSq = uVal * uVal;
    }

    // block reduce with cub and atomic add to global sum by the leader
    const float blockTotalSq = BlockReduceT(temp_storage).Sum(threadSumSq);
    if (linearTid == 0)
        atomicAdd(globalSumSq, toScaledUint64(blockTotalSq));
}

// main kernel for error sequence calculation
template <int p>
__global__ void calculate_error_sequence(const float* __restrict__ inputA, const float* __restrict__ inputB, float* __restrict__ x_, const float* __restrict__ coeffs, const int width,
                                         const int height, const bool calculateAbs, const int* __restrict__ stopFlag) {
    constexpr int pad = p / 2;
    constexpr int coeffsSize = (p * p) - 1;
    constexpr int shDimFast = 32 + (2 * pad);
    constexpr int shDimSlow = 8 + (2 * pad);

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;

    __shared__ alignas(16) float region[shDimSlow][shDimFast];
    __shared__ alignas(16) float sCoeffs[coeffsSize];

    if (tid < coeffsSize)
        sCoeffs[tid] = coeffs[tid];
    fillBlock<false, p, shDimFast, shDimSlow>(inputA, inputB, &region[0][0], width, height);
    __syncthreads();

    const int y = blockIdx.x * blockDim.x + threadIdx.x;
    const int x = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        if (*stopFlag) {
            x_[(x * height + y)] = 0.0f;
            return;
        }
        const int shFast = threadIdx.x + pad;
        const int shSlow = threadIdx.y + pad;
        float dot = 0.0f;
        int k = 0;
#pragma unroll
        for (int i = -pad; i <= pad; i++) {
#pragma unroll
            for (int j = -pad; j <= pad; j++) {
                if (i == 0 && j == 0)
                    continue; // skip the center pixel (branch is optimized fully at compile time)
                dot += sCoeffs[k] * region[shSlow + i][shFast + j];
                k++;
            }
        }
        const float errorSequence = region[shSlow][shFast] - dot;
        x_[(x * height + y)] = calculateAbs ? fabsf(errorSequence) : errorSequence;
    }
}

// main fused kernel for correlation calculation (error sequence and partial correlation), used in detection
template <int p>
__global__ void calculate_error_sequence_and_partial_corr_fused(const float* __restrict__ mask, const float* __restrict__ w, const float* __restrict__ e_u, const float* __restrict__ coeffs,
                                                                float* __restrict__ partialDots, float* __restrict__ partialNormU, float* __restrict__ partialNormZ, const int width, const int height,
                                                                const int* __restrict__ stopFlag) {
    constexpr int pad = p / 2;
    constexpr int coeffsSize = (p * p) - 1;
    constexpr int shDimFast = 32 + (2 * pad);
    constexpr int shDimSlow = 8 + (2 * pad);

    const int linearTid = threadIdx.y * blockDim.x + threadIdx.x;

    using BlockReduceT = cub::BlockReduce<CorrelationData, 32, cub::BLOCK_REDUCE_WARP_REDUCTIONS, 8>;
    __shared__ typename BlockReduceT::TempStorage temp_storage;
    __shared__ alignas(16) float region[shDimSlow][shDimFast];
    __shared__ alignas(16) float sCoeffs[coeffsSize];

    if (linearTid < coeffsSize)
        sCoeffs[linearTid] = coeffs[linearTid];

    fillBlock<true, p, shDimFast, shDimSlow>(mask, w, &region[0][0], width, height);
    __syncthreads();

    const int y = blockIdx.x * blockDim.x + threadIdx.x;
    const int x = blockIdx.y * blockDim.y + threadIdx.y;

    // default to zero for threads out of bounds (or if stopFlag says so)
    CorrelationData threadData = {0.0f, 0.0f, 0.0f};

    if (x < width && y < height && !(*stopFlag)) {
        const int shFast = threadIdx.x + pad;
        const int shSlow = threadIdx.y + pad;
        float dot = 0.0f;
        int k = 0;

#pragma unroll
        for (int i = -pad; i <= pad; i++) {
#pragma unroll
            for (int j = -pad; j <= pad; j++) {
                if (i == 0 && j == 0)
                    continue;
                dot += sCoeffs[k] * region[shSlow + i][shFast + j];
                k++;
            }
        }
        // calculate the e_z pixel
        const float ez = region[shSlow][shFast] - dot;
        // fused read of e_u and compute of correlation values
        const float eu = e_u[x * height + y];
        threadData.dot = eu * ez;
        threadData.normU = eu * eu;
        threadData.normZ = ez * ez;
    }

    // CUB block reduce, thread 0 writes the partials for this block
    const CorrelationData blockSum = BlockReduceT(temp_storage).Sum(threadData);
    if (linearTid == 0) {
        const int blockIdxFlat = blockIdx.y * gridDim.x + blockIdx.x;
        partialDots[blockIdxFlat] = blockSum.dot;
        partialNormU[blockIdxFlat] = blockSum.normU;
        partialNormZ[blockIdxFlat] = blockSum.normZ;
    }
}

// helper method used to accumulate "rx" vector values in the ME kernel, we template the outer loop to fully unroll it and gain maximum performance
template <int N>
__device__ __forceinline__ void accumulateRxVec(const half8* __restrict__ localVec8, float* __restrict__ rxVals, const float center) {
#pragma unroll
    for (int i = 0; i < N; i++) {
        const half2* inPtr = reinterpret_cast<const half2*>(&localVec8[i]);
        float2* rxPtr = reinterpret_cast<float2*>(&rxVals[i * 8]);
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const float2 in_f2 = __half22float2(inPtr[j]);
            rxPtr[j].x = fmaf(in_f2.x, center, rxPtr[j].x);
            rxPtr[j].y = fmaf(in_f2.y, center, rxPtr[j].y);
        }
    }
}

// helper method used to reduce "rx" vector values and atomic add them (all threads cooperate)
template <int SIZE, typename StorageT>
__device__ __forceinline__ void writeRxVec(uint64_t* __restrict__ rx, const rxVecData<SIZE>& rxData, StorageT& temp_storage, float* __restrict__ warpStaging) {
    const rxVecData<SIZE> warpSum = cub::WarpReduce<rxVecData<SIZE>>(temp_storage).Sum(rxData);
    if ((threadIdx.x & 31) == 0) {
#pragma unroll
        for (int i = 0; i < SIZE; i++)
            warpStaging[i] = warpSum.vals[i];
    }
    __syncwarp();
    // cooperative global atomicAdd
#pragma unroll
    for (int i = threadIdx.x & 31; i < SIZE; i += 32)
        atomicAdd(rx + i, toScaledUint64(warpStaging[i]));
}

// this function reverts the transpose introduced by the ME kernel. To achieve coalesced VRAM reads the image was loaded as column-major, this transposed
// the resulting system of equations. Because the Rx matrix is symmetric, the cholesky solver solves this transposed system, but outputs the
// coefficients in column-major order. Here we re-map this to row-major (we skip the center because it is not part of the predictor!)
template <int p>
__device__ __forceinline__ int getMappedVarIndex(const int k) {
    constexpr int center = (p * p) / 2;
    constexpr int p2_minus_1 = ((p * p) - 1);

    const int kPixel = k + (k >= center);
    const int r = kPixel / p;
    const int originalPixel = (kPixel * p) - (r * p2_minus_1);
    return originalPixel - (originalPixel > center);
}

// naive 1-thread Cholesky solver used for its very low latency versus cuSOLVER but useful only for very small systems, p = 3 (N = 8) or p = 5 (N = 24)
template <int p>
__global__ void cholesky_solver(const uint64_t* __restrict__ A, const uint64_t* __restrict__ B, float* __restrict__ X, int* __restrict__ stopFlag) {
    static_assert(p <= 5, "Simple 1-thread cholesky solver kernel should NEVER be instantiated for p > 5");
    constexpr int N = (p * p) - 1;

    auto IDX = [](const int r, const int c) { return (r * (r + 1)) / 2 + c; };

    if (threadIdx.x > 0 || blockIdx.x > 0)
        return;

    // packed format: N*(N+1)/2 elements
    // p=3 (N=8) -> 36 floats
    // p=5 (N=24) -> 300 floats
    constexpr int SIZE = (N * (N + 1)) / 2;
    float alignas(16) packed[SIZE];
    float alignas(16) localB[N];

    // check if A, B, and X are 16byte aligned for vectorized loads
    const bool isAligned = (((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B) | reinterpret_cast<uintptr_t>(X)) & 0xF) == 0);
    if (isAligned) {
        constexpr int vecLimitA = SIZE / 2;
        constexpr int vecLimitB = N / 2;
        const ulonglong2* vecA = reinterpret_cast<const ulonglong2*>(A);
        const ulonglong2* vecB = reinterpret_cast<const ulonglong2*>(B);
#pragma unroll
        for (int k = 0; k < vecLimitA; k++) {
            const ulonglong2 v = vecA[k];
            packed[k * 2 + 0] = toUnscaledFloat(v.x);
            packed[k * 2 + 1] = toUnscaledFloat(v.y);
        }
        for (int k = vecLimitA << 1; k < SIZE; k++)
            packed[k] = toUnscaledFloat(A[k]);

#pragma unroll
        for (int i = 0; i < vecLimitB; i++) {
            const ulonglong2 v = vecB[i];
            localB[i * 2 + 0] = toUnscaledFloat(v.x);
            localB[i * 2 + 1] = toUnscaledFloat(v.y);
        }
        for (int i = vecLimitB << 1; i < N; i++)
            localB[i] = toUnscaledFloat(B[i]);
    } else {
        // scalar path
#pragma unroll
        for (int k = 0; k < SIZE; k++)
            packed[k] = toUnscaledFloat(A[k]);
#pragma unroll
        for (int i = 0; i < N; i++)
            localB[i] = toUnscaledFloat(B[i]);
    }

    // in-place Cholesky Decomposition
    // overwrite packed (which holds A) with L
#pragma unroll
    for (int i = 0; i < N; i++) {
#pragma unroll
        for (int j = 0; j <= i; j++) {
            float sum = 0.0f;
            // dot product of previous L rows
#pragma unroll
            for (int k = 0; k < j; k++)
                sum += packed[IDX(i, k)] * packed[IDX(j, k)];
            if (i == j) {
                const float val = packed[IDX(i, i)] - sum;
                if (val <= 1e-12f) {
                    *stopFlag = 1;
                    goto exit;
                }
                packed[IDX(i, i)] = sqrtf(val);
            } else {
                // off-diagonal, packed[IDX(j, j)] was updated in previous iteration of j loop
                packed[IDX(i, j)] = (packed[IDX(i, j)] - sum) * __frcp_rn(packed[IDX(j, j)]);
            }
        }
    }

    // forward Substitution (solve L * y = b)
    // use packed[IDX(i, k)] which now holds L_ik
#pragma unroll
    for (int i = 0; i < N; i++) {
        float sum = 0.0f;
#pragma unroll
        for (int k = 0; k < i; k++)
            sum += packed[IDX(i, k)] * localB[k];
        localB[i] = (localB[i] - sum) * __frcp_rn(packed[IDX(i, i)]);
    }

    // backward Substitution (solve L^T * x = y)
    // solving U * x = y where U = L^T
    // U_ik = L_ki, we need L_ki, since k > i in the logic (upper tri), we access packed[IDX(k, i)] because we stored lower
#pragma unroll
    for (int i = N - 1; i >= 0; i--) {
        float sum = 0.0f;
#pragma unroll
        for (int k = i + 1; k < N; k++)
            sum += packed[IDX(k, i)] * localB[k];
        localB[i] = (localB[i] - sum) * __frcp_rn(packed[IDX(i, i)]);
    }
    *stopFlag = 0;
exit:
    // write Result
#pragma unroll
    for (int i = 0; i < N; i++)
        X[getMappedVarIndex<p>(i)] = localB[i];
}

// parallel cholesky solver for p = 7 (N = 48) and p = 9 (N = 80), using one warp (32 threads)
template <int p>
__global__ void cholesky_solver_parallel(const uint64_t* __restrict__ A, const uint64_t* __restrict__ B, float* __restrict__ X, int* __restrict__ stopFlag) {
    static_assert(p > 5, "Parallel cholesky solver kernel should NEVER be instantiated for p <= 5");
    constexpr int N = (p * p) - 1;
    constexpr int packedSize = (N * (N + 1)) / 2;

    const int laneId = threadIdx.x;

    __shared__ alignas(16) float sA[N][N + 1]; // +1 to avoid bank conflicts
    __shared__ alignas(16) float sB[N];

    // cooperative load (packedSize elements -> NxN Shared)
    // check if A, B, and X are 16-byte aligned for vectorized loads
    const bool isAligned = ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B) | reinterpret_cast<uintptr_t>(X)) & 0xF) == 0;
    // Rx
    if (isAligned) {
        constexpr int vecPackedLimit = packedSize / 2;
        const ulonglong2* vecA = reinterpret_cast<const ulonglong2*>(A);
        // vectorized path
        for (int k = laneId; k < vecPackedLimit; k += 32) {
            ulonglong2 v = vecA[k];
            const int baseIdx = k * 2;
            const int2 c0 = getPackedCoords(baseIdx + 0);
            sA[c0.x][c0.y] = toUnscaledFloat(v.x);
            const int2 c1 = getPackedCoords(baseIdx + 1);
            sA[c1.x][c1.y] = toUnscaledFloat(v.y);
        }
        // scalar path
    } else {
        for (int k = laneId; k < packedSize; k += 32) {
            const int2 c = getPackedCoords(k);
            sA[c.x][c.y] = toUnscaledFloat(A[k]);
        }
    }
    // rx
    if (isAligned) {
        constexpr int vecBlimit = N / 2;
        const ulonglong2* vecB = reinterpret_cast<const ulonglong2*>(B);
        // vectorized path
        for (int k = laneId; k < vecBlimit; k += 32) {
            const ulonglong2 v = vecB[k];
            sB[k * 2 + 0] = toUnscaledFloat(v.x);
            sB[k * 2 + 1] = toUnscaledFloat(v.y);
        }
        // scalar path
    } else {
        for (int k = laneId; k < N; k += 32)
            sB[k] = toUnscaledFloat(B[k]);
    }

    // initialize stop flag
    if (laneId == 0)
        *stopFlag = 0;
    __syncwarp();

    // in-place Cholesky Decomposition
    for (int k = 0; k < N; k++) {
        // check diagonal and calculate sqrt
        const float diag = sA[k][k];
        int abortFlag = 0;
        float invDiag = 0.0f;
        if (laneId == 0) {
            if (diag <= 1e-12f) {
                *stopFlag = 1; // write by 1 thread only
                abortFlag = 1;
            } else {
                invDiag = rsqrtf(diag);
                sA[k][k] = invDiag;
            }
        }

        // broadcast abort
        const int abortWarp = __shfl_sync(0xFFFFFFFF, abortFlag, 0);
        if (abortWarp)
            return;

        // broadcast L_kk
        const float L_kk_inv = __shfl_sync(0xFFFFFFFF, invDiag, 0);

        for (int i = k + 1 + laneId; i < N; i += 32)
            sA[i][k] = sA[i][k] * L_kk_inv;
        __syncwarp();

        // update trailing matrix
        for (int j = k + 1; j < N; j += 2) {
            const float L_jk0 = sA[j][k];
            const float L_jk1 = (j + 1 < N) ? sA[j + 1][k] : 0.0f;
            for (int i = j + laneId; i < N; i += 32) {
                const float Lik = sA[i][k];
                sA[i][j] = sA[i][j] - (Lik * L_jk0);
                if (j + 1 < N && i >= j + 1)
                    sA[i][j + 1] = sA[i][j + 1] - (Lik * L_jk1);
            }
        }
        __syncwarp();
    }

    // forward Substitution (solve L * y = b)
    for (int k = 0; k < N; k++) {
        float val = sB[k];
        if (laneId == 0) {
            val *= sA[k][k];
            sB[k] = val;
        }
        const float y_k = __shfl_sync(0xFFFFFFFF, val, 0);
        for (int i = k + 1 + laneId; i < N; i += 32)
            sB[i] = sB[i] - (sA[i][k] * y_k);
        __syncwarp();
    }

    // backward Substitution (solve L^T * x = y)
    // solving U * x = y where U = L^T
    for (int k = N - 1; k >= 0; k--) {
        float val = sB[k];
        if (laneId == 0) {
            val *= sA[k][k];
            sB[k] = val;
        }
        const float x_k = __shfl_sync(0xFFFFFFFF, val, 0);
        for (int i = laneId; i < k; i += 32)
            sB[i] = sB[i] - (sA[k][i] * x_k);
        __syncwarp();
    }

    // write Result
    for (int k = laneId; k < N; k += 32)
        X[getMappedVarIndex<p>(k)] = sB[k];
}

// helper method to perform a streaming reduction of Rx values from shared window to global memory
template <int startIdx, int endIdx, int rowOffset>
__device__ void RxStreamPass(const int tid, float (*__restrict__ RxLocal)[92], uint64_t* __restrict__ Rx) {
    for (int k = startIdx + tid; k < endIdx; k += 128) {
        const int2 coords = getPackedCoords(k);
        const int rowInWindow = coords.x - rowOffset;
        float sum = 0.0f;
#pragma unroll
        for (int w = 0; w < 4; w++)
            sum += RxLocal[w * 32 + rowInWindow][coords.y];
        atomicAdd(Rx + k, toScaledUint64(sum));
    }
}

// helper funnel shift functions, used to load neighbors ("patches") by shifting half values with funnel shift
template <int p>
__device__ void load_neighbor_row_funnel(half* dst, const half* rowBase, const int col) {
    // if our starting column is odd, the data we want starts "halfway" in a 32-bit chunk, we must shift right by 16 bits
    const uint32_t shift = (col & 1) * 16;
    // force the pointer to the nearest 32-bit aligned place, by masking the lowest bit (of the column index)
    const uint32_t* ptr = reinterpret_cast<const uint32_t*>(&rowBase[col & ~1]);
    // window extract, we read two 32-bit chunks, concat them into 64-bits and funnel shift right to get the 32-bit (half2) window we want
#pragma unroll
    for (int i = 0; i < p / 2; i++) {
        uint32_t pair = __funnelshift_r(ptr[i], ptr[i + 1], shift);
        reinterpret_cast<half2*>(dst)[i] = reinterpret_cast<half2&>(pair);
    }
    // p is always odd -> we will always have one dangling half left over at the end of the row, We shift it and extract the lowest 16 bits
    uint32_t lastChunk = ptr[p / 2] >> shift;
    dst[p - 1] = reinterpret_cast<half2&>(lastChunk).x;
}

template <int p, int RowStride>
__device__ void load_neighbor_vec(half8* dst, const half blockValues[p][RowStride], half& center, const int col) {
    constexpr int centerIdx = p / 2;

    // pad the row width to p + 1 (forcing an even number of elements), combined with alignas(4) this guarantees every row starts on 4-byte boundary,
    // ensuring the reinterpret_cast<half2*> in the funnel function never throws misaligned access error
    alignas(4) half rows[p][p + 1];
#pragma unroll
    for (int i = 0; i < p; i++)
        load_neighbor_row_funnel<p>(rows[i], blockValues[i], col);
    // extract the center pixel
    center = rows[centerIdx][centerIdx];
    // flatten the remaining PxP window into a 1D vector array
    half* d = reinterpret_cast<half*>(dst);
    int idx = 0;
#pragma unroll
    for (int r = 0; r < p; r++) {
#pragma unroll
        for (int c = 0; c < p; c++) {
            if (r == centerIdx && c == centerIdx)
                continue;
            d[idx++] = rows[r][c];
        }
    }
    // for p=5 only clear the last vec (last 8 halfs) for the WMMA path later
    if constexpr (p == 5)
        dst[3] = {};
}

// Prediction Error kernels (ME) for p = 3, p = 5, p = 7 and p = 9
__global__ void me_p3(const float* __restrict__ input, uint64_t* __restrict__ Rx, uint64_t* __restrict__ rx, const int width, const int height, const int totalBlocksY, const int taskTotal);
__global__ void me_p5(const float* __restrict__ input, uint64_t* __restrict__ Rx, uint64_t* __restrict__ rx, const int width, const int height, const int totalBlocksY, const int taskTotal);
__global__ void me_p7(const float* __restrict__ input, uint64_t* __restrict__ Rx, uint64_t* __restrict__ rx, const int width, const int height, const int totalBlocksY, const int taskTotal);
__global__ void me_p9(const float* __restrict__ input, uint64_t* __restrict__ Rx, uint64_t* __restrict__ rx, const int width, const int height, const int totalBlocksY, const int taskTotal);

// fused calculation of ME mask, u, and sum of squares
__global__ void me_u_and_sumsq_fused(const float* __restrict__ errorSeq, const float* __restrict__ w, float* __restrict__ u, uint64_t* __restrict__ globalSumSq, const float* __restrict__ maxVal,
                                     const int N);

// fused application of watermark: applies the watermark and calculates the output in one pass, using the precomputed u and sum of squares for normalization
__global__ void apply_watermark_fused(const float* __restrict__ input, const float* __restrict__ u, const uint64_t* __restrict__ sumSqPtr, uint8_t* __restrict__ output, const float strengthNumerator,
                                      const int planeElements, const int numChannels);

// calculation of the absolute value of the error sequence normalized by its max value, used in detection of ME mask
__global__ void compute_abs_normalized_mask(const float* __restrict__ errorSeq, float* __restrict__ mask, const float* __restrict__ maxVal, const int N);

// main kernel for correlation calculation, used in detection
__global__ void calculate_final_correlation(const float* __restrict__ partialDots, const float* __restrict__ partialNormU, const float* __restrict__ partialNormZ, float* __restrict__ result,
                                            const int numBlocks);

// used for converting NV12 to YUV420p format, used in HW accelerated video decoding
__global__ void nV12ToYUV420p(const uint8_t* __restrict__ uvSrc, const int uvPitch, uint8_t* __restrict__ uvDst, const int uvWidth, const int uvHeight);

// used for converting uint8 pitched memory to non pitched float, used in HW accelerated video decoding
__global__ void pitchedToFloat(const uint8_t* __restrict__ input, float* __restrict__ output, const int width, const int height, const int pitch);

// uint8 col-major (1 or 3 channel) to float col-major grayscale, with optional RGB weighting
__global__ void u8ToFloatGray(const uint8_t* __restrict__ input, float* __restrict__ output, const int planeSize, const int numChannels);

// used for converting column-major uint8 GPU array back to row-major uint8, multichannel via z-dimension
__global__ void colMajorToRowMajorU8(const uint8_t* __restrict__ src, uint8_t* __restrict__ dst, const int width, const int height);

// used for converting row-major float (CImg) to column-major float (CudaArray), coalesced tiled transpose
__global__ void rowMajorToColMajorFloat(const float* __restrict__ src, float* __restrict__ dst, const int width, const int height);

// fused row-major 3-channel RGB to col-major grayscale with ITU-R 601 luma weights
__global__ void rowMajorRGBToColMajorGray(const float* __restrict__ src, float* __restrict__ dst, const int width, const int height);

// HDR (P010LE, BT.2020 PQ) to SDR (BT.709) helpers

// offline generated 1024 entry LUTs for PQ EOTF and BT.1886 gamma, always accessed via __ldg
// main reason to use them is to avoid many powf() per thread which is very slow
static __device__ const float pqEotfLUT[1024] = {
    0.00000000e+00f, 4.04227176e-07f, 1.31113719e-06f, 2.62368259e-06f, 4.31514955e-06f, 6.37468853e-06f, 8.79823827e-06f, 1.15853619e-05f, 1.47378191e-05f, 1.82588184e-05f, 2.21525860e-05f,
    2.64240984e-05f, 3.10789073e-05f, 3.61230211e-05f, 4.15628212e-05f, 4.74050009e-05f, 5.36565210e-05f, 6.03245762e-05f, 6.74165685e-05f, 7.49400881e-05f, 8.29028967e-05f, 9.13129157e-05f,
    1.00178216e-04f, 1.09507010e-04f, 1.19307645e-04f, 1.29588598e-04f, 1.40358472e-04f, 1.51625993e-04f, 1.63400004e-04f, 1.75689469e-04f, 1.88503463e-04f, 2.01851179e-04f, 2.15741921e-04f,
    2.30185106e-04f, 2.45190261e-04f, 2.60767028e-04f, 2.76925156e-04f, 2.93674508e-04f, 3.11025056e-04f, 3.28986884e-04f, 3.47570188e-04f, 3.66785276e-04f, 3.86642568e-04f, 4.07152596e-04f,
    4.28326007e-04f, 4.50173561e-04f, 4.72706132e-04f, 4.95934711e-04f, 5.19870404e-04f, 5.44524435e-04f, 5.69908144e-04f, 5.96032991e-04f, 6.22910555e-04f, 6.50552534e-04f, 6.78970751e-04f,
    7.08177146e-04f, 7.38183787e-04f, 7.69002863e-04f, 8.00646690e-04f, 8.33127708e-04f, 8.66458488e-04f, 9.00651724e-04f, 9.35720245e-04f, 9.71677006e-04f, 1.00853510e-03f, 1.04630774e-03f,
    1.08500828e-03f, 1.12465022e-03f, 1.16524719e-03f, 1.20681293e-03f, 1.24936136e-03f, 1.29290652e-03f, 1.33746260e-03f, 1.38304390e-03f, 1.42966491e-03f, 1.47734024e-03f, 1.52608464e-03f,
    1.57591302e-03f, 1.62684044e-03f, 1.67888210e-03f, 1.73205336e-03f, 1.78636971e-03f, 1.84184684e-03f, 1.89850054e-03f, 1.95634680e-03f, 2.01540174e-03f, 2.07568166e-03f, 2.13720301e-03f,
    2.19998240e-03f, 2.26403661e-03f, 2.32938257e-03f, 2.39603741e-03f, 2.46401838e-03f, 2.53334295e-03f, 2.60402872e-03f, 2.67609348e-03f, 2.74955520e-03f, 2.82443200e-03f, 2.90074222e-03f,
    2.97850434e-03f, 3.05773702e-03f, 3.13845914e-03f, 3.22068972e-03f, 3.30444799e-03f, 3.38975335e-03f, 3.47662541e-03f, 3.56508395e-03f, 3.65514894e-03f, 3.74684057e-03f, 3.84017918e-03f,
    3.93518536e-03f, 4.03187985e-03f, 4.13028361e-03f, 4.23041782e-03f, 4.33230383e-03f, 4.43596321e-03f, 4.54141774e-03f, 4.64868941e-03f, 4.75780041e-03f, 4.86877315e-03f, 4.98163025e-03f,
    5.09639456e-03f, 5.21308912e-03f, 5.33173722e-03f, 5.45236236e-03f, 5.57498825e-03f, 5.69963885e-03f, 5.82633832e-03f, 5.95511108e-03f, 6.08598176e-03f, 6.21897523e-03f, 6.35411660e-03f,
    6.49143121e-03f, 6.63094464e-03f, 6.77268271e-03f, 6.91667151e-03f, 7.06293733e-03f, 7.21150675e-03f, 7.36240657e-03f, 7.51566386e-03f, 7.67130594e-03f, 7.82936040e-03f, 7.98985505e-03f,
    8.15281800e-03f, 8.31827761e-03f, 8.48626251e-03f, 8.65680159e-03f, 8.82992402e-03f, 9.00565922e-03f, 9.18403692e-03f, 9.36508710e-03f, 9.54884004e-03f, 9.73532628e-03f, 9.92457666e-03f,
    1.01166223e-02f, 1.03114946e-02f, 1.05092253e-02f, 1.07098464e-02f, 1.09133902e-02f, 1.11198892e-02f, 1.13293765e-02f, 1.15418851e-02f, 1.17574485e-02f, 1.19761007e-02f, 1.21978758e-02f,
    1.24228081e-02f, 1.26509325e-02f, 1.28822840e-02f, 1.31168981e-02f, 1.33548105e-02f, 1.35960573e-02f, 1.38406748e-02f, 1.40886999e-02f, 1.43401695e-02f, 1.45951210e-02f, 1.48535924e-02f,
    1.51156215e-02f, 1.53812470e-02f, 1.56505076e-02f, 1.59234424e-02f, 1.62000910e-02f, 1.64804933e-02f, 1.67646895e-02f, 1.70527202e-02f, 1.73446264e-02f, 1.76404495e-02f, 1.79402312e-02f,
    1.82440135e-02f, 1.85518391e-02f, 1.88637508e-02f, 1.91797919e-02f, 1.95000060e-02f, 1.98244373e-02f, 2.01531301e-02f, 2.04861294e-02f, 2.08234804e-02f, 2.11652288e-02f, 2.15114208e-02f,
    2.18621028e-02f, 2.22173218e-02f, 2.25771252e-02f, 2.29415608e-02f, 2.33106767e-02f, 2.36845218e-02f, 2.40631449e-02f, 2.44465958e-02f, 2.48349245e-02f, 2.52281813e-02f, 2.56264171e-02f,
    2.60296834e-02f, 2.64380319e-02f, 2.68515150e-02f, 2.72701854e-02f, 2.76940963e-02f, 2.81233015e-02f, 2.85578551e-02f, 2.89978119e-02f, 2.94432270e-02f, 2.98941560e-02f, 3.03506553e-02f,
    3.08127813e-02f, 3.12805914e-02f, 3.17541431e-02f, 3.22334948e-02f, 3.27187051e-02f, 3.32098334e-02f, 3.37069393e-02f, 3.42100832e-02f, 3.47193261e-02f, 3.52347292e-02f, 3.57563545e-02f,
    3.62842645e-02f, 3.68185224e-02f, 3.73591917e-02f, 3.79063366e-02f, 3.84600219e-02f, 3.90203129e-02f, 3.95872756e-02f, 4.01609764e-02f, 4.07414825e-02f, 4.13288615e-02f, 4.19231818e-02f,
    4.25245123e-02f, 4.31329224e-02f, 4.37484824e-02f, 4.43712629e-02f, 4.50013355e-02f, 4.56387720e-02f, 4.62836451e-02f, 4.69360282e-02f, 4.75959952e-02f, 4.82636206e-02f, 4.89389799e-02f,
    4.96221488e-02f, 5.03132041e-02f, 5.10122230e-02f, 5.17192834e-02f, 5.24344640e-02f, 5.31578442e-02f, 5.38895040e-02f, 5.46295242e-02f, 5.53779862e-02f, 5.61349722e-02f, 5.69005652e-02f,
    5.76748488e-02f, 5.84579073e-02f, 5.92498260e-02f, 6.00506906e-02f, 6.08605878e-02f, 6.16796050e-02f, 6.25078304e-02f, 6.33453529e-02f, 6.41922623e-02f, 6.50486489e-02f, 6.59146043e-02f,
    6.67902204e-02f, 6.76755901e-02f, 6.85708073e-02f, 6.94759664e-02f, 7.03911630e-02f, 7.13164931e-02f, 7.22520538e-02f, 7.31979432e-02f, 7.41542599e-02f, 7.51211036e-02f, 7.60985748e-02f,
    7.70867750e-02f, 7.80858064e-02f, 7.90957722e-02f, 8.01167765e-02f, 8.11489243e-02f, 8.21923215e-02f, 8.32470749e-02f, 8.43132924e-02f, 8.53910827e-02f, 8.64805554e-02f, 8.75818211e-02f,
    8.86949916e-02f, 8.98201792e-02f, 9.09574977e-02f, 9.21070615e-02f, 9.32689862e-02f, 9.44433884e-02f, 9.56303856e-02f, 9.68300965e-02f, 9.80426406e-02f, 9.92681386e-02f, 1.00506712e-01f,
    1.01758485e-01f, 1.03023579e-01f, 1.04302121e-01f, 1.05594236e-01f, 1.06900052e-01f, 1.08219696e-01f, 1.09553298e-01f, 1.10900989e-01f, 1.12262900e-01f, 1.13639164e-01f, 1.15029915e-01f,
    1.16435287e-01f, 1.17855418e-01f, 1.19290444e-01f, 1.20740505e-01f, 1.22205740e-01f, 1.23686289e-01f, 1.25182296e-01f, 1.26693905e-01f, 1.28221258e-01f, 1.29764504e-01f, 1.31323789e-01f,
    1.32899261e-01f, 1.34491070e-01f, 1.36099369e-01f, 1.37724308e-01f, 1.39366042e-01f, 1.41024727e-01f, 1.42700518e-01f, 1.44393573e-01f, 1.46104052e-01f, 1.47832114e-01f, 1.49577924e-01f,
    1.51341642e-01f, 1.53123435e-01f, 1.54923468e-01f, 1.56741910e-01f, 1.58578929e-01f, 1.60434696e-01f, 1.62309383e-01f, 1.64203164e-01f, 1.66116213e-01f, 1.68048707e-01f, 1.70000825e-01f,
    1.71972746e-01f, 1.73964651e-01f, 1.75976722e-01f, 1.78009146e-01f, 1.80062106e-01f, 1.82135791e-01f, 1.84230390e-01f, 1.86346094e-01f, 1.88483096e-01f, 1.90641589e-01f, 1.92821769e-01f,
    1.95023833e-01f, 1.97247982e-01f, 1.99494416e-01f, 2.01763337e-01f, 2.04054951e-01f, 2.06369463e-01f, 2.08707081e-01f, 2.11068015e-01f, 2.13452476e-01f, 2.15860679e-01f, 2.18292838e-01f,
    2.20749170e-01f, 2.23229895e-01f, 2.25735233e-01f, 2.28265408e-01f, 2.30820643e-01f, 2.33401166e-01f, 2.36007205e-01f, 2.38638991e-01f, 2.41296757e-01f, 2.43980736e-01f, 2.46691166e-01f,
    2.49428286e-01f, 2.52192336e-01f, 2.54983558e-01f, 2.57802199e-01f, 2.60648504e-01f, 2.63522723e-01f, 2.66425108e-01f, 2.69355912e-01f, 2.72315390e-01f, 2.75303801e-01f, 2.78321404e-01f,
    2.81368462e-01f, 2.84445240e-01f, 2.87552004e-01f, 2.90689024e-01f, 2.93856571e-01f, 2.97054920e-01f, 3.00284345e-01f, 3.03545127e-01f, 3.06837546e-01f, 3.10161885e-01f, 3.13518430e-01f,
    3.16907471e-01f, 3.20329297e-01f, 3.23784202e-01f, 3.27272482e-01f, 3.30794436e-01f, 3.34350364e-01f, 3.37940570e-01f, 3.41565361e-01f, 3.45225045e-01f, 3.48919934e-01f, 3.52650342e-01f,
    3.56416586e-01f, 3.60218986e-01f, 3.64057864e-01f, 3.67933546e-01f, 3.71846360e-01f, 3.75796636e-01f, 3.79784709e-01f, 3.83810915e-01f, 3.87875593e-01f, 3.91979086e-01f, 3.96121740e-01f,
    4.00303903e-01f, 4.04525926e-01f, 4.08788163e-01f, 4.13090973e-01f, 4.17434716e-01f, 4.21819756e-01f, 4.26246459e-01f, 4.30715196e-01f, 4.35226340e-01f, 4.39780267e-01f, 4.44377357e-01f,
    4.49017992e-01f, 4.53702561e-01f, 4.58431451e-01f, 4.63205056e-01f, 4.68023772e-01f, 4.72887999e-01f, 4.77798141e-01f, 4.82754605e-01f, 4.87757799e-01f, 4.92808139e-01f, 4.97906042e-01f,
    5.03051929e-01f, 5.08246224e-01f, 5.13489355e-01f, 5.18781756e-01f, 5.24123861e-01f, 5.29516110e-01f, 5.34958946e-01f, 5.40452817e-01f, 5.45998174e-01f, 5.51595471e-01f, 5.57245168e-01f,
    5.62947727e-01f, 5.68703615e-01f, 5.74513303e-01f, 5.80377267e-01f, 5.86295984e-01f, 5.92269938e-01f, 5.98299617e-01f, 6.04385513e-01f, 6.10528120e-01f, 6.16727940e-01f, 6.22985477e-01f,
    6.29301240e-01f, 6.35675742e-01f, 6.42109500e-01f, 6.48603037e-01f, 6.55156880e-01f, 6.61771560e-01f, 6.68447613e-01f, 6.75185579e-01f, 6.81986004e-01f, 6.88849438e-01f, 6.95776435e-01f,
    7.02767555e-01f, 7.09823362e-01f, 7.16944425e-01f, 7.24131320e-01f, 7.31384624e-01f, 7.38704922e-01f, 7.46092804e-01f, 7.53548864e-01f, 7.61073702e-01f, 7.68667921e-01f, 7.76332133e-01f,
    7.84066952e-01f, 7.91873000e-01f, 7.99750901e-01f, 8.07701289e-01f, 8.15724799e-01f, 8.23822074e-01f, 8.31993763e-01f, 8.40240518e-01f, 8.48563000e-01f, 8.56961874e-01f, 8.65437810e-01f,
    8.73991485e-01f, 8.82623582e-01f, 8.91334789e-01f, 9.00125801e-01f, 9.08997318e-01f, 9.17950048e-01f, 9.26984703e-01f, 9.36102002e-01f, 9.45302670e-01f, 9.54587439e-01f, 9.63957047e-01f,
    9.73412239e-01f, 9.82953764e-01f, 9.92582382e-01f, 1.00229886e+00f, 1.01210396e+00f, 1.02199846e+00f, 1.03198315e+00f, 1.04205882e+00f, 1.05222627e+00f, 1.06248630e+00f, 1.07283972e+00f,
    1.08328736e+00f, 1.09383004e+00f, 1.10446858e+00f, 1.11520384e+00f, 1.12603667e+00f, 1.13696790e+00f, 1.14799842e+00f, 1.15912909e+00f, 1.17036078e+00f, 1.18169439e+00f, 1.19313080e+00f,
    1.20467092e+00f, 1.21631566e+00f, 1.22806594e+00f, 1.23992267e+00f, 1.25188679e+00f, 1.26395925e+00f, 1.27614099e+00f, 1.28843297e+00f, 1.30083615e+00f, 1.31335152e+00f, 1.32598006e+00f,
    1.33872275e+00f, 1.35158060e+00f, 1.36455461e+00f, 1.37764581e+00f, 1.39085523e+00f, 1.40418389e+00f, 1.41763284e+00f, 1.43120314e+00f, 1.44489586e+00f, 1.45871205e+00f, 1.47265282e+00f,
    1.48671924e+00f, 1.50091242e+00f, 1.51523348e+00f, 1.52968352e+00f, 1.54426369e+00f, 1.55897513e+00f, 1.57381897e+00f, 1.58879639e+00f, 1.60390856e+00f, 1.61915666e+00f, 1.63454187e+00f,
    1.65006541e+00f, 1.66572848e+00f, 1.68153231e+00f, 1.69747813e+00f, 1.71356719e+00f, 1.72980074e+00f, 1.74618005e+00f, 1.76270640e+00f, 1.77938108e+00f, 1.79620539e+00f, 1.81318065e+00f,
    1.83030816e+00f, 1.84758928e+00f, 1.86502536e+00f, 1.88261773e+00f, 1.90036779e+00f, 1.91827692e+00f, 1.93634650e+00f, 1.95457796e+00f, 1.97297270e+00f, 1.99153216e+00f, 2.01025780e+00f,
    2.02915105e+00f, 2.04821341e+00f, 2.06744635e+00f, 2.08685137e+00f, 2.10642998e+00f, 2.12618371e+00f, 2.14611409e+00f, 2.16622268e+00f, 2.18651104e+00f, 2.20698074e+00f, 2.22763339e+00f,
    2.24847060e+00f, 2.26949397e+00f, 2.29070515e+00f, 2.31210579e+00f, 2.33369755e+00f, 2.35548212e+00f, 2.37746119e+00f, 2.39963646e+00f, 2.42200967e+00f, 2.44458256e+00f, 2.46735687e+00f,
    2.49033439e+00f, 2.51351690e+00f, 2.53690620e+00f, 2.56050411e+00f, 2.58431247e+00f, 2.60833313e+00f, 2.63256796e+00f, 2.65701884e+00f, 2.68168768e+00f, 2.70657640e+00f, 2.73168692e+00f,
    2.75702121e+00f, 2.78258124e+00f, 2.80836899e+00f, 2.83438647e+00f, 2.86063570e+00f, 2.88711873e+00f, 2.91383762e+00f, 2.94079445e+00f, 2.96799130e+00f, 2.99543031e+00f, 3.02311360e+00f,
    3.05104334e+00f, 3.07922168e+00f, 3.10765084e+00f, 3.13633301e+00f, 3.16527044e+00f, 3.19446537e+00f, 3.22392008e+00f, 3.25363686e+00f, 3.28361803e+00f, 3.31386591e+00f, 3.34438288e+00f,
    3.37517130e+00f, 3.40623357e+00f, 3.43757211e+00f, 3.46918937e+00f, 3.50108780e+00f, 3.53326990e+00f, 3.56573816e+00f, 3.59849513e+00f, 3.63154336e+00f, 3.66488542e+00f, 3.69852391e+00f,
    3.73246145e+00f, 3.76670070e+00f, 3.80124433e+00f, 3.83609502e+00f, 3.87125550e+00f, 3.90672851e+00f, 3.94251683e+00f, 3.97862324e+00f, 4.01505056e+00f, 4.05180165e+00f, 4.08887936e+00f,
    4.12628660e+00f, 4.16402628e+00f, 4.20210137e+00f, 4.24051483e+00f, 4.27926966e+00f, 4.31836890e+00f, 4.35781560e+00f, 4.39761286e+00f, 4.43776377e+00f, 4.47827149e+00f, 4.51913918e+00f,
    4.56037005e+00f, 4.60196732e+00f, 4.64393424e+00f, 4.68627412e+00f, 4.72899025e+00f, 4.77208600e+00f, 4.81556473e+00f, 4.85942985e+00f, 4.90368482e+00f, 4.94833309e+00f, 4.99337817e+00f,
    5.03882359e+00f, 5.08467292e+00f, 5.13092977e+00f, 5.17759775e+00f, 5.22468054e+00f, 5.27218184e+00f, 5.32010538e+00f, 5.36845493e+00f, 5.41723428e+00f, 5.46644727e+00f, 5.51609779e+00f,
    5.56618972e+00f, 5.61672702e+00f, 5.66771366e+00f, 5.71915367e+00f, 5.77105108e+00f, 5.82341000e+00f, 5.87623454e+00f, 5.92952888e+00f, 5.98329721e+00f, 6.03754378e+00f, 6.09227287e+00f,
    6.14748879e+00f, 6.20319592e+00f, 6.25939864e+00f, 6.31610141e+00f, 6.37330870e+00f, 6.43102503e+00f, 6.48925497e+00f, 6.54800312e+00f, 6.60727415e+00f, 6.66707273e+00f, 6.72740360e+00f,
    6.78827154e+00f, 6.84968139e+00f, 6.91163799e+00f, 6.97414628e+00f, 7.03721120e+00f, 7.10083776e+00f, 7.16503102e+00f, 7.22979606e+00f, 7.29513804e+00f, 7.36106215e+00f, 7.42757363e+00f,
    7.49467777e+00f, 7.56237991e+00f, 7.63068543e+00f, 7.69959978e+00f, 7.76912844e+00f, 7.83927695e+00f, 7.91005092e+00f, 7.98145597e+00f, 8.05349781e+00f, 8.12618219e+00f, 8.19951490e+00f,
    8.27350182e+00f, 8.34814884e+00f, 8.42346194e+00f, 8.49944713e+00f, 8.57611051e+00f, 8.65345819e+00f, 8.73149639e+00f, 8.81023134e+00f, 8.88966936e+00f, 8.96981682e+00f, 9.05068014e+00f,
    9.13226582e+00f, 9.21458040e+00f, 9.29763050e+00f, 9.38142279e+00f, 9.46596400e+00f, 9.55126093e+00f, 9.63732045e+00f, 9.72414948e+00f, 9.81175502e+00f, 9.90014412e+00f, 9.98932391e+00f,
    1.00793016e+01f, 1.01700844e+01f, 1.02616797e+01f, 1.03540948e+01f, 1.04473373e+01f, 1.05414147e+01f, 1.06363345e+01f, 1.07321045e+01f, 1.08287324e+01f, 1.09262260e+01f, 1.10245933e+01f,
    1.11238422e+01f, 1.12239808e+01f, 1.13250172e+01f, 1.14269595e+01f, 1.15298162e+01f, 1.16335954e+01f, 1.17383058e+01f, 1.18439558e+01f, 1.19505539e+01f, 1.20581090e+01f, 1.21666297e+01f,
    1.22761249e+01f, 1.23866035e+01f, 1.24980746e+01f, 1.26105472e+01f, 1.27240306e+01f, 1.28385339e+01f, 1.29540667e+01f, 1.30706384e+01f, 1.31882584e+01f, 1.33069364e+01f, 1.34266822e+01f,
    1.35475056e+01f, 1.36694165e+01f, 1.37924249e+01f, 1.39165409e+01f, 1.40417748e+01f, 1.41681367e+01f, 1.42956372e+01f, 1.44242868e+01f, 1.45540959e+01f, 1.46850754e+01f, 1.48172361e+01f,
    1.49505887e+01f, 1.50851445e+01f, 1.52209145e+01f, 1.53579098e+01f, 1.54961419e+01f, 1.56356223e+01f, 1.57763623e+01f, 1.59183739e+01f, 1.60616686e+01f, 1.62062584e+01f, 1.63521553e+01f,
    1.64993715e+01f, 1.66479191e+01f, 1.67978106e+01f, 1.69490585e+01f, 1.71016752e+01f, 1.72556736e+01f, 1.74110665e+01f, 1.75678668e+01f, 1.77260878e+01f, 1.78857425e+01f, 1.80468444e+01f,
    1.82094069e+01f, 1.83734436e+01f, 1.85389684e+01f, 1.87059950e+01f, 1.88745375e+01f, 1.90446101e+01f, 1.92162270e+01f, 1.93894027e+01f, 1.95641517e+01f, 1.97404888e+01f, 1.99184288e+01f,
    2.00979867e+01f, 2.02791777e+01f, 2.04620170e+01f, 2.06465202e+01f, 2.08327028e+01f, 2.10205805e+01f, 2.12101694e+01f, 2.14014853e+01f, 2.15945447e+01f, 2.17893638e+01f, 2.19859591e+01f,
    2.21843475e+01f, 2.23845457e+01f, 2.25865708e+01f, 2.27904401e+01f, 2.29961707e+01f, 2.32037805e+01f, 2.34132869e+01f, 2.36247080e+01f, 2.38380617e+01f, 2.40533664e+01f, 2.42706405e+01f,
    2.44899026e+01f, 2.47111715e+01f, 2.49344661e+01f, 2.51598056e+01f, 2.53872094e+01f, 2.56166971e+01f, 2.58482884e+01f, 2.60820032e+01f, 2.63178616e+01f, 2.65558841e+01f, 2.67960912e+01f,
    2.70385035e+01f, 2.72831421e+01f, 2.75300282e+01f, 2.77791830e+01f, 2.80306282e+01f, 2.82843856e+01f, 2.85404772e+01f, 2.87989253e+01f, 2.90597523e+01f, 2.93229810e+01f, 2.95886341e+01f,
    2.98567349e+01f, 3.01273068e+01f, 3.04003734e+01f, 3.06759584e+01f, 3.09540861e+01f, 3.12347807e+01f, 3.15180669e+01f, 3.18039693e+01f, 3.20925132e+01f, 3.23837239e+01f, 3.26776268e+01f,
    3.29742480e+01f, 3.32736134e+01f, 3.35757494e+01f, 3.38806827e+01f, 3.41884402e+01f, 3.44990490e+01f, 3.48125366e+01f, 3.51289307e+01f, 3.54482593e+01f, 3.57705507e+01f, 3.60958336e+01f,
    3.64241366e+01f, 3.67554891e+01f, 3.70899205e+01f, 3.74274605e+01f, 3.77681392e+01f, 3.81119870e+01f, 3.84590345e+01f, 3.88093127e+01f, 3.91628530e+01f, 3.95196869e+01f, 3.98798465e+01f,
    4.02433639e+01f, 4.06102718e+01f, 4.09806031e+01f, 4.13543911e+01f, 4.17316695e+01f, 4.21124721e+01f, 4.24968333e+01f, 4.28847878e+01f, 4.32763706e+01f, 4.36716170e+01f, 4.40705627e+01f,
    4.44732440e+01f, 4.48796973e+01f, 4.52899594e+01f, 4.57040677e+01f, 4.61220596e+01f, 4.65439732e+01f, 4.69698470e+01f, 4.73997196e+01f, 4.78336304e+01f, 4.82716190e+01f, 4.87137253e+01f,
    4.91599897e+01f, 4.96104533e+01f, 5.00651571e+01f, 5.05241429e+01f, 5.09874530e+01f, 5.14551297e+01f, 5.19272162e+01f, 5.24037560e+01f, 5.28847928e+01f, 5.33703712e+01f, 5.38605360e+01f,
    5.43553325e+01f, 5.48548064e+01f, 5.53590040e+01f, 5.58679721e+01f, 5.63817579e+01f, 5.69004092e+01f, 5.74239741e+01f, 5.79525014e+01f, 5.84860404e+01f, 5.90246408e+01f, 5.95683528e+01f,
    6.01172274e+01f, 6.06713159e+01f, 6.12306701e+01f, 6.17953425e+01f, 6.23653861e+01f, 6.29408545e+01f, 6.35218017e+01f, 6.41082824e+01f, 6.47003520e+01f, 6.52980662e+01f, 6.59014816e+01f,
    6.65106550e+01f, 6.71256443e+01f, 6.77465076e+01f, 6.83733038e+01f, 6.90060925e+01f, 6.96449337e+01f, 7.02898882e+01f, 7.09410175e+01f, 7.15983836e+01f, 7.22620492e+01f, 7.29320778e+01f,
    7.36085334e+01f, 7.42914808e+01f, 7.49809855e+01f, 7.56771136e+01f, 7.63799320e+01f, 7.70895083e+01f, 7.78059107e+01f, 7.85292084e+01f, 7.92594710e+01f, 7.99967692e+01f, 8.07411742e+01f,
    8.14927580e+01f, 8.22515935e+01f, 8.30177544e+01f, 8.37913150e+01f, 8.45723505e+01f, 8.53609369e+01f, 8.61571512e+01f, 8.69610709e+01f, 8.77727747e+01f, 8.85923417e+01f, 8.94198524e+01f,
    9.02553878e+01f, 9.10990298e+01f, 9.19508613e+01f, 9.28109661e+01f, 9.36794288e+01f, 9.45563352e+01f, 9.54417716e+01f, 9.63358255e+01f, 9.72385855e+01f, 9.81501408e+01f, 9.90705818e+01f,
    1.00000000e+02f,
};

static __device__ const float bt1886LUT[1024] = {
    0.00000000e+00f, 5.57038423e-02f, 7.43557087e-02f, 8.80411437e-02f, 9.92529634e-02f, 1.08923767e-01f, 1.17520827e-01f, 1.25316811e-01f, 1.32486811e-01f, 1.39150957e-01f, 1.45395786e-01f,
    1.51286011e-01f, 1.56871484e-01f, 1.62191547e-01f, 1.67277874e-01f, 1.72156402e-01f, 1.76848676e-01f, 1.81372819e-01f, 1.85744243e-01f, 1.89976181e-01f, 1.94080090e-01f, 1.98065967e-01f,
    2.01942597e-01f, 2.05717743e-01f, 2.09398309e-01f, 2.12990462e-01f, 2.16499741e-01f, 2.19931138e-01f, 2.23289173e-01f, 2.26577954e-01f, 2.29801226e-01f, 2.32962414e-01f, 2.36064661e-01f,
    2.39110856e-01f, 2.42103667e-01f, 2.45045560e-01f, 2.47938819e-01f, 2.50785566e-01f, 2.53587778e-01f, 2.56347295e-01f, 2.59065839e-01f, 2.61745019e-01f, 2.64386347e-01f, 2.66991238e-01f,
    2.69561026e-01f, 2.72096966e-01f, 2.74600242e-01f, 2.77071971e-01f, 2.79513208e-01f, 2.81924955e-01f, 2.84308158e-01f, 2.86663716e-01f, 2.88992483e-01f, 2.91295269e-01f, 2.93572848e-01f,
    2.95825954e-01f, 2.98055287e-01f, 3.00261518e-01f, 3.02445284e-01f, 3.04607195e-01f, 3.06747836e-01f, 3.08867764e-01f, 3.10967515e-01f, 3.13047602e-01f, 3.15108517e-01f, 3.17150732e-01f,
    3.19174700e-01f, 3.21180858e-01f, 3.23169624e-01f, 3.25141402e-01f, 3.27096579e-01f, 3.29035531e-01f, 3.30958617e-01f, 3.32866184e-01f, 3.34758569e-01f, 3.36636094e-01f, 3.38499072e-01f,
    3.40347806e-01f, 3.42182586e-01f, 3.44003695e-01f, 3.45811406e-01f, 3.47605984e-01f, 3.49387683e-01f, 3.51156753e-01f, 3.52913432e-01f, 3.54657955e-01f, 3.56390545e-01f, 3.58111423e-01f,
    3.59820801e-01f, 3.61518885e-01f, 3.63205875e-01f, 3.64881966e-01f, 3.66547347e-01f, 3.68202202e-01f, 3.69846709e-01f, 3.71481042e-01f, 3.73105370e-01f, 3.74719858e-01f, 3.76324665e-01f,
    3.77919949e-01f, 3.79505860e-01f, 3.81082546e-01f, 3.82650153e-01f, 3.84208819e-01f, 3.85758683e-01f, 3.87299878e-01f, 3.88832535e-01f, 3.90356780e-01f, 3.91872737e-01f, 3.93380529e-01f,
    3.94880273e-01f, 3.96372084e-01f, 3.97856076e-01f, 3.99332359e-01f, 4.00801041e-01f, 4.02262226e-01f, 4.03716018e-01f, 4.05162518e-01f, 4.06601824e-01f, 4.08034032e-01f, 4.09459236e-01f,
    4.10877529e-01f, 4.12289001e-01f, 4.13693740e-01f, 4.15091832e-01f, 4.16483363e-01f, 4.17868416e-01f, 4.19247070e-01f, 4.20619407e-01f, 4.21985504e-01f, 4.23345437e-01f, 4.24699281e-01f,
    4.26047110e-01f, 4.27388996e-01f, 4.28725009e-01f, 4.30055219e-01f, 4.31379694e-01f, 4.32698499e-01f, 4.34011701e-01f, 4.35319364e-01f, 4.36621550e-01f, 4.37918322e-01f, 4.39209740e-01f,
    4.40495864e-01f, 4.41776752e-01f, 4.43052461e-01f, 4.44323049e-01f, 4.45588570e-01f, 4.46849079e-01f, 4.48104630e-01f, 4.49355275e-01f, 4.50601065e-01f, 4.51842052e-01f, 4.53078286e-01f,
    4.54309815e-01f, 4.55536688e-01f, 4.56758953e-01f, 4.57976655e-01f, 4.59189842e-01f, 4.60398558e-01f, 4.61602847e-01f, 4.62802754e-01f, 4.63998321e-01f, 4.65189590e-01f, 4.66376604e-01f,
    4.67559404e-01f, 4.68738029e-01f, 4.69912519e-01f, 4.71082915e-01f, 4.72249253e-01f, 4.73411572e-01f, 4.74569910e-01f, 4.75724303e-01f, 4.76874788e-01f, 4.78021400e-01f, 4.79164174e-01f,
    4.80303146e-01f, 4.81438348e-01f, 4.82569816e-01f, 4.83697581e-01f, 4.84821677e-01f, 4.85942137e-01f, 4.87058990e-01f, 4.88172270e-01f, 4.89282007e-01f, 4.90388232e-01f, 4.91490973e-01f,
    4.92590262e-01f, 4.93686127e-01f, 4.94778597e-01f, 4.95867700e-01f, 4.96953464e-01f, 4.98035918e-01f, 4.99115087e-01f, 5.00191000e-01f, 5.01263683e-01f, 5.02333161e-01f, 5.03399462e-01f,
    5.04462609e-01f, 5.05522629e-01f, 5.06579546e-01f, 5.07633385e-01f, 5.08684170e-01f, 5.09731925e-01f, 5.10776674e-01f, 5.11818439e-01f, 5.12857244e-01f, 5.13893112e-01f, 5.14926065e-01f,
    5.15956124e-01f, 5.16983313e-01f, 5.18007653e-01f, 5.19029164e-01f, 5.20047869e-01f, 5.21063787e-01f, 5.22076940e-01f, 5.23087348e-01f, 5.24095031e-01f, 5.25100008e-01f, 5.26102300e-01f,
    5.27101926e-01f, 5.28098905e-01f, 5.29093255e-01f, 5.30084997e-01f, 5.31074147e-01f, 5.32060725e-01f, 5.33044748e-01f, 5.34026235e-01f, 5.35005203e-01f, 5.35981669e-01f, 5.36955651e-01f,
    5.37927166e-01f, 5.38896231e-01f, 5.39862862e-01f, 5.40827077e-01f, 5.41788890e-01f, 5.42748319e-01f, 5.43705380e-01f, 5.44660087e-01f, 5.45612458e-01f, 5.46562507e-01f, 5.47510250e-01f,
    5.48455701e-01f, 5.49398876e-01f, 5.50339790e-01f, 5.51278457e-01f, 5.52214891e-01f, 5.53149108e-01f, 5.54081121e-01f, 5.55010944e-01f, 5.55938592e-01f, 5.56864078e-01f, 5.57787415e-01f,
    5.58708617e-01f, 5.59627698e-01f, 5.60544671e-01f, 5.61459548e-01f, 5.62372343e-01f, 5.63283068e-01f, 5.64191737e-01f, 5.65098361e-01f, 5.66002953e-01f, 5.66905526e-01f, 5.67806092e-01f,
    5.68704663e-01f, 5.69601250e-01f, 5.70495865e-01f, 5.71388521e-01f, 5.72279229e-01f, 5.73168000e-01f, 5.74054846e-01f, 5.74939778e-01f, 5.75822807e-01f, 5.76703945e-01f, 5.77583202e-01f,
    5.78460588e-01f, 5.79336116e-01f, 5.80209795e-01f, 5.81081636e-01f, 5.81951650e-01f, 5.82819847e-01f, 5.83686236e-01f, 5.84550829e-01f, 5.85413636e-01f, 5.86274666e-01f, 5.87133929e-01f,
    5.87991435e-01f, 5.88847194e-01f, 5.89701215e-01f, 5.90553508e-01f, 5.91404083e-01f, 5.92252948e-01f, 5.93100114e-01f, 5.93945589e-01f, 5.94789382e-01f, 5.95631503e-01f, 5.96471960e-01f,
    5.97310762e-01f, 5.98147919e-01f, 5.98983438e-01f, 5.99817329e-01f, 6.00649600e-01f, 6.01480260e-01f, 6.02309317e-01f, 6.03136779e-01f, 6.03962655e-01f, 6.04786953e-01f, 6.05609681e-01f,
    6.06430847e-01f, 6.07250460e-01f, 6.08068527e-01f, 6.08885055e-01f, 6.09700054e-01f, 6.10513530e-01f, 6.11325492e-01f, 6.12135947e-01f, 6.12944902e-01f, 6.13752365e-01f, 6.14558344e-01f,
    6.15362845e-01f, 6.16165877e-01f, 6.16967446e-01f, 6.17767560e-01f, 6.18566226e-01f, 6.19363451e-01f, 6.20159241e-01f, 6.20953605e-01f, 6.21746548e-01f, 6.22538078e-01f, 6.23328202e-01f,
    6.24116926e-01f, 6.24904257e-01f, 6.25690202e-01f, 6.26474767e-01f, 6.27257959e-01f, 6.28039784e-01f, 6.28820249e-01f, 6.29599360e-01f, 6.30377124e-01f, 6.31153546e-01f, 6.31928634e-01f,
    6.32702393e-01f, 6.33474830e-01f, 6.34245950e-01f, 6.35015760e-01f, 6.35784265e-01f, 6.36551473e-01f, 6.37317387e-01f, 6.38082016e-01f, 6.38845363e-01f, 6.39607436e-01f, 6.40368240e-01f,
    6.41127781e-01f, 6.41886064e-01f, 6.42643094e-01f, 6.43398879e-01f, 6.44153422e-01f, 6.44906730e-01f, 6.45658809e-01f, 6.46409662e-01f, 6.47159297e-01f, 6.47907718e-01f, 6.48654931e-01f,
    6.49400940e-01f, 6.50145752e-01f, 6.50889371e-01f, 6.51631802e-01f, 6.52373052e-01f, 6.53113123e-01f, 6.53852023e-01f, 6.54589756e-01f, 6.55326326e-01f, 6.56061739e-01f, 6.56796000e-01f,
    6.57529113e-01f, 6.58261084e-01f, 6.58991917e-01f, 6.59721618e-01f, 6.60450190e-01f, 6.61177638e-01f, 6.61903968e-01f, 6.62629183e-01f, 6.63353290e-01f, 6.64076291e-01f, 6.64798192e-01f,
    6.65518997e-01f, 6.66238710e-01f, 6.66957337e-01f, 6.67674882e-01f, 6.68391348e-01f, 6.69106741e-01f, 6.69821065e-01f, 6.70534324e-01f, 6.71246522e-01f, 6.71957664e-01f, 6.72667754e-01f,
    6.73376796e-01f, 6.74084794e-01f, 6.74791753e-01f, 6.75497676e-01f, 6.76202568e-01f, 6.76906433e-01f, 6.77609274e-01f, 6.78311097e-01f, 6.79011904e-01f, 6.79711700e-01f, 6.80410489e-01f,
    6.81108275e-01f, 6.81805061e-01f, 6.82500852e-01f, 6.83195651e-01f, 6.83889462e-01f, 6.84582289e-01f, 6.85274136e-01f, 6.85965007e-01f, 6.86654904e-01f, 6.87343833e-01f, 6.88031796e-01f,
    6.88718798e-01f, 6.89404841e-01f, 6.90089930e-01f, 6.90774068e-01f, 6.91457259e-01f, 6.92139507e-01f, 6.92820814e-01f, 6.93501184e-01f, 6.94180621e-01f, 6.94859128e-01f, 6.95536709e-01f,
    6.96213368e-01f, 6.96889106e-01f, 6.97563929e-01f, 6.98237839e-01f, 6.98910840e-01f, 6.99582934e-01f, 7.00254126e-01f, 7.00924418e-01f, 7.01593814e-01f, 7.02262317e-01f, 7.02929931e-01f,
    7.03596658e-01f, 7.04262501e-01f, 7.04927465e-01f, 7.05591551e-01f, 7.06254764e-01f, 7.06917105e-01f, 7.07578580e-01f, 7.08239189e-01f, 7.08898937e-01f, 7.09557826e-01f, 7.10215860e-01f,
    7.10873042e-01f, 7.11529374e-01f, 7.12184860e-01f, 7.12839502e-01f, 7.13493303e-01f, 7.14146267e-01f, 7.14798396e-01f, 7.15449693e-01f, 7.16100161e-01f, 7.16749803e-01f, 7.17398622e-01f,
    7.18046620e-01f, 7.18693801e-01f, 7.19340167e-01f, 7.19985720e-01f, 7.20630465e-01f, 7.21274402e-01f, 7.21917536e-01f, 7.22559869e-01f, 7.23201403e-01f, 7.23842142e-01f, 7.24482087e-01f,
    7.25121242e-01f, 7.25759610e-01f, 7.26397192e-01f, 7.27033992e-01f, 7.27670011e-01f, 7.28305254e-01f, 7.28939722e-01f, 7.29573417e-01f, 7.30206343e-01f, 7.30838502e-01f, 7.31469896e-01f,
    7.32100528e-01f, 7.32730400e-01f, 7.33359516e-01f, 7.33987876e-01f, 7.34615485e-01f, 7.35242343e-01f, 7.35868455e-01f, 7.36493821e-01f, 7.37118445e-01f, 7.37742329e-01f, 7.38365475e-01f,
    7.38987885e-01f, 7.39609563e-01f, 7.40230510e-01f, 7.40850728e-01f, 7.41470221e-01f, 7.42088989e-01f, 7.42707036e-01f, 7.43324364e-01f, 7.43940975e-01f, 7.44556872e-01f, 7.45172056e-01f,
    7.45786529e-01f, 7.46400295e-01f, 7.47013355e-01f, 7.47625711e-01f, 7.48237366e-01f, 7.48848322e-01f, 7.49458581e-01f, 7.50068145e-01f, 7.50677016e-01f, 7.51285196e-01f, 7.51892688e-01f,
    7.52499494e-01f, 7.53105615e-01f, 7.53711055e-01f, 7.54315814e-01f, 7.54919895e-01f, 7.55523300e-01f, 7.56126031e-01f, 7.56728090e-01f, 7.57329480e-01f, 7.57930201e-01f, 7.58530257e-01f,
    7.59129649e-01f, 7.59728379e-01f, 7.60326449e-01f, 7.60923861e-01f, 7.61520618e-01f, 7.62116720e-01f, 7.62712170e-01f, 7.63306971e-01f, 7.63901123e-01f, 7.64494629e-01f, 7.65087490e-01f,
    7.65679709e-01f, 7.66271287e-01f, 7.66862227e-01f, 7.67452530e-01f, 7.68042198e-01f, 7.68631232e-01f, 7.69219636e-01f, 7.69807410e-01f, 7.70394556e-01f, 7.70981077e-01f, 7.71566973e-01f,
    7.72152248e-01f, 7.72736901e-01f, 7.73320937e-01f, 7.73904355e-01f, 7.74487158e-01f, 7.75069348e-01f, 7.75650927e-01f, 7.76231895e-01f, 7.76812256e-01f, 7.77392010e-01f, 7.77971159e-01f,
    7.78549706e-01f, 7.79127651e-01f, 7.79704996e-01f, 7.80281744e-01f, 7.80857895e-01f, 7.81433452e-01f, 7.82008416e-01f, 7.82582789e-01f, 7.83156572e-01f, 7.83729768e-01f, 7.84302377e-01f,
    7.84874401e-01f, 7.85445842e-01f, 7.86016702e-01f, 7.86586982e-01f, 7.87156684e-01f, 7.87725809e-01f, 7.88294359e-01f, 7.88862335e-01f, 7.89429740e-01f, 7.89996574e-01f, 7.90562839e-01f,
    7.91128537e-01f, 7.91693669e-01f, 7.92258238e-01f, 7.92822243e-01f, 7.93385687e-01f, 7.93948572e-01f, 7.94510898e-01f, 7.95072668e-01f, 7.95633883e-01f, 7.96194544e-01f, 7.96754653e-01f,
    7.97314211e-01f, 7.97873220e-01f, 7.98431681e-01f, 7.98989595e-01f, 7.99546965e-01f, 8.00103792e-01f, 8.00660076e-01f, 8.01215820e-01f, 8.01771025e-01f, 8.02325692e-01f, 8.02879823e-01f,
    8.03433418e-01f, 8.03986481e-01f, 8.04539011e-01f, 8.05091010e-01f, 8.05642480e-01f, 8.06193422e-01f, 8.06743838e-01f, 8.07293728e-01f, 8.07843095e-01f, 8.08391938e-01f, 8.08940261e-01f,
    8.09488064e-01f, 8.10035348e-01f, 8.10582115e-01f, 8.11128367e-01f, 8.11674103e-01f, 8.12219327e-01f, 8.12764038e-01f, 8.13308239e-01f, 8.13851931e-01f, 8.14395115e-01f, 8.14937792e-01f,
    8.15479963e-01f, 8.16021630e-01f, 8.16562795e-01f, 8.17103458e-01f, 8.17643620e-01f, 8.18183283e-01f, 8.18722449e-01f, 8.19261117e-01f, 8.19799291e-01f, 8.20336970e-01f, 8.20874156e-01f,
    8.21410851e-01f, 8.21947055e-01f, 8.22482769e-01f, 8.23017996e-01f, 8.23552736e-01f, 8.24086990e-01f, 8.24620760e-01f, 8.25154046e-01f, 8.25686851e-01f, 8.26219174e-01f, 8.26751018e-01f,
    8.27282383e-01f, 8.27813271e-01f, 8.28343683e-01f, 8.28873619e-01f, 8.29403082e-01f, 8.29932072e-01f, 8.30460590e-01f, 8.30988638e-01f, 8.31516216e-01f, 8.32043326e-01f, 8.32569970e-01f,
    8.33096147e-01f, 8.33621859e-01f, 8.34147108e-01f, 8.34671894e-01f, 8.35196218e-01f, 8.35720082e-01f, 8.36243487e-01f, 8.36766433e-01f, 8.37288922e-01f, 8.37810955e-01f, 8.38332534e-01f,
    8.38853658e-01f, 8.39374329e-01f, 8.39894549e-01f, 8.40414318e-01f, 8.40933637e-01f, 8.41452508e-01f, 8.41970931e-01f, 8.42488908e-01f, 8.43006439e-01f, 8.43523526e-01f, 8.44040169e-01f,
    8.44556370e-01f, 8.45072130e-01f, 8.45587449e-01f, 8.46102329e-01f, 8.46616771e-01f, 8.47130776e-01f, 8.47644344e-01f, 8.48157477e-01f, 8.48670176e-01f, 8.49182441e-01f, 8.49694275e-01f,
    8.50205677e-01f, 8.50716648e-01f, 8.51227191e-01f, 8.51737305e-01f, 8.52246992e-01f, 8.52756252e-01f, 8.53265087e-01f, 8.53773497e-01f, 8.54281484e-01f, 8.54789048e-01f, 8.55296191e-01f,
    8.55802913e-01f, 8.56309216e-01f, 8.56815100e-01f, 8.57320565e-01f, 8.57825614e-01f, 8.58330247e-01f, 8.58834465e-01f, 8.59338269e-01f, 8.59841660e-01f, 8.60344639e-01f, 8.60847206e-01f,
    8.61349363e-01f, 8.61851110e-01f, 8.62352449e-01f, 8.62853379e-01f, 8.63353904e-01f, 8.63854022e-01f, 8.64353735e-01f, 8.64853044e-01f, 8.65351950e-01f, 8.65850453e-01f, 8.66348555e-01f,
    8.66846256e-01f, 8.67343558e-01f, 8.67840461e-01f, 8.68336965e-01f, 8.68833073e-01f, 8.69328784e-01f, 8.69824100e-01f, 8.70319021e-01f, 8.70813549e-01f, 8.71307684e-01f, 8.71801426e-01f,
    8.72294778e-01f, 8.72787739e-01f, 8.73280311e-01f, 8.73772494e-01f, 8.74264289e-01f, 8.74755697e-01f, 8.75246720e-01f, 8.75737356e-01f, 8.76227608e-01f, 8.76717477e-01f, 8.77206962e-01f,
    8.77696066e-01f, 8.78184788e-01f, 8.78673130e-01f, 8.79161092e-01f, 8.79648675e-01f, 8.80135880e-01f, 8.80622708e-01f, 8.81109159e-01f, 8.81595235e-01f, 8.82080936e-01f, 8.82566262e-01f,
    8.83051215e-01f, 8.83535796e-01f, 8.84020005e-01f, 8.84503843e-01f, 8.84987310e-01f, 8.85470408e-01f, 8.85953138e-01f, 8.86435499e-01f, 8.86917493e-01f, 8.87399121e-01f, 8.87880383e-01f,
    8.88361281e-01f, 8.88841814e-01f, 8.89321983e-01f, 8.89801790e-01f, 8.90281235e-01f, 8.90760319e-01f, 8.91239042e-01f, 8.91717406e-01f, 8.92195410e-01f, 8.92673056e-01f, 8.93150345e-01f,
    8.93627277e-01f, 8.94103853e-01f, 8.94580073e-01f, 8.95055939e-01f, 8.95531451e-01f, 8.96006610e-01f, 8.96481416e-01f, 8.96955870e-01f, 8.97429974e-01f, 8.97903727e-01f, 8.98377130e-01f,
    8.98850184e-01f, 8.99322890e-01f, 8.99795248e-01f, 9.00267260e-01f, 9.00738925e-01f, 9.01210244e-01f, 9.01681219e-01f, 9.02151850e-01f, 9.02622137e-01f, 9.03092082e-01f, 9.03561684e-01f,
    9.04030945e-01f, 9.04499865e-01f, 9.04968445e-01f, 9.05436685e-01f, 9.05904587e-01f, 9.06372151e-01f, 9.06839377e-01f, 9.07306267e-01f, 9.07772820e-01f, 9.08239038e-01f, 9.08704921e-01f,
    9.09170470e-01f, 9.09635685e-01f, 9.10100568e-01f, 9.10565118e-01f, 9.11029337e-01f, 9.11493225e-01f, 9.11956783e-01f, 9.12420011e-01f, 9.12882910e-01f, 9.13345480e-01f, 9.13807723e-01f,
    9.14269639e-01f, 9.14731228e-01f, 9.15192491e-01f, 9.15653429e-01f, 9.16114043e-01f, 9.16574332e-01f, 9.17034298e-01f, 9.17493942e-01f, 9.17953263e-01f, 9.18412262e-01f, 9.18870941e-01f,
    9.19329300e-01f, 9.19787338e-01f, 9.20245058e-01f, 9.20702459e-01f, 9.21159542e-01f, 9.21616308e-01f, 9.22072757e-01f, 9.22528890e-01f, 9.22984707e-01f, 9.23440210e-01f, 9.23895398e-01f,
    9.24350273e-01f, 9.24804834e-01f, 9.25259082e-01f, 9.25713019e-01f, 9.26166644e-01f, 9.26619959e-01f, 9.27072963e-01f, 9.27525658e-01f, 9.27978043e-01f, 9.28430120e-01f, 9.28881889e-01f,
    9.29333350e-01f, 9.29784505e-01f, 9.30235353e-01f, 9.30685896e-01f, 9.31136133e-01f, 9.31586066e-01f, 9.32035695e-01f, 9.32485021e-01f, 9.32934043e-01f, 9.33382763e-01f, 9.33831182e-01f,
    9.34279299e-01f, 9.34727115e-01f, 9.35174632e-01f, 9.35621848e-01f, 9.36068766e-01f, 9.36515385e-01f, 9.36961706e-01f, 9.37407729e-01f, 9.37853456e-01f, 9.38298886e-01f, 9.38744021e-01f,
    9.39188860e-01f, 9.39633404e-01f, 9.40077655e-01f, 9.40521611e-01f, 9.40965274e-01f, 9.41408645e-01f, 9.41851723e-01f, 9.42294510e-01f, 9.42737006e-01f, 9.43179211e-01f, 9.43621126e-01f,
    9.44062751e-01f, 9.44504088e-01f, 9.44945136e-01f, 9.45385896e-01f, 9.45826368e-01f, 9.46266554e-01f, 9.46706453e-01f, 9.47146066e-01f, 9.47585393e-01f, 9.48024436e-01f, 9.48463194e-01f,
    9.48901668e-01f, 9.49339858e-01f, 9.49777766e-01f, 9.50215391e-01f, 9.50652734e-01f, 9.51089796e-01f, 9.51526577e-01f, 9.51963077e-01f, 9.52399297e-01f, 9.52835237e-01f, 9.53270899e-01f,
    9.53706282e-01f, 9.54141387e-01f, 9.54576214e-01f, 9.55010764e-01f, 9.55445038e-01f, 9.55879035e-01f, 9.56312757e-01f, 9.56746203e-01f, 9.57179375e-01f, 9.57612272e-01f, 9.58044896e-01f,
    9.58477246e-01f, 9.58909323e-01f, 9.59341128e-01f, 9.59772662e-01f, 9.60203923e-01f, 9.60634914e-01f, 9.61065634e-01f, 9.61496084e-01f, 9.61926264e-01f, 9.62356175e-01f, 9.62785818e-01f,
    9.63215192e-01f, 9.63644299e-01f, 9.64073138e-01f, 9.64501710e-01f, 9.64930016e-01f, 9.65358056e-01f, 9.65785830e-01f, 9.66213339e-01f, 9.66640583e-01f, 9.67067564e-01f, 9.67494280e-01f,
    9.67920733e-01f, 9.68346924e-01f, 9.68772852e-01f, 9.69198517e-01f, 9.69623922e-01f, 9.70049065e-01f, 9.70473947e-01f, 9.70898570e-01f, 9.71322932e-01f, 9.71747035e-01f, 9.72170879e-01f,
    9.72594464e-01f, 9.73017792e-01f, 9.73440861e-01f, 9.73863674e-01f, 9.74286229e-01f, 9.74708528e-01f, 9.75130571e-01f, 9.75552359e-01f, 9.75973891e-01f, 9.76395169e-01f, 9.76816192e-01f,
    9.77236962e-01f, 9.77657478e-01f, 9.78077741e-01f, 9.78497751e-01f, 9.78917509e-01f, 9.79337015e-01f, 9.79756270e-01f, 9.80175273e-01f, 9.80594026e-01f, 9.81012529e-01f, 9.81430782e-01f,
    9.81848786e-01f, 9.82266541e-01f, 9.82684047e-01f, 9.83101305e-01f, 9.83518315e-01f, 9.83935078e-01f, 9.84351593e-01f, 9.84767863e-01f, 9.85183886e-01f, 9.85599663e-01f, 9.86015194e-01f,
    9.86430481e-01f, 9.86845523e-01f, 9.87260321e-01f, 9.87674875e-01f, 9.88089186e-01f, 9.88503253e-01f, 9.88917078e-01f, 9.89330661e-01f, 9.89744001e-01f, 9.90157100e-01f, 9.90569958e-01f,
    9.90982575e-01f, 9.91394952e-01f, 9.91807089e-01f, 9.92218986e-01f, 9.92630644e-01f, 9.93042063e-01f, 9.93453243e-01f, 9.93864186e-01f, 9.94274891e-01f, 9.94685358e-01f, 9.95095588e-01f,
    9.95505582e-01f, 9.95915339e-01f, 9.96324861e-01f, 9.96734147e-01f, 9.97143198e-01f, 9.97552014e-01f, 9.97960595e-01f, 9.98368943e-01f, 9.98777057e-01f, 9.99184937e-01f, 9.99592585e-01f,
    1.00000000e+00f};

// LUT lookup with linear interpolation between adjacent values
__device__ __forceinline__ float lutLerp1024(const float* __restrict__ lut, const float x01) {
    const float idx = clamp(x01, 0.0f, 1.0f) * 1023.0f;
    const int i0 = __float2int_rd(idx);
    const int i1 = min(i0 + 1, 1023);
    const float frac = idx - static_cast<float>(i0);
    const float v0 = __ldg(&lut[i0]);
    const float v1 = __ldg(&lut[i1]);
    return fmaf(v1 - v0, frac, v0);
}

// Mobius tonemapping: exact FFmpeg vf_tonemap.c formula: K * (x+a)/(x+b)
// a, b, K depend ONLY on hdrPeak so they are precomputed on the host
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
    const float Y2020 = clamp(fmaf(static_cast<float>(yRaw >> 6), 1.0f / 876.0f, -64.0f / 876.0f), 0.0f, 1.0f);
    const float Cb2020 = fmaf(static_cast<float>(cbRaw >> 6), 1.0f / 896.0f, -512.0f / 896.0f);
    const float Cr2020 = fmaf(static_cast<float>(crRaw >> 6), 1.0f / 896.0f, -512.0f / 896.0f);
    // BT.2020 YCbCr -> PQ-encoded R'G'B' (Kr=0.2627, Kb=0.0593), explicit FMAs!
    const float Rp = clamp(fmaf(1.4746f, Cr2020, Y2020), 0.0f, 1.0f);
    const float Gp = clamp(fmaf(-0.16455f, Cb2020, fmaf(-0.57135f, Cr2020, Y2020)), 0.0f, 1.0f);
    const float Bp = clamp(fmaf(1.8814f, Cb2020, Y2020), 0.0f, 1.0f);
    // PQ EOTF -> linear RGB (BT.2020), npl=100 is premultiplied in the LUT (we save 3 MULs per thread)
    float R = lutLerp1024(pqEotfLUT, Rp);
    float G = lutLerp1024(pqEotfLUT, Gp);
    float B = lutLerp1024(pqEotfLUT, Bp);
    // highlight desaturation
    constexpr float desat = 2.0f;
    const float luma2020 = fmaf(0.2627f, R, fmaf(0.6780f, G, 0.0593f * B));
    const float overbright = fmaxf(luma2020 - desat, 1e-6f) * __frcp_rn(fmaxf(luma2020, 1e-6f)); // fast reciprocal too
    R = fmaf(overbright, luma2020 - R, R);
    G = fmaf(overbright, luma2020 - G, G);
    B = fmaf(overbright, luma2020 - B, B);
    // hue preserving Mobius, tonemap MAX channel, then scale ALL 3 uniformly to preserve hue
    const float sig = fmaxf(fmaxf(R, G), B);
    if (sig > 1e-6f) {
        const float scale = clamp(mobiusTonemap(sig, mobA, mobB, mobK), 0.0f, 1.0f) * __frcp_rn(sig);
        R *= scale;
        G *= scale;
        B *= scale;
    } else {
        R = G = B = 0.0f;
    }
    // BT.2020 -> BT.709 gamut matrix with clamping
    const float R7 = clamp(fmaf(1.6605f, R, fmaf(-0.5876f, G, -0.0728f * B)), 0.0f, 1.0f);
    const float G7 = clamp(fmaf(-0.1246f, R, fmaf(1.1329f, G, -0.0083f * B)), 0.0f, 1.0f);
    const float B7 = clamp(fmaf(-0.0182f, R, fmaf(-0.1006f, G, 1.1187f * B)), 0.0f, 1.0f);
    // BT.1886 gamma (gamma 2.4) -> display referred [0,1]
    return make_float3(lutLerp1024(bt1886LUT, R7), lutLerp1024(bt1886LUT, G7), lutLerp1024(bt1886LUT, B7));
}

// BT.709 RGB [0,1] -> Y limited range [16, 235] (encoder expects limited range Y for SDR yuv420p)
__device__ __forceinline__ float rgbToYLimited(const float3 rgb) {
    const float Y = fmaf(0.2126f, rgb.x, fmaf(0.7152f, rgb.y, 0.0722f * rgb.z));
    return clamp(fmaf(Y, 219.0f, 16.0f), 16.0f, 235.0f);
}

// HDR Y: P010LE pitched + UV -> col-major float [16,235] limited range (equivalent to pitchedToFloat for HDR frames)
// Needs UV for per pixel hue preserving tonemap
__global__ void p010HdrYToSdrFloat(const uint16_t* __restrict__ ySrc, const int yPitchBytes, const uint16_t* __restrict__ uvSrc, const int uvPitchBytes, float* __restrict__ output, const int width,
                                   const int height, const float mobA, const float mobB, const float mobK);

// HDR UV: P010LE interleaved UV + Y -> uint8_t interleaved NV12 UV with hue preserving BT.2020->BT.709 tonemap
__global__ void p010HdrUVToSdrNV12(const uint16_t* __restrict__ ySrc, const int yPitchBytes, const uint16_t* __restrict__ uvSrc, const int uvPitchBytes, uint8_t* __restrict__ uvDst, const int width,
                                   const int height, const float mobA, const float mobB, const float mobK);

// HDR Y: P010LE pitched + UV -> uint8_t row-major limited range [16,235] (passthrough encoding without watermarking)
__global__ void p010HdrYToSdrU8(const uint16_t* __restrict__ ySrc, const int yPitchBytes, const uint16_t* __restrict__ uvSrc, const int uvPitchBytes, uint8_t* __restrict__ output, const int width,
                                const int height, const float mobA, const float mobB, const float mobK);