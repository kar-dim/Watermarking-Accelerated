#pragma once
#include <cstdint>
#include <cub/cub.cuh>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

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
template <int p, int BlockSize, int StripHeight = BlockSize + p - 1>
__device__ __forceinline__ void fillBlockStripVertical(half blockValues[p][StripHeight], const float* __restrict__ input, const int width, const int height, const int bx, const int by) {
    constexpr float scaleFactor = 0.00392156862f;
    constexpr int radius = (p - 1) / 2;
    constexpr int totalPixels = p * StripHeight;

    const int baseGlobalCol = (bx * 1) - radius;
    const int baseGlobalRow = (by * BlockSize) - radius;
    int idx = threadIdx.x;
    while (idx < totalPixels) {
        const int r = idx % StripHeight;
        const int c = idx / StripHeight;
        const int globalCol = clamp(width - 1, 0, baseGlobalCol + c);
        const int globalRow = clamp(height - 1, 0, baseGlobalRow + r);
        blockValues[c][r] = __float2half(input[(globalCol * height) + globalRow] * scaleFactor);
        idx += BlockSize;
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
__global__ void nvf_u_and_sumsq_fused(const float* __restrict__ input, const float* __restrict__ w, float* __restrict__ u, float* __restrict__ globalSumSq, const int width, const int height) {
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
        atomicAdd(globalSumSq, blockTotalSq);
}

// main kernel for error sequence calculation
template <int p, bool FUSED>
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
    fillBlock<FUSED, p, shDimFast, shDimSlow>(inputA, inputB, &region[0][0], width, height);
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

// helper method used to reduce "rx" vector values and atomic add them (all threads cooperate)
template <int SIZE, typename StorageT>
__device__ __forceinline__ void writeRxVec(float* __restrict__ rx, const rxVecData<SIZE>& rxData, StorageT& temp_storage, float* __restrict__ warpStaging) {
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
        atomicAdd(&rx[i], warpStaging[i]);
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
__global__ void cholesky_solver(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ X, int* __restrict__ stopFlag) {
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

    // check if A, B, and X are 16-byte aligned for vectorized loads
    const bool isAligned = (((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B) | reinterpret_cast<uintptr_t>(X)) & 0xF) == 0);
    if (isAligned) {
        const float4* vecA = reinterpret_cast<const float4*>(A);
        const float4* vecB = reinterpret_cast<const float4*>(B);

        // load A (Rx)
        constexpr int vecLimitA = SIZE / 4;
#pragma unroll
        for (int k = 0; k < vecLimitA; k++) {
            const float4 v = vecA[k];
            packed[k * 4 + 0] = v.x;
            packed[k * 4 + 1] = v.y;
            packed[k * 4 + 2] = v.z;
            packed[k * 4 + 3] = v.w;
        }
        // tail elements of A (if SIZE is not multiple of 4)
        for (int k = vecLimitA << 2; k < SIZE; k++)
            packed[k] = A[k];

        // load B (rx)
        constexpr int vecLimitB = N / 4;
#pragma unroll
        for (int i = 0; i < vecLimitB; i++) {
            const float4 v = vecB[i];
            localB[i * 4 + 0] = v.x;
            localB[i * 4 + 1] = v.y;
            localB[i * 4 + 2] = v.z;
            localB[i * 4 + 3] = v.w;
        }
        // tail elements of B (if SIZE is not multiple of 4)
        for (int i = vecLimitB << 2; i < N; i++)
            localB[i] = B[i];
        // scalar path
    } else {
#pragma unroll
        for (int k = 0; k < SIZE; k++)
            packed[k] = A[k];
#pragma unroll
        for (int i = 0; i < N; i++)
            localB[i] = B[i];
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
__global__ void cholesky_solver_parallel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ X, int* __restrict__ stopFlag) {
    constexpr int N = (p * p) - 1;
    constexpr int packedSize = (N * (N + 1)) / 2;
    constexpr int vecPackedLimit = packedSize / 4;
    constexpr int vecBlimit = N / 4;

    const int laneId = threadIdx.x;

    __shared__ alignas(16) float sA[N][N + 1]; // +1 to avoid bank conflicts
    __shared__ alignas(16) float sB[N];

    // cooperative load (packedSize elements -> NxN Shared)
    // check if A, B, and X are 16-byte aligned for vectorized loads
    const bool isAligned = ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B) | reinterpret_cast<uintptr_t>(X)) & 0xF) == 0;

    // if aligned: packedSize floats = vecPackedLimit float4
    if (isAligned) {
        const float4* vecA = reinterpret_cast<const float4*>(A);
        // loop over vecPackedLimit vectors
        for (int k = laneId; k < vecPackedLimit; k += 32) {
            float4 v = vecA[k];
            // unpack vector into Shared Memory
            const int baseIdx = k * 4;
            const int2 c0 = getPackedCoords(baseIdx + 0);
            sA[c0.x][c0.y] = v.x;
            const int2 c1 = getPackedCoords(baseIdx + 1);
            sA[c1.x][c1.y] = v.y;
            const int2 c2 = getPackedCoords(baseIdx + 2);
            sA[c2.x][c2.y] = v.z;
            const int2 c3 = getPackedCoords(baseIdx + 3);
            sA[c3.x][c3.y] = v.w;
        }
        // scalar path
    } else {
        for (int k = laneId; k < packedSize; k += 32) {
            const int2 c = getPackedCoords(k);
            sA[c.x][c.y] = A[k];
        }
    }

    // if aligned: N floats = vecBlimit float4
    if (isAligned) {
        const float4* vecB = reinterpret_cast<const float4*>(B);
        if (laneId < vecBlimit) {
            const float4 v = vecB[laneId];
            sB[laneId * 4 + 0] = v.x;
            sB[laneId * 4 + 1] = v.y;
            sB[laneId * 4 + 2] = v.z;
            sB[laneId * 4 + 3] = v.w;
        }
        // scalar path
    } else {
        for (int k = laneId; k < N; k += 32)
            sB[k] = B[k];
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
__device__ void RxStreamPass(const int tid, float (*__restrict__ RxLocal)[92], float* __restrict__ Rx) {
    for (int k = startIdx + tid; k < endIdx; k += 128) {
        const int2 coords = getPackedCoords(k);
        const int rowInWindow = coords.x - rowOffset;
        float sum = 0.0f;
#pragma unroll
        for (int w = 0; w < 4; w++)
            sum += RxLocal[w * 32 + rowInWindow][coords.y];
        atomicAdd(&Rx[k], sum);
    }
}

// Prediction Error helper device kernels
__device__ void load_neighbor_row_funnel_p3(half& p0, half& p1, half& p2, const half* rowBase);
__device__ void load_neighbor_row_funnel_p3_col(half& a, half& b, half& c, const half* rowBase, const int col);
__device__ void load_neighbor_vec_p3_col(half8& dst, half& center, const half blockValues[3][258], const int col);
__device__ void load_neighbor_row_funnel_p5(half& p0, half& p1, half& p2, half& p3, half& p4, const half* rowBase);
__device__ void load_neighbor_row_funnel_p7(half* dst, const half* rowBase);
__device__ void load_neighbor_row_funnel_p9(half* dst, const half* rowBase);
__device__ void load_neighbor_vec_p5(half8* dst, const half blockValues[5][260], half& center);
__device__ void load_neighbor_vec_p7(half8* dst, const half blockValues[7][134], half& center);
__device__ void load_neighbor_vec_p9(half8* dst, const half blockValues[9][136], half& center);

// Prediction Error kernels (ME) for p = 3, p = 5, p = 7 and p = 9
__global__ void me_p3(const float* __restrict__ input, float* __restrict__ Rx, float* __restrict__ rx, const int width, const int height, const int totalBlocksY, const int taskTotal);
__global__ void me_p5(const float* __restrict__ input, float* __restrict__ Rx, float* __restrict__ rx, const int width, const int height, const int totalBlocksY, const int taskTotal);
__global__ void me_p7(const float* __restrict__ input, float* __restrict__ Rx, float* __restrict__ rx, const int width, const int height, const int totalBlocksY, const int taskTotal);
__global__ void me_p9(const float* __restrict__ input, float* __restrict__ Rx, float* __restrict__ rx, const int width, const int height, const int totalBlocksY, const int taskTotal);

// fused calculation of ME mask, u, and sum of squares
__global__ void me_u_and_sumsq_fused(const float* __restrict__ errorSeq, const float* __restrict__ w, float* __restrict__ u, float* __restrict__ globalSumSq, const float* __restrict__ maxVal,
                                     const int N);

// fused application of watermark: applies the watermark and calculates the output in one pass, using the precomputed u and sum of squares for normalization
__global__ void apply_watermark_fused(const float* __restrict__ input, const float* __restrict__ u, const float* __restrict__ sumSqPtr, uint8_t* __restrict__ output, const float strengthNumerator,
                                      const int planeElements, const int numChannels);

// main kernels for correlation calculation, used in detection.
__global__ void calculate_partial_correlation(const float* __restrict__ e_u, const float* __restrict__ e_z, float* __restrict__ partialDots, float* __restrict__ partialNormU,
                                              float* __restrict__ partialNormZ, const int size);
__global__ void calculate_final_correlation(const float* __restrict__ partialDots, const float* __restrict__ partialNormU, const float* __restrict__ partialNormZ, float* __restrict__ result,
                                            const int numBlocks);

// used for converting NV12 to YUV420p format, used in HW accelerated video decoding
__global__ void nV12ToYUV420p(const uint8_t* __restrict__ uvSrc, const int uvPitch, uint8_t* __restrict__ uvDst, const int uvWidth, const int uvHeight);

// used for converting uint8 pitched memory to non pitched float, used in HW accelerated video decoding
__global__ void pitchedToFloat(const uint8_t* __restrict__ input, float* __restrict__ output, const int width, const int height, const int pitch);