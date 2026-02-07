#pragma once
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

struct alignas(16) half8 {
    half a, b, c, d, e, f, g, h;
};

// maps a linear index k to(row, col) coordinates for a packed lower triangular matrix
__device__ inline int2 getPackedCoords(const int k) {
    // inverse triangular number formula: r = floor((sqrt(1 + 8k) - 1) / 2)
    const int r = __float2int_rd(0.5f * (sqrtf(1.0f + 8.0f * k) - 1.0f));
    const int c = k - (r * (r + 1)) / 2;
    return make_int2(r, c);
}

// helper method to clamp a value between two limits
template <typename T> __device__ inline T clamp(const T& val, const T& lo, const T& hi) { return (val < lo) ? lo : (val > hi) ? hi : val; }

// helper method to fill block-wide shared memory cooperatively for error sequence and NVF kernels
// sharedMem must be of size: [sharedSize][sharedSize + 1] to avoid bank conflicts
template <bool FUSED, int p, int pad = p / 2, int sharedSize = 16 + (2 * pad)>
__device__ void fillBlockMain(const float* __restrict__ inputA, const float* __restrict__ inputB, float* __restrict__ sharedMem, const int width, const int height) {
    const int baseGlobalX = (int)(blockIdx.y * blockDim.y) - pad;
    const int baseGlobalY = (int)(blockIdx.x * blockDim.x) - pad;
    // cooperatively fill 2D shared memory
    for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < sharedSize * sharedSize; i += blockDim.x * blockDim.y) {
        const int tileRow = i % sharedSize;
        const int tileCol = i / sharedSize;
        const int globalX = clamp(baseGlobalX + tileCol, 0, width - 1);
        const int globalY = clamp(baseGlobalY + tileRow, 0, height - 1);
        const int idx = globalX * height + globalY;
        float val = inputA[idx];
        // if we need to fuse (A*B), do it here, branch-free because it its known at compile time
        if constexpr (FUSED)
            val *= inputB[idx];
        sharedMem[tileRow * (sharedSize + 1) + tileCol] = val;
    }
}

// non-fused version, one input only
template <int p> __device__ void fillBlock(const float* __restrict__ input, float* __restrict__ sharedMem, const int width, const int height) {
    fillBlockMain<false, p>(input, nullptr, sharedMem, width, height);
}

// fused version, two inputs multiplied together
template <int p> __device__ void fillBlock(const float* __restrict__ inputA, const float* __restrict__ inputB, float* __restrict__ sharedMem, const int width, const int height) {
    fillBlockMain<true, p>(inputA, inputB, sharedMem, width, height);
}

// helper method to fill block-wide shared memory cooperatively for prediction error kernels where
// the shared memory is a wide "strip" on the x-axis
template <int p, int stripWidth = 256 + p - 1, int radius = (p - 1) / 2>
__device__ inline void fillBlockStrip(half blockValues[p][stripWidth], const float* __restrict__ input, const int width, const int height) {
    constexpr float scaleFactor = 0.00392156862f;
    constexpr int totalPixels = p * stripWidth;
    constexpr int colStep = 256 / p;
    constexpr int rowStep = 256 % p;
    const int tid = threadIdx.x;
    const int baseGlobalCol = (int)(blockIdx.x * 256) - radius;
    const int baseGlobalRow = (int)(blockIdx.y * 1) - radius;
    int r = tid % p;
    int c = tid / p;
    int idx = tid;
    while (idx < totalPixels) {
        const int globalCol = clamp(baseGlobalCol + c, 0, width - 1);
        const int globalRow = clamp(baseGlobalRow + r, 0, height - 1);
        blockValues[r][c] = __float2half(input[globalCol * height + globalRow] * scaleFactor);
        idx += 256;
        c += colStep;
        r += rowStep;
        // if r exceeds p-1, we wrap it (and carry 1 to column)
        if (r >= p) {
            r -= p;
            c += 1;
        }
    }
}

// NVF kernel, calculates NVF values for each pixel in the image
// works for all p values (3,5,7 and 9)
template <int p, int pad = p / 2, int sharedSize = 16 + (2 * pad)> __global__ void nvf(const float* __restrict__ input, float* __restrict__ nvf, const unsigned int width, const unsigned int height) {
    constexpr float nPixels = static_cast<float>(p * p);
    constexpr float nPixelsSq = nPixels * nPixels;

    const int x = blockIdx.y * blockDim.y + threadIdx.y;
    const int y = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float region[sharedSize][sharedSize + 1]; //+1 for bank conflicts

    fillBlock<p>(input, &region[0][0], width, height);
    __syncthreads();

    if (x >= width || y >= height)
        return;

    // local (shared memory) coordinates for center pixel
    const int shX = threadIdx.y + pad;
    const int shY = threadIdx.x + pad;

    float sum = 0.0f, sumSq = 0.0f;
    for (int i = -pad; i <= pad; i++) {
        for (int j = -pad; j <= pad; j++) {
            const float pixelValue = region[shY + j][shX + i];
            sum += pixelValue;
            sumSq += pixelValue * pixelValue;
        }
    }
    // calculate NVF with optimized math (avoid divisions)
    const float numerator = (nPixels * sumSq) - (sum * sum);
    const float output = __fdividef(numerator, nPixelsSq + numerator);
    nvf[x * height + y] = clamp(output, 0.0f, 255.0f);
}

// helper method for error sequence calculation with p = 3
template <int p, int pad = p / 2, int sharedSize = 16 + (2 * pad), int stride = sharedSize + 1, int coeffsSize = (p * p) - 1>
__device__ inline float error_sequence_coeffs_filter(const float* __restrict__ region, const float* __restrict__ sCoeffs, const int localRow, const int localCol) {
    const int r = localRow + pad;
    const int c = localCol + pad;
    float dot = 0.0f;
    int k = 0;
#pragma unroll
    for (int i = -pad; i <= pad; i++) {
#pragma unroll
        for (int j = -pad; j <= pad; j++) {
            if (i == 0 && j == 0)
                continue;
            dot += sCoeffs[k] * region[(r + i) * stride + (c + j)];
            k++;
        }
    }
    return region[r * stride + c] - dot;
}

// main kernel for error sequence calculation
template <int p, bool FUSED, int pad = p / 2, int sharedSize = 16 + (2 * pad), int coeffsSize = (p * p) - 1>
__global__ void calculate_error_sequence(const float* __restrict__ inputA, const float* __restrict__ inputB, float* __restrict__ x_, const float* __restrict__ coeffs, const unsigned int width,
                                         const unsigned int height, const bool calculateAbs, const int* __restrict__ stopFlag) {
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;

    __shared__ float region[sharedSize][sharedSize + 1]; //+1 for bank conflicts
    __shared__ float sCoeffs[coeffsSize];
    if (tid < coeffsSize)
        sCoeffs[tid] = coeffs[tid];
    fillBlockMain<FUSED, p>(inputA, inputB, &region[0][0], width, height);
    __syncthreads();

    const int y = blockIdx.x * blockDim.x + threadIdx.x;
    const int x = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        if (*stopFlag) {
            x_[(x * height + y)] = 0.0f;
            return;
        }
        const float output = error_sequence_coeffs_filter<p>(&region[0][0], sCoeffs, threadIdx.x, threadIdx.y);
        x_[(x * height + y)] = calculateAbs ? fabsf(output) : output;
    }
}

// naive 1-thread Cholesky solver used for its very low latency versus cuSOLVER but useful only for very small systems, p = 3 (N = 8) or p = 5 (N = 24)
template <int p, int N = (p * p) - 1> __global__ void cholesky_solver(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ X, int* __restrict__ stopFlag) {
#define IDX(r, c) ((r * (r + 1)) / 2 + c)

    if (threadIdx.x > 0 || blockIdx.x > 0)
        return;

    // packed format: N*(N+1)/2 elements
    // p=3 (N=8) -> 36 floats
    // p=5 (N=24) -> 300 floats
    constexpr int SIZE = (N * (N + 1)) / 2;
    float packed[SIZE];
    float localB[N];

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
    if (isAligned) {
        float4* vecX = reinterpret_cast<float4*>(X);
        constexpr int vecLimitB = N / 4;
        // vectorized store
#pragma unroll
        for (int i = 0; i < vecLimitB; i++) {
            float4 v;
            v.x = localB[i * 4 + 0];
            v.y = localB[i * 4 + 1];
            v.z = localB[i * 4 + 2];
            v.w = localB[i * 4 + 3];
            vecX[i] = v;
        }
        // tail elements
        for (int i = vecLimitB * 4; i < N; i++)
            X[i] = localB[i];
        // scalar store
    } else {
#pragma unroll
        for (int i = 0; i < N; i++)
            X[i] = localB[i];
    }
#undef IDX
}

// parallel cholesky solver for p = 7 (N = 48) and p = 9 (N = 80), using one warp (32 threads)
template <int p, int N = (p * p) - 1> __global__ void cholesky_solver_parallel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ X, int* __restrict__ stopFlag) {
    const int laneId = threadIdx.x;

    // constants derived from N
    constexpr int packedSize = (N * (N + 1)) / 2;
    constexpr int vecPackedLimit = packedSize / 4;
    constexpr int vecBlimit = N / 4;

    // volatile to ensure visibility (because we use the faster __syncwarp)
    __shared__ volatile float sA[N][N + 1]; // +1 to avoid bank conflicts
    __shared__ volatile float sB[N];

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
    if (isAligned) {
        float4* vecX = reinterpret_cast<float4*>(X);
        const float4* sbVec = (const float4*)((float*)sB);
        if (laneId < vecBlimit)
            vecX[laneId] = sbVec[laneId];
    } else {
        for (int k = laneId; k < N; k += 32)
            X[k] = sB[k];
    }
}

// helper method to perform a streaming reduction of rx values from registers to global memory
// used ONLY in the ME kernels for p = 9 specifically
template <int startIdx, int count>
__device__ void rxStreamPass(const int tid, const int warpWindowStart, half (*__restrict__ RxLocal)[88], const half8* __restrict__ rxVec, float* __restrict__ rxGlobal, const int rxBaseIndex) {
    // warp leaders flush registers to shared memory window
    if ((tid & 31) == 0) {
        // process 8 elements per iteration (one half8 vector)
#pragma unroll
        for (int v = 0; v < count >> 3; v++) {
            const int absVecIdx = (startIdx >> 3) + v;
            half8* sharedVec = reinterpret_cast<half8*>(&RxLocal[warpWindowStart + v][80]);
            // STS.128
            *sharedVec = rxVec[absVecIdx];
        }
    }
    __syncthreads(); // wait for leaders to write
    for (int k = startIdx + tid; k < startIdx + count; k += 128) {
        float sum = 0.0f;
        const int localK = k - startIdx; // map global K to window index [0,31]
        const int rowInWarp = localK >> 3;
        const int colInRow = 80 + (localK & 7);
        // sum across the 4 warps
#pragma unroll
        for (int w = 0; w < 4; w++)
            sum += __half2float(RxLocal[w * 32 + rowInWarp][colInRow]);
        rxGlobal[rxBaseIndex + k] = sum;
    }
    __syncthreads(); // wait for readers before next pass overwrites the window
}

// helper method to perform a streaming reduction of Rx values from shared window to global memory
// Templates allow the compiler to hardcode constants for each specific chunk
template <int startIdx, int endIdx, int rowOffset> __device__ void RxStreamPass(const int tid, half (*__restrict__ RxLocal)[88], float* __restrict__ Rx, const int RxBaseIndex) {
    for (int k = startIdx + tid; k < endIdx; k += 128) {
        const int2 coords = getPackedCoords(k);
        const int rowInWindow = coords.x - rowOffset;
        float sum = 0.0f;
        // sum across the 4 warps (stacked vertically in shared mem at offsets 0, 32, 64, 96)
#pragma unroll
        for (int w = 0; w < 4; w++)
            sum += __half2float(RxLocal[w * 32 + rowInWindow][coords.y]);
        Rx[RxBaseIndex + k] = sum;
    }
}

// Prediction Error helper device kernels
__device__ void me_p3_rxCalculate(half8* RxLocalVec8, const half8& vec, const half& center);
__device__ void load_neighbor_row_funnel_p3(half& p0, half& p1, half& p2, const half* rowBase);
__device__ void load_neighbor_row_funnel_p5(half& p0, half& p1, half& p2, half& p3, half& p4, const half* rowBase);
__device__ void load_neighbor_row_funnel_p7(half* dst, const half* rowBase);
__device__ void load_neighbor_row_funnel_p9(half* dst, const half* rowBase);
__device__ void load_neighbor_vec_p5(half8* dst, const half blockValues[5][260], half& center);
__device__ void load_neighbor_vec_p7(half8* dst, const half blockValues[7][262], half& center);
__device__ void load_neighbor_vec_p9(half8* dst, const half blockValues[9][136], half& center);

// Prediction Error kernels (ME) for p = 3, p = 5, p = 7 and p = 9
__global__ void me_p3(const float* __restrict__ input, float* __restrict__ Rx, float* __restrict__ rx, const unsigned int width, const unsigned int height);
__global__ void me_p5(const float* __restrict__ input, float* __restrict__ Rx, float* __restrict__ rx, const unsigned int width, const unsigned int height);
__global__ void me_p7(const float* __restrict__ input, float* __restrict__ Rx, float* __restrict__ rx, const unsigned int width, const unsigned int height);
__global__ void me_p9(const float* __restrict__ input, float* __restrict__ Rx, float* __restrict__ rx, const unsigned int width, const unsigned int height);

// fused calculation of u and sum of squares
__global__ void compute_u_and_sumsq(const float* __restrict__ mask, const float* __restrict__ w, float* __restrict__ u, float* __restrict__ globalSumSq, const int N);

// fused application of watermark: applies the watermark and calculates the output in one pass, using the precomputed u and sum of squares for normalization
__global__ void apply_watermark_fused(const float* __restrict__ input, const float* __restrict__ u, const float* __restrict__ sumSqPtr, uint8_t* __restrict__ output, const float strengthNumerator,
                                      const int planeElements, const int numChannels);

// main kernels for correlation calculation, used in detection.
__global__ void calculate_partial_correlation(const float* __restrict__ e_u, const float* __restrict__ e_z, float* __restrict__ partialDots, float* __restrict__ partialNormU,
                                              float* __restrict__ partialNormZ, const unsigned int size);
__global__ void calculate_final_correlation(const float* __restrict__ partialDots, const float* __restrict__ partialNormU, const float* __restrict__ partialNormZ, float* __restrict__ result,
                                            const unsigned int numBlocks);

// used for converting NV12 to YUV420p format, used in HW accelerated video decoding
__global__ void nV12ToYUV420p(const uint8_t* __restrict__ uvSrc, const int uvPitch, uint8_t* __restrict__ uvDst, const int uvWidth, const int uvHeight);

// used for converting uint8 pitched memory to non pitched float, used in HW accelerated video decoding
__global__ void pitchedToFloat(const uint8_t* __restrict__ input, float* __restrict__ output, const int width, const int height, const int pitch);