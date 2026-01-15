#pragma once
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

struct alignas(16) half8 {
    half a, b, c, d, e, f, g, h;
};

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
        const int globalX = clamp<int>(baseGlobalX + tileCol, 0, width - 1);
        const int globalY = clamp<int>(baseGlobalY + tileRow, 0, height - 1);
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
template <int p> __device__ inline void fillBlockStrip(half blockValues[p][256 + p - 1], const float* __restrict__ input, const int width, const int height) {
    const half halfScaleFactor = __float2half(0.00392156862f);
    // parallel indexing, fastest for p=3
    if constexpr (p == 3) {
#pragma unroll
        for (int i = threadIdx.x; i < 3 * 258; i += 256) {
            const int tileCol = i / 3;
            const int tileRow = i % 3;
            // clamp (mimic cudaAddressModeClamp)
            const int globalX = clamp<int>((int)(blockIdx.x * 256) + tileCol - 1, 0, width - 1);
            const int globalY = clamp<int>((int)(blockIdx.y * 1) + tileRow - 1, 0, height - 1);
            // normalize from [0,255] to [0,1] to support half precision and avoid overflow in multiplications
            blockValues[tileRow][tileCol] = __float2half(input[globalX * height + globalY]) * halfScaleFactor;
        }
    }
    // Incremental update, fastest for p=5
    else {
        constexpr int radius = (p - 1) / 2;
        constexpr int stride = 256 / p;
        constexpr int remainderLimit = p * (p - 1);
        const int blockGlobalX = (int)(blockIdx.x * 256) - radius;
        const int blockGlobalY = (int)(blockIdx.y * 1) - radius;
        int tileRow = threadIdx.x % p;
        int tileCol = threadIdx.x / p;
#pragma unroll
        for (int k = 0; k < p; k++) {
            // clamp (mimic cudaAddressModeClamp)
            const int globalX = clamp<int>(blockGlobalX + tileCol, 0, width - 1);
            const int globalY = clamp<int>(blockGlobalY + tileRow, 0, height - 1);
            // normalize from [0,255] to [0,1] to support half precision and avoid overflow in multiplications
            blockValues[tileRow][tileCol] = __float2half(input[globalX * height + globalY]) * halfScaleFactor;
            tileRow++;
            const int wrap = (tileRow >= p);
            tileRow -= (wrap * p);
            tileCol += stride + wrap;
        }
        if (threadIdx.x < remainderLimit) {
            const int globalX = clamp<int>(blockGlobalX + tileCol, 0, width - 1);
            const int globalY = clamp<int>(blockGlobalY + tileRow, 0, height - 1);
            blockValues[tileRow][tileCol] = __float2half(input[globalX * height + globalY]) * halfScaleFactor;
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
    nvf[x * height + y] = fmaxf(output, 0.0f);
}

// helper method for error sequence calculation with p = 3
template <int p, int pad = p / 2, int sharedSize = 16 + (2 * pad), int coeffsSize = (p * p) - 1>
__device__ inline float error_sequence_coeffs_filter(const float region[sharedSize][sharedSize], const float sCoeffs[coeffsSize], const int localRow, const int localCol) {
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
            dot += sCoeffs[k] * region[r + i][c + j];
            k++;
        }
    }
    return region[r][c] - dot;
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
        const float output = error_sequence_coeffs_filter<p>(region, sCoeffs, threadIdx.x, threadIdx.y);
        x_[(x * height + y)] = calculateAbs ? fabsf(output) : output;
    }
}

// naive 1-thread Cholesky solver used for its very low latency versus cuSOLVER but useful only for very small systems, p = 3 (N = 8) or p = 5 (N = 24)
template <int p, int N = (p * p) - 1> __global__ void cholesky_solver(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ X, int* __restrict__ stopFlag) {
    if (threadIdx.x > 0 || blockIdx.x > 0)
        return;

    float localA[N * N], localB[N], localX[N];

    // initialize result to 0.0f (safe fallback for unsolvable systems)
    for (int i = 0; i < N; i++)
        localX[i] = 0.0f;

    // initialize Rx and rx
    // because we use 1 thread, it is faster to reconstruct Rx manually than to use the constant cache map!
    int k = 0;
#pragma unroll
    for (int r = 0; r < N; r++) {
#pragma unroll
        for (int c = 0; c <= r; c++) {
            float val = A[k++];
            localA[r * N + c] = val;
            localA[c * N + r] = val;
        }
    }
    // copy rx
    for (int i = 0; i < N; i++)
        localB[i] = B[i];

    // Cholesky Decomposition: A = L*L^T
    // A is symmetric positive definite
    float L[N][N];
    // clear L
#pragma unroll
    for (int i = 0; i < N; i++)
#pragma unroll
        for (int j = 0; j < N; j++)
            L[i][j] = 0.0f;
#pragma unroll
    for (int i = 0; i < N; i++) {
#pragma unroll
        for (int j = 0; j <= i; j++) {
            float sum = 0.0f;
            for (int k = 0; k < j; k++)
                sum += L[i][k] * L[j][k];
            if (i == j) {
                // diagonal element
                const float val = localA[i * N + i] - sum;
                // check if singular! if so, exit early with X = 0.0f
                if (val <= 1e-12f) {
                    *stopFlag = 1;
                    goto exit;
                }
                L[i][j] = sqrtf(val);
            } else                                                        // non diagonal
                L[i][j] = (localA[i * N + j] - sum) * __frcp_rn(L[j][j]); // fast reciprocal
        }
    }
    // solve the system with forward and backward substitution
    // we again use fast reciprocal for better performance (1 GPU thread is weak, needs as fast math as possible)
    // forward substitution -> solve L*y = b
    float y[N];
#pragma unroll
    for (int i = 0; i < N; i++) {
        float sum = 0.0f;
        for (int k = 0; k < i; k++)
            sum += L[i][k] * y[k];
        y[i] = (localB[i] - sum) * __frcp_rn(L[i][i]);
    }

    // backward substitution -> solve L^T * x = y
#pragma unroll
    for (int i = N - 1; i >= 0; i--) {
        float sum = 0.0f;
        for (int k = i + 1; k < N; k++)
            sum += L[k][i] * localX[k]; // transposed
        localX[i] = (y[i] - sum) * __frcp_rn(L[i][i]);
    }
    *stopFlag = 0;
    // write
exit:
    for (int i = 0; i < N; i++)
        X[i] = localX[i];
}

// Prediction Error helper device kernels
__device__ void me_p3_rxCalculate(half8* RxLocalVec8, const half8& vec, const half& center);
__device__ void load_neighbor_row_funnel_p3(half& p0, half& p1, half& p2, const half* rowBase);
__device__ void load_neighbor_row_funnel_p5(half& p0, half& p1, half& p2, half& p3, half& p4, const half* rowBase);
__device__ void load_neighbor_vec_p5(half8* dst, const half blockValues[5][260], half& center);
// Prediction Error kernels (ME) for p = 3 and p = 5
__global__ void me_p3(const float* __restrict__ input, float* __restrict__ Rx, float* __restrict__ rx, const unsigned int width, const unsigned int height);
__global__ void me_p5(const float* __restrict__ input, float* __restrict__ Rx, float* __restrict__ rx, const unsigned int width, const unsigned int height);

// main kernels for correlation calculation. used in detection.
__global__ void calculate_partial_correlation(const float* __restrict__ e_u, const float* __restrict__ e_z, float* __restrict__ partialDots, float* __restrict__ partialNormU,
                                              float* __restrict__ partialNormZ, const unsigned int size);
__global__ void calculate_final_correlation(const float* __restrict__ partialDots, const float* __restrict__ partialNormU, const float* __restrict__ partialNormZ, float* __restrict__ result,
                                            const unsigned int numBlocks);

// used for converting NV12 to YUV420p format, in HW accelerated video decoding
__global__ void nV12ToYUV420p(const uint8_t* __restrict__ uvSrc, const int uvPitch, uint8_t* __restrict__ uvDst, const int uvWidth, const int uvHeight);

// used for converting 8 pitched memory to float, in HW accelerated video decoding
__global__ void pitchedToFloat(const uint8_t* __restrict__ input, float* __restrict__ output, const int width, const int height, const int pitch);