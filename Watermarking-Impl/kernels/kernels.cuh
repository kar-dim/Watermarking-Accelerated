#pragma once
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

struct alignas(16) half8
{
	half a, b, c, d, e, f, g, h;
};

__device__ half8 make_half8(const float& a, const float& b, const float& c, const float& d, const float& e, const float& f, const float& g, const float& h);
__device__ half8 make_half8(const half& a, const half& b, const half& c, const half& d, const half& e, const half& f, const half& g, const half& h);

//helper method to clamp a value between two limits
template<typename T>
__device__ __host__ inline T clamp(const T& val, const T& lo, const T& hi) { return (val < lo) ? lo : (val > hi) ? hi : val; }

//helper method to fill block-wide shared memory cooperatively for error sequence and NVF kernels
template<int p, int pad = p / 2, int sharedSize = 16 + (2 * pad)>
__device__ void fillBlock(const float* __restrict__ input, float* __restrict__ sharedMem, const int width, const int height)
{
    for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < sharedSize * sharedSize; i += blockDim.x * blockDim.y)
    {
        const int tileRow = i % sharedSize;
        const int tileCol = i / sharedSize;
        //clamp (mimic cudaAddressModeClamp)
        const int globalX = clamp<int>((int)(blockIdx.y * blockDim.y) + tileCol - pad, 0, width - 1);
        const int globalY = clamp<int>((int)(blockIdx.x * blockDim.x) + tileRow - pad, 0, height - 1);
        sharedMem[tileRow * sharedSize + tileCol] = input[globalX * height + globalY];
    }
}

//helper methods of ME kernel, to calculate block-wide rx values in shared memory
__device__ inline void me_p3_rxCalculate(half8* RxLocalVec8, const half8& vec, const half& x4);

//NVF kernel, calculates NVF values for each pixel in the image
//works for all p values (3,5,7 and 9)
template<int p, int pSquared = p * p, int pad = p / 2>
__global__ void nvf(const float* __restrict__ input, float* __restrict__ nvf, const unsigned int width, const unsigned int height)
{
    constexpr int sharedSize = 16 + (2 * pad);
    const int x = blockIdx.y * blockDim.y + threadIdx.y;
    const int y = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float region[sharedSize][sharedSize]; //hold the region for this 16 x 16 block

	fillBlock<p>(input, &region[0][0], width, height);
    __syncthreads();

    if (x >= width || y >= height)
        return;

    //local (shared memory) coordinates for center pixel
    const int shX = threadIdx.y + pad;
    const int shY = threadIdx.x + pad;

    float sum = 0.0f, sumSq = 0.0f;
    for (int i = -pad; i <= pad; i++)
    {
        for (int j = -pad; j <= pad; j++)
        {
            float val = region[shY + j][shX + i];
            sum += val;
            sumSq += val * val;
        }
    }
    const float mean = sum / pSquared;
    const float variance = (sumSq / pSquared) - (mean * mean);
    nvf[x * height + y] = fmaxf(variance / (1.0f + variance), 0.0f);
}

template <int N>
__device__ void choleskyInverse(float* A, float* b, float* x) {
    // 1. Cholesky Decomposition: A = L * L^T
    // A is symmetric positive definite. We work in-place on local registers/array.
    float L[N][N];

    // Clear L
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
                // Diagonal element
                float val = A[i * N + i] - sum;
                // Safety check for non-SPD (numerical noise)
                L[i][j] = (val > 0.0f) ? sqrtf(val) : 1e-6f;
            }
            else {
                // Off-diagonal
                L[i][j] = (1.0f / L[j][j] * (A[i * N + j] - sum));
            }
        }
    }

    // 2. Forward Substitution: Solve L * y = b
    float y[N];
#pragma unroll
    for (int i = 0; i < N; i++) {
        float sum = 0.0f;
        for (int k = 0; k < i; k++)
            sum += L[i][k] * y[k];
        y[i] = (b[i] - sum) / L[i][i];
    }

    // 3. Backward Substitution: Solve L^T * x = y
    // Note: L^T means we access L[row][col] swapped
#pragma unroll
    for (int i = N - 1; i >= 0; i--) {
        float sum = 0.0f;
        for (int k = i + 1; k < N; k++)
            sum += L[k][i] * x[k]; // Transposed access
        x[i] = (y[i] - sum) / L[i][i];
    }
}

__global__ void tiny_solver_kernel(const float* __restrict__ A_global, const float* __restrict__ b_global, float* __restrict__ x_global, int N);

//main ME kernel, calculates ME values for each pixel in the image
__global__ void me_p3(const float* __restrict__ input, float* __restrict__ Rx, float* __restrict__ rx, const unsigned int width, const unsigned int height);

//main kernel for error sequence calculation. used in ME kernel
__global__ void calculate_error_sequence_p3(const float* __restrict__ input, float* __restrict__ x_, const float* __restrict__ coeffs, const unsigned int width, const unsigned int height, const bool calculateAbs);

//main kernels for correlation calculation. used in detection.
__global__ void calculate_partial_correlation(const float* __restrict__ e_u, const float* __restrict__ e_z, float* __restrict__ partialDots, float* __restrict__ partialNormU, float* __restrict__ partialNormZ, const unsigned int size);
__global__ void calculate_final_correlation(const float* __restrict__ partialDots, const float* __restrict__ partialNormU, const float* __restrict__ partialNormZ, float* __restrict__ result, const unsigned int numBlocks);

//used for converting NV12 to YUV420p format, in HW accelerated video decoding
__global__ void nV12ToYUV420p(const uint8_t* __restrict__ uvSrc, const int uvPitch, uint8_t* __restrict__ uvDst, const int uvWidth, const int uvHeight);

//used for converting 8 pitched memory to float, in HW accelerated video decoding
__global__ void pitchedToFloat(const uint8_t* __restrict__ input, float* __restrict__ output, const int width, const int height, const int pitch);