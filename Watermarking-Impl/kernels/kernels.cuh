#pragma once
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

struct alignas(16) half8
{
    half a, b, c, d, e, f, g, h;
};

//helper method to clamp a value between two limits
template<typename T>
__device__ __host__ inline T clamp(const T& val, const T& lo, const T& hi) { return (val < lo) ? lo : (val > hi) ? hi : val; }

//helper method to fill block-wide shared memory cooperatively for error sequence and NVF kernels
template<bool FUSED, int p, int pad = p / 2, int sharedSize = 16 + (2 * pad)>
__device__ void fillBlockMain(const float* __restrict__ inputA, const float* __restrict__ inputB, float* __restrict__ sharedMem, const int width, const int height)
{
    //cooperatively fill 2D shared memory
    for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < sharedSize * sharedSize; i += blockDim.x * blockDim.y)
    {
        const int tileRow = i % sharedSize;
        const int tileCol = i / sharedSize;
        const int globalX = clamp<int>((int)(blockIdx.y * blockDim.y) + tileCol - pad, 0, width - 1);
        const int globalY = clamp<int>((int)(blockIdx.x * blockDim.x) + tileRow - pad, 0, height - 1);
        const int idx = globalX * height + globalY;
        float val = inputA[idx];
        //if we need to fuse (A*B), do it here, branch-free because it its known at compile time
        if constexpr (FUSED)
            val *= inputB[idx];
        sharedMem[tileRow * sharedSize + tileCol] = val;
    }
}

//non-fused version, one input only
template<int p, int pad = p / 2, int sharedSize = 16 + (2 * pad)>
__device__ void fillBlock(const float* __restrict__ input, float* __restrict__ sharedMem, const int width, const int height)
{
    fillBlockMain<false, p, pad, sharedSize>(input, nullptr, sharedMem, width, height);
}

//fused version, two inputs multiplied together
template<int p, int pad = p / 2, int sharedSize = 16 + (2 * pad)>
__device__ void fillBlock(const float* __restrict__ inputA, const float* __restrict__ inputB, float* __restrict__ sharedMem, const int width, const int height)
{
    fillBlockMain<true, p, pad, sharedSize>(inputA, inputB, sharedMem, width, height);
}

//NVF kernel, calculates NVF values for each pixel in the image
//works for all p values (3,5,7 and 9)
template<int p, int pad = p / 2, int sharedSize = 16 + (2 * pad)>
__global__ void nvf(const float* __restrict__ input, float* __restrict__ nvf, const unsigned int width, const unsigned int height)
{
    constexpr float nPixels = static_cast<float>(p * p);
    constexpr float nPixelsSq = nPixels * nPixels;

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
            const float pixelValue = region[shY + j][shX + i];
            sum += pixelValue;
            sumSq += pixelValue * pixelValue;
        }
    }
    //calculate NVF with optimized math (avoid divisions)
    const float numerator = (nPixels * sumSq) - (sum * sum);
    const float output = __fdividef(numerator, nPixelsSq + numerator);
    nvf[x * height + y] = fmaxf(output, 0.0f);
}

//helper method for error sequence calculation with p = 3
template<int p, int pad = p / 2, int sharedSize = 16 + (2 * pad), int coeffsSize = (p* p) - 1>
__device__ inline float error_sequence_coeffs_filter(const float region[sharedSize][sharedSize], const float sCoeffs[coeffsSize], const int localRow, const int localCol)
{
    const int r = localRow + pad;
    const int c = localCol + pad;
    float dot = 0.0f;
    int k = 0;
#pragma unroll
    for (int i = -pad; i <= pad; i++)
    {
#pragma unroll
        for (int j = -pad; j <= pad; j++)
        {
            if (i == 0 && j == 0)
                continue;
            dot += sCoeffs[k] * region[r + i][c + j];
            k++;
        }
    }
    return region[r][c] - dot;
}

//main kernel for error sequence calculation
template<int p, bool FUSED, int pad = p / 2, int sharedSize = 16 + (2 * pad), int coeffsSize = (p* p) - 1>
__global__ void calculate_error_sequence(const float* __restrict__ inputA, const float* __restrict__ inputB, float* __restrict__ x_, const float* __restrict__ coeffs, const unsigned int width, const unsigned int height, const bool calculateAbs, const int* __restrict__ stopFlag)
{
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;

    __shared__ float region[sharedSize][sharedSize];
    __shared__ float sCoeffs[coeffsSize];
    if (tid < coeffsSize)
        sCoeffs[tid] = coeffs[tid];
    fillBlockMain<FUSED, p>(inputA, inputB, &region[0][0], width, height);
    __syncthreads();

    const int y = blockIdx.x * blockDim.x + threadIdx.x;
    const int x = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height)
    {
        if (*stopFlag)
        {
            x_[(x * height + y)] = 0.0f;
            return;
        }
        const float output = error_sequence_coeffs_filter<p>(region, sCoeffs, threadIdx.x, threadIdx.y);
        x_[(x * height + y)] = calculateAbs ? fabsf(output) : output;
    }
}

//main Cholesky solver kernel for p = 3 (8x8 system), faster than af::solve for small systems (no cuSOLVE overhead), no LU pivoting and most importantly:
//we can stop early by passing a GPU flag if the system is not solvable, without returning to host (used in both ME and detection)
__global__ void cholesky_solver_p3(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ X, int* __restrict__ stopFlag);

//helper methods of ME kernel, to calculate block-wide rx values in shared memory
__device__ inline void me_p3_rxCalculate(half8* RxLocalVec8, const half8& vec, const half& x4);

//main Prediction error kernel and the most heavy: computes prediction error matrices (Rx, rx) for each pixel of the image
__global__ void me_p3(const float* __restrict__ input, float* __restrict__ Rx, float* __restrict__ rx, const unsigned int width, const unsigned int height);

//main kernels for correlation calculation. used in detection.
__global__ void calculate_partial_correlation(const float* __restrict__ e_u, const float* __restrict__ e_z, float* __restrict__ partialDots, float* __restrict__ partialNormU, float* __restrict__ partialNormZ, const unsigned int size);
__global__ void calculate_final_correlation(const float* __restrict__ partialDots, const float* __restrict__ partialNormU, const float* __restrict__ partialNormZ, float* __restrict__ result, const unsigned int numBlocks);

//used for converting NV12 to YUV420p format, in HW accelerated video decoding
__global__ void nV12ToYUV420p(const uint8_t* __restrict__ uvSrc, const int uvPitch, uint8_t* __restrict__ uvDst, const int uvWidth, const int uvHeight);

//used for converting 8 pitched memory to float, in HW accelerated video decoding
__global__ void pitchedToFloat(const uint8_t* __restrict__ input, float* __restrict__ output, const int width, const int height, const int pitch);