#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

struct alignas(16) half8
{
	half a, b, c, d, e, f, g, h;
};
__host__ void setCoeffs(const float* c);

__device__ half8 make_half8(const float& a, const float& b, const float& c, const float& d, const float& e, const float& f, const float& g, const float& h);
__device__ half8 make_half8(const half& a, const half& b, const half& c, const half& d, const half& e, const half& f, const half& g, const half& h);

//helper methods of ME kernel, to calculate block-wide Rx/rx values in shared memory
__device__ void me_p3_rxCalculate(half8* RxLocalVec, const half& x_0, const half& x_1, const half& x_2, const half& x_3, const half& x_4, const half& x_5, const half& x_6, const half& x_7, const half& x_8);
__device__ void me_p3_RxCalculate(half8* RxLocalVec, const half& x_0, const half& x_1, const half& x_2, const half& x_3, const half& x_5, const half& x_6, const half& x_7, const half& x_8);

// NVF kernel, calculates NVF values for each pixel in the image
// works for all p values (3,5,7 and 9)
template<int p, int pSquared = p * p, int pad = p / 2>
__global__ void nvf(cudaTextureObject_t texObj, float* nvf, const unsigned int width, const unsigned int height)
{
    constexpr int sharedSize = 16 + (2 * pad);
    const int x = blockIdx.y * blockDim.y + threadIdx.y;
    const int y = blockIdx.x * blockDim.x + threadIdx.x;
    const int localId = threadIdx.y * blockDim.x + threadIdx.x; // 0 to 255 for 16 x 16 block

    __shared__ float region[sharedSize][sharedSize]; //hold the region for this 16 x 16 block

    //load cooperatively the padded region for this 16 x 16 block
    for (int i = localId; i < sharedSize * sharedSize; i += blockDim.x * blockDim.y)
    {
        const int tileRow = i / sharedSize;
        const int tileCol = i % sharedSize;
        const int globalX = blockIdx.y * blockDim.y + tileCol - pad;
        const int globalY = blockIdx.x * blockDim.x + tileRow - pad;
        region[tileRow][tileCol] = tex2D<float>(texObj, globalY, globalX);
    }
    __syncthreads();

    // Bounds check
    if (x >= width || y >= height)
        return;

    // Local (shared memory) coordinates for center pixel
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

    float mean = sum / pSquared;
    float variance = (sumSq / pSquared) - (mean * mean);

    // Store result
    nvf[x * height + y] = variance / (1.0f + variance);
}

//main ME kernel, calculates ME values for each pixel in the image
__global__ void me_p3(cudaTextureObject_t texObj, float* __restrict__ Rx, float* __restrict__ rx, const unsigned int width, const unsigned int paddedWidth, const unsigned int height);

//main kernel for scaled neighbors calculation. used in ME kernel
__global__ void calculate_scaled_neighbors_p3(cudaTextureObject_t texObj, float* x_, const unsigned int width, const unsigned int height);