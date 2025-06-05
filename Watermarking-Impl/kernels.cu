#include "kernels.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>

#define HALF(x) __float2half(x)
#define FLOAT(x) __half2float(x)

__constant__ float coeffs[8];

__host__ void setCoeffs(const float* c)
{
	cudaMemcpyToSymbol(coeffs, c, 8 * sizeof(float), 0, cudaMemcpyDeviceToDevice);
}

//constant array used for optimizing share memory accesses for Rx
//Helps with reducing the local memory required for each block for Rx arrays from 4096 to 2304
__constant__ int RxMappings[64] =
{
    0,  1,  2,  3,  4,  5,  6,  7,
    1,  8,  9,  10, 11, 12, 13, 14,
    2,  9,  15, 16, 17, 18, 19, 20,
    3,  10, 16, 21, 22, 23, 24, 25,
    4,  11, 17, 22, 26, 27, 28, 29,
    5,  12, 18, 23, 27, 30, 31, 32,
    6,  13, 19, 24, 28, 31, 33, 34,
    7,  14, 20, 25, 29, 32, 34, 35
};

__device__ half8 make_half8(const float& a, const float& b, const float& c, const float& d, const float& e, const float& f, const float& g, const float& h)
{
    return half8 { HALF(a), HALF(b), HALF(c), HALF(d), HALF(e), HALF(f), HALF(g), HALF(h) };
}

__device__ half8 make_half8(const half& a, const half& b, const half& c, const half& d, const half& e, const half& f, const half& g, const half& h)
{
    return half8 { a, b, c, d, e, f, g, h };
}

__device__ void me_p3_rxCalculate(half8* RxLocalVec8, const half& x_0, const half& x_1, const half& x_2, const half& x_3, const half& x_4, const half& x_5, const half& x_6, const half& x_7, const half& x_8)
{
    *RxLocalVec8 = make_half8(x_0 * x_4, x_1 * x_4, x_2 * x_4, x_3 * x_4, x_5 * x_4, x_6 * x_4, x_7 * x_4, x_8 * x_4);
}

__device__ void me_p3_RxCalculate(half8* RxLocalVec8, const half& x_0, const half& x_1, const half& x_2, const half& x_3, const half& x_5, const half& x_6, const half& x_7, const half& x_8)
{
    const half zero = HALF(0.0f);
    RxLocalVec8[0] = make_half8(x_0 * x_0, x_0 * x_1, x_0 * x_2, x_0 * x_3, x_0 * x_5, x_0 * x_6, x_0 * x_7, x_0 * x_8);
    RxLocalVec8[1] = make_half8(x_1 * x_1, x_1 * x_2, x_1 * x_3, x_1 * x_5, x_1 * x_6, x_1 * x_7, x_1 * x_8, x_2 * x_2);
    RxLocalVec8[2] = make_half8(x_2 * x_3, x_2 * x_5, x_2 * x_6, x_2 * x_7, x_2 * x_8, x_3 * x_3, x_3 * x_5, x_3 * x_6);
    RxLocalVec8[3] = make_half8(x_3 * x_7, x_3 * x_8, x_5 * x_5, x_5 * x_6, x_5 * x_7, x_5 * x_8, x_6 * x_6, x_6 * x_7);
    RxLocalVec8[4] = make_half8(x_6 * x_8, x_7 * x_7, x_7 * x_8, x_8 * x_8, zero, zero, zero, zero);
}

__global__ void me_p3(cudaTextureObject_t texObj, float* __restrict__ Rx, float* __restrict__ rx, const unsigned int width, const unsigned int paddedWidth, const unsigned int height)
{
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int outputIndex = (y * paddedWidth) + x;

    //re-use shared memory for Rx and rx calculation, helps with occupancy
    __shared__ half RxLocal[64][40]; //36 + 4 for 16-byte alignment (in order to use vectorized 128-bit load/store)
    half8* RxLocalVec8 = reinterpret_cast<half8*>(RxLocal[threadIdx.x]);

    //initialize shared memory, assign a portion for all threads for parallelism
    #pragma unroll
    for (int i = 0; i < 5; i++)
        RxLocalVec8[i] = make_half8(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);

    if (y >= height)
        return;

    //load 3x3 window
    half x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8;
    if (x < width)
    {
        x_0 = HALF(tex2D<float>(texObj, y - 1, x - 1));
        x_1 = HALF(tex2D<float>(texObj, y - 1, x));
        x_2 = HALF(tex2D<float>(texObj, y - 1, x + 1));
        x_3 = HALF(tex2D<float>(texObj, y, x - 1));
        x_4 = HALF(tex2D<float>(texObj, y, x)); //x_4 is central pixel
        x_5 = HALF(tex2D<float>(texObj, y, x + 1));
        x_6 = HALF(tex2D<float>(texObj, y + 1, x - 1));
        x_7 = HALF(tex2D<float>(texObj, y + 1, x));
        x_8 = HALF(tex2D<float>(texObj, y + 1, x + 1));
        //calculate this thread's 8 rx values
        me_p3_rxCalculate(RxLocalVec8, x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
    }
    __syncthreads();

    //optimized summation for rx with warp shuffling
    float sum = 0;
    const int row = threadIdx.x / 8;
    #pragma unroll
    for (int i = 0; i < 64; i += 8)
        sum += FLOAT(RxLocal[(threadIdx.x + i) % 64][row]);
    // reduce 32 results to 4 per warp
    for (int i = 4; i > 0; i = i / 2)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, i);
    if (threadIdx.x % 8 == 0)
        rx[(outputIndex + row) / 8] = sum;
    __syncthreads();

    //calculate 36 Rx values
    if (x < width)
        me_p3_RxCalculate(RxLocalVec8, x_0, x_1, x_2, x_3, x_5, x_6, x_7, x_8);
    __syncthreads();

    //simplified summation for Rx
    //we cannot use warp shuffling because it introduces too much stalling for Rx
    sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < 64; i++)
        sum += FLOAT(RxLocal[i][RxMappings[threadIdx.x]]);
    Rx[outputIndex] = sum;
}

__global__ void calculate_scaled_neighbors_p3(cudaTextureObject_t texObj, float* x_, const unsigned int width, const unsigned int height)
{
    constexpr int sharedSize = 16 + 2;
    const int x = blockIdx.y * blockDim.y + threadIdx.y;
    const int y = blockIdx.x * blockDim.x + threadIdx.x;
    const int localId = threadIdx.y * blockDim.x + threadIdx.x; // 0 to 255 for 16 x 16 block

    __shared__ float region[sharedSize][sharedSize]; //hold the 18 x 18 region for this 16 x 16 block

    //load cooperatively the 18 x 18 region for this 16 x 16 block
    for (int i = localId; i < 18 * 18; i += blockDim.x * blockDim.y)
    {
        const int tileRow = i / sharedSize;
        const int tileCol = i % sharedSize;
        const int globalX = blockIdx.y * blockDim.y + tileCol - 1;
        const int globalY = blockIdx.x * blockDim.x + tileRow - 1;
        region[tileRow][tileCol] = tex2D<float>(texObj, globalY, globalX);
    }
    __syncthreads();

    //calculate the dot product of the coefficients and the neighborhood for this pixel
    if (x < width && y < height)
    {
        const int centerCol = threadIdx.y + 1;
        const int centerRow = threadIdx.x + 1;
        float dot = 0.0f;
        dot += coeffs[0] * region[centerRow - 1][centerCol - 1];
        dot += coeffs[1] * region[centerRow - 1][centerCol];
        dot += coeffs[2] * region[centerRow - 1][centerCol + 1];
        dot += coeffs[3] * region[centerRow][centerCol - 1];
        dot += coeffs[4] * region[centerRow][centerCol + 1];
        dot += coeffs[5] * region[centerRow + 1][centerCol - 1];
        dot += coeffs[6] * region[centerRow + 1][centerCol];
        dot += coeffs[7] * region[centerRow + 1][centerCol + 1];
        x_[(x * height + y)] = dot;
    }
}