#include "kernels.cuh"
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <mma.h>
using namespace nvcuda;

__device__ inline half HALF(float x) { return __float2half(x); }
__device__ inline float FLOAT(half x) { return __half2float(x); }

__constant__ float coeffs[8];

__host__ void setCoeffs(const float* c)
{
	cudaMemcpyToSymbol(coeffs, c, 8 * sizeof(float), 0, cudaMemcpyDeviceToDevice);
}

__device__ half8 make_half8(const float& a, const float& b, const float& c, const float& d, const float& e, const float& f, const float& g, const float& h)
{
    return half8 { HALF(a), HALF(b), HALF(c), HALF(d), HALF(e), HALF(f), HALF(g), HALF(h) };
}

__device__ half8 make_half8(const half& a, const half& b, const half& c, const half& d, const half& e, const half& f, const half& g, const half& h)
{
    return half8 { a, b, c, d, e, f, g, h };
}

//STS.128
__device__ void me_p3_rxCalculate(half8* RxLocalVec8, const half8& vec, const half& x4)
{
    half8 tmp;
    tmp.a = vec.a * x4;
    tmp.b = vec.b * x4;
    tmp.c = vec.c * x4;
    tmp.d = vec.d * x4;
    tmp.e = vec.e * x4;
    tmp.f = vec.f * x4;
    tmp.g = vec.g * x4;
    tmp.h = vec.h * x4;
    *RxLocalVec8 = tmp;
}

__global__ void me_p3(const float* __restrict__ input, float* __restrict__ Rx, float* __restrict__ rx, const unsigned int width, const unsigned int height)
{
	constexpr int sharedMemStride = 24; //16 + 8 padding for WMMA

    const int tid = threadIdx.x;
    const int x = blockIdx.x * 256 + tid;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int warpId = tid / 32;
    const int startRow = warpId * 32;
	const half halfScaleFactor = HALF(0.00392156862f); //multiplication with stored value of (1/255) is faster than division by 255

    //shared memory for Rx, rx, scratch and all pixels utilized by the whole block
    __shared__ alignas(16) half RxLocal[256][sharedMemStride];
    __shared__ half blockValues[3][258];

    //cooperatively load the 3 x (blockSize + 2) block (window size 3x3, for all threads in the block)
    for (int i = tid; i < 3 * 258; i += 256)
    {
        const int tileCol = i / 3;
        const int tileRow = i % 3;
        //clamp (mimic cudaAddressModeClamp)
        const int globalX = clamp<int>((int)(blockIdx.x * 256) + tileCol - 1, 0, width - 1);
        const int globalY = clamp<int>((int)(blockIdx.y * 1) + tileRow - 1, 0, height - 1);
        //normalize from [0,255] to [0,1] to support half precision and avoid overflow in multiplications
        blockValues[tileRow][tileCol] = HALF(input[globalX * height + globalY]) * halfScaleFactor;
    }
    __syncthreads();

    if (y >= height)
        return;

    //read the 3x3 window from shared memory
    const int localX = tid + 1; //center index
    const half center = blockValues[1][localX];
    half8 localBlock = make_half8(
        blockValues[0][localX - 1], blockValues[0][localX], blockValues[0][localX + 1],
        blockValues[1][localX - 1], blockValues[1][localX + 1],
        blockValues[2][localX - 1], blockValues[2][localX], blockValues[2][localX + 1]
    );

    
    half8* RxLocalVec8 = reinterpret_cast<half8*>(&RxLocal[tid][0]);
    RxLocalVec8[1] = make_half8(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
	//if/else is better than always writing (even vectorized) to shared memory unnecessarily
    if (x < width)
        RxLocalVec8[0] = localBlock;
    else 
    {
        RxLocalVec8[0] = make_half8(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
        localBlock = make_half8(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    }
	//compute rx, use half2 for faster reductions
    half8 rxVec;
    me_p3_rxCalculate(&rxVec, localBlock, center);
    half2* rxHalf2Ptr = reinterpret_cast<half2*>(&rxVec);
#pragma unroll
    for (int k = 0; k < 4; k++)
    {
        half2 val = rxHalf2Ptr[k];
        for (int offset = 16; offset > 0; offset /= 2)
            val = __hadd2(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
        rxHalf2Ptr[k] = val;
    }

    //exchange (each warp leader)
    if (tid % 32 == 0)
        RxLocalVec8[2] = rxVec;
    __syncthreads();

    //compute Rx with Tensor Cores
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> A;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> B;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> C;
    wmma::fill_fragment(C, 0.0f);
#pragma unroll
    for (int k0 = 0; k0 < 32; k0 += 16)
    {
        const half* tile_ptr = &RxLocal[startRow + k0][0];
        wmma::load_matrix_sync(A, tile_ptr, sharedMemStride);
        wmma::load_matrix_sync(B, tile_ptr, sharedMemStride);
        wmma::mma_sync(C, A, B, C);
    }

    //store matrix at warpId * 32 (row 0, 32, 64...)
    //so that we will overwrite only the first 16 rows of the warp's chunk
    //rows 16-31 of the chunk are NOT touched (preventing race conditions with neighbor warps)
    half* myWarpOutput = &RxLocal[warpId * 32][0];
    wmma::store_matrix_sync(myWarpOutput, C, sharedMemStride, wmma::mem_row_major);
    __syncthreads();

    //first 8 threads write rx
    if (tid < 8)
    {
        const int outputIndex = (y * gridDim.x * 8) + (blockIdx.x * 8);
        float sum_v = 0.0f;
#pragma unroll
        for (int w = 0; w < 8; w++)
            sum_v += FLOAT(RxLocal[w * 32][16 + tid]);
        rx[outputIndex + tid] = sum_v;
    }

	//first 64 threads write Rx
    if (tid < 64)
    {
        const int r = tid / 8;
        const int c = tid % 8;
        float sum = 0.0f;
#pragma unroll
        for (int w = 0; w < 8; w++)
            sum += FLOAT(RxLocal[w * 32 + r][c]);
        const int outputIndex = (y * gridDim.x * 64) + (blockIdx.x * 64);
        Rx[outputIndex + tid] = sum;
    }
}

__global__ void calculate_error_sequence_p3(const float* __restrict__ input, float* __restrict__ x_, const unsigned int width, const unsigned int height, const bool calculateAbs)
{
    constexpr int sharedSize = 16 + 2;
    const int y = blockIdx.x * blockDim.x + threadIdx.x;
    const int x = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ float region[sharedSize][sharedSize]; //hold the 18 x 18 region for this 16 x 16 block

    fillBlock<3>(input, &region[0][0], width, height);
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
        const float output = region[centerRow][centerCol] - dot;
        x_[(x * height + y)] = calculateAbs ? fabs(output) : output;
    }
}

__global__ void calculate_partial_correlation(const float* __restrict__ e_u, const float* __restrict__ e_z, float* __restrict__ partialDots, float* __restrict__ partialNormU, float* __restrict__ partialNormZ, const unsigned int size)
{
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + tid;
    const int warpId = tid / 32;

    //support for up to 1024/32 = 32 warps per block
    __shared__ float dotCache[32];
    __shared__ float normUCache[32];
    __shared__ float normZCache[32];

    float a = 0.0f, b = 0.0f;
    if (idx < size) 
    {
        a = e_u[idx];
        b = e_z[idx];
    }

    float dotVal = a * b;
    float normUVal = a * a;
    float normZVal = b * b;

    //intra-warp reduction
    for (int offset = 16; offset > 0; offset /= 2) 
    {
        dotVal += __shfl_down_sync(0xFFFFFFFF, dotVal, offset);
        normUVal += __shfl_down_sync(0xFFFFFFFF, normUVal, offset);
        normZVal += __shfl_down_sync(0xFFFFFFFF, normZVal, offset);
    }

    //warp leaders write to shared memory
    if ((tid & 31) == 0) 
    {
        dotCache[warpId] = dotVal;
        normUCache[warpId] = normUVal;
        normZCache[warpId] = normZVal;
    }
    __syncthreads();

    //final reduction by first warp
    if (tid < 32) 
    {
		const bool validTid = tid < (blockDim.x + warpSize - 1) / 32;
        dotVal = validTid ? dotCache[tid] : 0.0f;
        normUVal = validTid ? normUCache[tid] : 0.0f;
        normZVal = validTid ? normZCache[tid] : 0.0f;

        for (int offset = 16; offset > 0; offset /= 2) 
        {
            dotVal += __shfl_down_sync(0xFFFFFFFF, dotVal, offset);
            normUVal += __shfl_down_sync(0xFFFFFFFF, normUVal, offset);
            normZVal += __shfl_down_sync(0xFFFFFFFF, normZVal, offset);
        }

        if (tid == 0) 
        {
            partialDots[blockIdx.x] = dotVal;
            partialNormU[blockIdx.x] = normUVal;
            partialNormZ[blockIdx.x] = normZVal;
        }
    }
}

__global__ void calculate_final_correlation(const float* __restrict__ partialDots, const float* __restrict__ partialNormU, const float* __restrict__ partialNormZ, float* __restrict__ result, const unsigned int numBlocks)
{
    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int warpId = tid / 32;
    const int numWarps = (blockDim.x + 31) / 32;

    //shared memory must match number of warps
    __shared__ float warpDot[32];
    __shared__ float warpU[32];
    __shared__ float warpZ[32];

    float localDot = 0.0f;
    float localU = 0.0f;
    float localZ = 0.0f;

    for (int i = tid; i < numBlocks; i += blockDim.x) 
    {
        localDot += partialDots[i];
        localU += partialNormU[i];
        localZ += partialNormZ[i];
    }

    //intra-warp reduction
    for (int offset = 16; offset > 0; offset >>= 1) 
    {
        localDot += __shfl_down_sync(0xFFFFFFFF, localDot, offset);
        localU += __shfl_down_sync(0xFFFFFFFF, localU, offset);
        localZ += __shfl_down_sync(0xFFFFFFFF, localZ, offset);
    }
    if (lane == 0) 
    {
        warpDot[warpId] = localDot;
        warpU[warpId] = localU;
        warpZ[warpId] = localZ;
    }
    __syncthreads();

    //final warp reduces
    if (warpId == 0) 
    {
		const bool validTid = tid < numWarps;
        localDot = validTid ? warpDot[lane] : 0.0f;
        localU = validTid ? warpU[lane] : 0.0f;
        localZ = validTid ? warpZ[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) 
        {
            localDot += __shfl_down_sync(0xFFFFFFFF, localDot, offset);
            localU += __shfl_down_sync(0xFFFFFFFF, localU, offset);
            localZ += __shfl_down_sync(0xFFFFFFFF, localZ, offset);
        }
        if (lane == 0) 
        {
            float normU = sqrtf(localU);
            float normZ = sqrtf(localZ);
            result[0] = (normU > 0.0f && normZ > 0.0f) ? (localDot / (normU * normZ)) : 0.0f;
        }
    }
}

__global__ void nV12ToYUV420p(const uint8_t* __restrict__ uvSrc, const int uvPitch, uint8_t* __restrict__ uvDst, const int uvWidth, const int uvHeight)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= uvWidth * uvHeight)
        return;
    const int y = idx / uvWidth;
    const int x = idx % uvWidth;
    const uint8_t* src = uvSrc + y * uvPitch + 2 * x;
    uvDst[idx] = src[0];
    uvDst[uvWidth * uvHeight + idx] = src[1];
}

__global__ void pitchedToFloat(const uint8_t* __restrict__ input, float* __restrict__ output, const int width, const int height, const int pitch)
{
    __shared__ float block[16][16 + 1]; //+1 to avoid bank conflicts
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    float convertedValue = 0.0f;
    if (x < width && y < height) 
        convertedValue = static_cast<float>(input[y * pitch + x]);

    block[threadIdx.y][threadIdx.x] = convertedValue;
    __syncthreads();

    //write transposed data (coalesced writes to column-major output)
    const int dstX = blockIdx.y * blockDim.y + threadIdx.x;
    const int dstY = blockIdx.x * blockDim.x + threadIdx.y;
    if (dstX < height && dstY < width)
        output[dstY * height + dstX] = block[threadIdx.x][threadIdx.y];
}