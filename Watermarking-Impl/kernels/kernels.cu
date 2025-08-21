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

__global__ void me_p3(const float* __restrict__ input, float* __restrict__ Rx, float* __restrict__ rx, const unsigned int width, const unsigned int paddedWidth, const unsigned int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int outputIndex = (y * paddedWidth) + x;

	//Shared memory for Rx and rx, helper accumulator for Rx WMMA plus block-wide window pixels 
    __shared__ alignas(16) half RxLocal[64][16];
    __shared__ float RxWmmaAccum[16][16];
    __shared__ half blockValues[3][66];
    half8* RxLocalVec8 = reinterpret_cast<half8*>(RxLocal[threadIdx.x]);

    //initialize Rx shared memory
#pragma unroll
    for (int i = 0; i < 2; i++)
        RxLocalVec8[i] = make_half8(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);

    if (y >= height)
        return;

	// cooperatively load the 3 x 66 block (window size 3x3, for all threads in the block)
    for (int i = threadIdx.x; i < 3 * 66; i += blockDim.x)
    {
        const int tileCol = i / 3;
        const int tileRow = i % 3;
        // clamp (mimic cudaAddressModeClamp)
		const int globalX = clamp<int>((int)(blockIdx.x * blockDim.x) + tileCol - 1, 0, width - 1);
		const int globalY = clamp<int>((int)(blockIdx.y * blockDim.y) + tileRow - 1, 0, height - 1);
        blockValues[tileRow][tileCol] = HALF(input[globalX * height + globalY]);
    }
    __syncthreads();

    //read the 3x3 window from shared memory
    const int localX = threadIdx.x + 1; // center
	const half x_4 = blockValues[1][threadIdx.x + 1]; // center pixel
    const half8 localBlock = make_half8(
        blockValues[0][localX - 1], blockValues[0][localX], blockValues[0][localX + 1], 
        blockValues[1][localX - 1], blockValues[1][localX + 1], 
        blockValues[2][localX - 1], blockValues[2][localX], blockValues[2][localX + 1]   
    );
    if (x < width)
    {
        //calculate this thread's 8 rx values
        me_p3_rxCalculate(RxLocalVec8, localBlock, x_4);
    }
    __syncthreads();

    //optimized summation for rx with warp shuffling
    float sum = 0;
    const int rxRow = threadIdx.x / 8;
    #pragma unroll
    for (int i = 0; i < 64; i += 8)
        sum += FLOAT(RxLocal[(threadIdx.x + i) % 64][rxRow]);
    // reduce 32 results to 4 per warp
    for (int i = 4; i > 0; i = i / 2)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, i);
    if (threadIdx.x % 8 == 0)
        rx[(outputIndex + rxRow) / 8] = sum;
    __syncthreads();

	//optimized summation for Rx with WMMA (Tensor Cores)
    *RxLocalVec8 = localBlock;
    __syncthreads();

    //compute C = X^T * X with Tensor Cores
    //using the first warp than using both warps is faster (lower shared memory contention)
    if (threadIdx.x < 32)
    {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> A;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> B;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> C;
        wmma::fill_fragment(C, 0.0f);
#pragma unroll
        for (int k0 = 0; k0 < 64; k0 += 16)
        {
            const half* tile = &RxLocal[k0][0];
            wmma::load_matrix_sync(A, tile, 16);
            wmma::load_matrix_sync(B, tile, 16);
            wmma::mma_sync(C, A, B, C);
        }
        //store full 16x16 accumulator to shared, row-major
        wmma::store_matrix_sync(&RxWmmaAccum[0][0], C, 16, wmma::mem_row_major);
    }
    __syncthreads();

    //write only the top-left 8x8 (64 values) per block
    const int r = threadIdx.x / 8;
    const int c = threadIdx.x % 8;
    Rx[outputIndex] = RxWmmaAccum[r][c];
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

__global__ void nV12ToYUV420p(const void* __restrict__ uvSrc, const int uvPitch, uint8_t* __restrict__ uvDst, const int uvWidth, const int uvHeight, const int bitDepth)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= uvWidth * uvHeight)
        return;
    const int y = idx / uvWidth;
    const int x = idx % uvWidth;
    if (bitDepth == 8)
    {
        const uint8_t* src = reinterpret_cast<const uint8_t*>(uvSrc) + y * uvPitch + 2 * x;
        uvDst[idx] = src[0];
        uvDst[uvWidth * uvHeight + idx] = src[1];
    }
    else 
    {
        //bitDepth == 10 (stored in 16-bit per pixel)
        const uint16_t* src = reinterpret_cast<const uint16_t*>(uvSrc) + y * (uvPitch / 2) +  2 * x;
        uvDst[idx] = static_cast<uint8_t>(scale10To8(src[0]));
        uvDst[uvWidth * uvHeight + idx] = static_cast<uint8_t>(scale10To8(src[1]));
    }
}

__global__ void pitchedToFloat(const void* __restrict__ input, float* __restrict__ output, const int width, const int height, const int pitch, const int bitDepth)
{
    __shared__ float block[16][16 + 1]; //+1 to avoid bank conflicts
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    float convertedValue = 0.f;
    if (x < width && y < height) 
    {
        if (bitDepth <= 8) 
        {
            const uint8_t* in = reinterpret_cast<const uint8_t*>(input);
            convertedValue = static_cast<float>(in[y * pitch + x]);
        }
        else 
        { 
            //10-bit stored in 16-bit, convert to 8-bit range
            const uint16_t* in = reinterpret_cast<const uint16_t*>(input);
            const int pitchOffset = pitch / 2; //pitch in elements
			convertedValue = static_cast<float>(scale10To8(in[y * pitchOffset + x]));
        }
    }

    block[threadIdx.y][threadIdx.x] = convertedValue;
    __syncthreads();

    //write transposed data (coalesced writes to column-major output)
    const int dstX = blockIdx.y * blockDim.y + threadIdx.x;
    const int dstY = blockIdx.x * blockDim.x + threadIdx.y;
    if (dstX < height && dstY < width)
        output[dstY * height + dstX] = block[threadIdx.x][threadIdx.y];
}

__global__ void pitched10To8Bit(const uint16_t* __restrict__ input, uint8_t* __restrict__ output, const int width, const int height, const int pitch)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height)
        output[y * width + x] = static_cast<uint8_t>(scale10To8(input[y * (pitch / 2) + x]));
}