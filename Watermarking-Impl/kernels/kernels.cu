#include "kernels.cuh"
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <mma.h>

using namespace nvcuda;

//constant cache used to map linear index of the [24x24] Rx matrix for p=5 into [x,y] coordinates
//this helps to write back to global memory 300 Rx values per block instead of 576 (24x24) improving performance
__constant__ short2 c_RxCoordsP5[300];
__host__ void initRxMapP5()
{
    short2 hRxCoordsP5[300];
    int idx = 0;
    for (int r = 0; r < 24; r++)
        for (int c = 0; c <= r; c++)
            hRxCoordsP5[idx++] = make_short2((short)r, (short)c);
    cudaMemcpyToSymbol(c_RxCoordsP5, hRxCoordsP5, sizeof(hRxCoordsP5));
}

//naive 1-thread Cholesky solver used for its very low latency versus cuSOLVER but useful only for very small systems, p = 3 (N = 8) or p = 5 (N = 24)
template<int p>
__global__ void cholesky_solver(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ X, int* __restrict__ stopFlag)
{
    constexpr int N = (p * p) - 1;

    if (threadIdx.x > 0 || blockIdx.x > 0)
        return;

    float localA[N * N], localB[N], localX[N];

    //initialize Result to 0.0f (safe fallback for unsolvable systems)
    for (int i = 0; i < N; i++)
        localX[i] = 0.0f;

    //initialize Rx and rx
    //p=5: Rx is in packed format, unpack fast 
    //because we use 1 thread, it is faster to unpack manually than use the constant cache map!
    if constexpr (p == 5)
    {
        int k = 0;
#pragma unroll
        for (int r = 0; r < N; r++)
        {
#pragma unroll
            for (int c = 0; c <= r; c++)
            {
                float val = A[k++];
                localA[r * N + c] = val;
                localA[c * N + r] = val;
            }
        }
    }
    //p=3: Rx is already full (64 floats, just copy)
    else
    {
#pragma unroll
        for (int i = 0; i < N * N; i++)
            localA[i] = A[i];
    }
    //copy rx
    for (int i = 0; i < N; i++)
        localB[i] = B[i];

    //Cholesky Decomposition: A = L*L^T
    //A is symmetric positive definite
    float L[N][N];
    //clear L
#pragma unroll
    for (int i = 0; i < N; i++)
#pragma unroll
        for (int j = 0; j < N; j++)
            L[i][j] = 0.0f;
#pragma unroll
    for (int i = 0; i < N; i++)
    {
#pragma unroll
        for (int j = 0; j <= i; j++)
        {
            float sum = 0.0f;
            for (int k = 0; k < j; k++)
                sum += L[i][k] * L[j][k];
            if (i == j)
            {
                //diagonal element
                const float val = localA[i * N + i] - sum;
                //check if singular! if so, exit early with X = 0.0f
                if (val <= 1e-12f)
                {
                    *stopFlag = 1;
                    goto exit;
                }
                L[i][j] = sqrtf(val);
            }
            else //non diagonal
                L[i][j] = (localA[i * N + j] - sum) * __frcp_rn(L[j][j]); //fast reciprocal
        }
    }
    //solve the system with forward and backward substitution
    //we again use fast reciprocal for better performance (1 GPU thread is weak, needs as fast math as possible)
    //forward substitution -> solve L*y = b
    float y[N];
#pragma unroll
    for (int i = 0; i < N; i++)
    {
        float sum = 0.0f;
        for (int k = 0; k < i; k++)
            sum += L[i][k] * y[k];
        y[i] = (localB[i] - sum) * __frcp_rn(L[i][i]);
    }

    //backward substitution -> solve L^T * x = y
#pragma unroll
    for (int i = N - 1; i >= 0; i--)
    {
        float sum = 0.0f;
        for (int k = i + 1; k < N; k++)
            sum += L[k][i] * localX[k]; //transposed
        localX[i] = (y[i] - sum) * __frcp_rn(L[i][i]);
    }
    *stopFlag = 0;
    //write
exit:
    for (int i = 0; i < N; i++)
        X[i] = localX[i];
}

template<>
void launch_cholesky<3>(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ X, int* __restrict__ stopFlag, cudaStream_t stream)
{
    cholesky_solver<3> << <1, 1, 0, stream >> > (A, B, X, stopFlag);
}

// Specialization for p=5
template<>
void launch_cholesky<5>(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ X, int* __restrict__ stopFlag, cudaStream_t stream)
{
    cholesky_solver<5> << <1, 1, 0, stream >> > (A, B, X, stopFlag);
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
    const half halfScaleFactor = __float2half(0.00392156862f); //multiplication with stored value of (1/255) is faster than division by 255

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
        blockValues[tileRow][tileCol] = __float2half(input[globalX * height + globalY]) * halfScaleFactor;
    }
    __syncthreads();

    if (y >= height)
        return;

    //read the 3x3 window from shared memory
    const int localX = tid + 1; //center index
    const half center = blockValues[1][localX];
    half8 localBlock = {
        blockValues[0][localX - 1], blockValues[0][localX], blockValues[0][localX + 1],
        blockValues[1][localX - 1], blockValues[1][localX + 1],
        blockValues[2][localX - 1], blockValues[2][localX], blockValues[2][localX + 1]
    };

    half8* RxLocalVec8 = reinterpret_cast<half8*>(&RxLocal[tid][0]);
    RxLocalVec8[1] = {};
    //if/else is better than always writing (even vectorized) to shared memory unnecessarily
    if (x < width)
        RxLocalVec8[0] = localBlock;
    else
    {
        RxLocalVec8[0] = {};
        localBlock = {};
    }
    //compute rx, use half2 for faster reductions
    half8 rxVec;
    me_p3_rxCalculate(&rxVec, localBlock, center);
    half2* rxHalf2Ptr = reinterpret_cast<half2*>(&rxVec);
#pragma unroll
    for (int k = 0; k < 4; k++)
    {
        half2 val = rxHalf2Ptr[k];
        for (int offset = 16; offset > 0; offset >>= 1)
            val = __hadd2(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
        rxHalf2Ptr[k] = val;
    }

    //exchange (each warp leader)
    if ((tid & 31) == 0)
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
    half* warpOutput = &RxLocal[warpId * 32][0];
    wmma::store_matrix_sync(warpOutput, C, sharedMemStride, wmma::mem_row_major);
    __syncthreads();

    //first 8 threads write rx
    if (tid < 8)
    {
        const int outputIndex = (y * gridDim.x * 8) + (blockIdx.x * 8);
        float sum_v = 0.0f;
#pragma unroll
        for (int w = 0; w < 8; w++)
            sum_v += __half2float(RxLocal[w * 32][16 + tid]);
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
            sum += __half2float(RxLocal[w * 32 + r][c]);
        const int outputIndex = (y * gridDim.x * 64) + (blockIdx.x * 64);
        Rx[outputIndex + tid] = sum;
    }
}

__device__ void load_neighbor_vec_p5(half8* dst, const half blockValues[5][260], const int localX)
{
    //load all (5x5) - 1 = 24 neighbors and store them as 32
    //(3 x half8, plus 1 x half8 with zeros for WMMA later) -> STS.128
    half8 v0, v1, v2;

    //row 0 begin
    v0.a = blockValues[0][localX - 2];
    v0.b = blockValues[0][localX - 1];
    v0.c = blockValues[0][localX];
    v0.d = blockValues[0][localX + 1];
    v0.e = blockValues[0][localX + 2];
    //row 1 begin
    v0.f = blockValues[1][localX - 2];
    v0.g = blockValues[1][localX - 1];
    v0.h = blockValues[1][localX];

    v1.a = blockValues[1][localX + 1];
    v1.b = blockValues[1][localX + 2];
    //row 2 begin (center row, skip center pixel)
    v1.c = blockValues[2][localX - 2];
    v1.d = blockValues[2][localX - 1];
    v1.e = blockValues[2][localX + 1];
    v1.f = blockValues[2][localX + 2];
    //row 3 begin
    v1.g = blockValues[3][localX - 2];
    v1.h = blockValues[3][localX - 1];

    v2.a = blockValues[3][localX];
    v2.b = blockValues[3][localX + 1];
    v2.c = blockValues[3][localX + 2];
    //row 4
    v2.d = blockValues[4][localX - 2];
    v2.e = blockValues[4][localX - 1];
    v2.f = blockValues[4][localX];
    v2.g = blockValues[4][localX + 1];
    v2.h = blockValues[4][localX + 2];

    //4x STS.128
    dst[0] = v0;
    dst[1] = v1;
    dst[2] = v2;
    dst[3] = {};
}

__global__ void me_p5(const float* __restrict__ input, float* __restrict__ Rx, float* __restrict__ rx, const unsigned int width, const unsigned int height)
{
    constexpr int sharedMemStride = 32;

    const int tid = threadIdx.x;
    const int x = blockIdx.x * 256 + tid;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int warpId = tid / 32;
    const int startRow = warpId * 32;
    const half halfScaleFactor = __float2half(0.00392156862f); //multiplication with stored value of (1/255) is faster than division by 255

    //shared memory for Rx, rx, scratch and all pixels utilized by the whole block
    __shared__ alignas(16) half RxLocal[256][sharedMemStride];
    __shared__ half blockValues[5][260];

    //cooperatively load the 5 x (blockSize + 4) block (window size 5x5, for all threads in the block)
    for (int i = tid; i < 5 * 260; i += 256)
    {
        const int tileCol = i / 5;
        const int tileRow = i % 5;
        //clamp (mimic cudaAddressModeClamp)
        const int globalX = clamp<int>((int)(blockIdx.x * 256) + tileCol - 2, 0, width - 1);
        const int globalY = clamp<int>((int)(blockIdx.y * 1) + tileRow - 2, 0, height - 1);
        //normalize from [0,255] to [0,1] to support half precision and avoid overflow in multiplications
        blockValues[tileRow][tileCol] = __float2half(input[globalX * height + globalY]) * halfScaleFactor;
    }
    __syncthreads();

    if (y >= height)
        return;

    //read the 5x5 window from shared memory
    const int localX = tid + 2; //center index
    const half2 center = __half2half2(blockValues[2][localX]); //half -> half2 for vectorized ops later
    half8* localVec8 = reinterpret_cast<half8*>(&RxLocal[tid][0]);
    //if/else is better than always writing (even vectorized) to shared memory unnecessarily
    if (x < width)
        load_neighbor_vec_p5(localVec8, blockValues, localX);
    else
    {
        half8 zero = {};
        localVec8[0] = zero;
        localVec8[1] = zero;
        localVec8[2] = zero;
        localVec8[3] = zero;
    }

    //do not compute rx yet, compute Rx first
    //TENSOR CORE Rx ACCUMULATION (24x24 matrix -> 32x32 tiled)
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> A_low, A_high;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> B_low, B_high;
    //accumulators for the 32x32 grid
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> C00, C01, C10, C11;
    wmma::fill_fragment(C00, 0.0f);
    wmma::fill_fragment(C01, 0.0f);
    wmma::fill_fragment(C10, 0.0f);
    wmma::fill_fragment(C11, 0.0f);

    //loop for the 32 pixels in this warp
#pragma unroll
    for (int k0 = 0; k0 < 32; k0 += 16)
    {
        //pointer to the start of the 16-pixel batch for this warp
        const half* tilePtr = &RxLocal[startRow + k0][0];
        //load lower half
        wmma::load_matrix_sync(A_low, tilePtr, sharedMemStride);
        wmma::load_matrix_sync(B_low, tilePtr, sharedMemStride);
        //load upper half
        wmma::load_matrix_sync(A_high, tilePtr + 16, sharedMemStride);
        wmma::load_matrix_sync(B_high, tilePtr + 16, sharedMemStride);
        //compute 4 Matrices (2x2)
        wmma::mma_sync(C00, A_low, B_low, C00);   //top left
        wmma::mma_sync(C01, A_low, B_high, C01);  //top right
        wmma::mma_sync(C10, A_high, B_low, C10);  //bottom left
        wmma::mma_sync(C11, A_high, B_high, C11); //bottom right
    }

    //compute rx (vectorized 128-bit plus half2 for maximum efficiency)
    half8 rxVec[3];
#pragma unroll
    for (int i = 0; i < 3; i++)
    {
        half2* inPtr = reinterpret_cast<half2*>(&localVec8[i]);
        half2* resultPtr = reinterpret_cast<half2*>(&rxVec[i]);
        resultPtr[0] = __hmul2(inPtr[0], center);
        resultPtr[1] = __hmul2(inPtr[1], center);
        resultPtr[2] = __hmul2(inPtr[2], center);
        resultPtr[3] = __hmul2(inPtr[3], center);
    }

    //store Rx to shared mem
    half* myWarpOutput = &RxLocal[warpId * 32][0];
    wmma::store_matrix_sync(myWarpOutput, C00, sharedMemStride, wmma::mem_row_major); //(0,0)
    wmma::store_matrix_sync(myWarpOutput + 16, C01, sharedMemStride, wmma::mem_row_major); //(0,16)
    wmma::store_matrix_sync(myWarpOutput + 16 * sharedMemStride, C10, sharedMemStride, wmma::mem_row_major); //(16,0)
    wmma::store_matrix_sync(myWarpOutput + 16 * sharedMemStride + 16, C11, sharedMemStride, wmma::mem_row_major); //(16,16)
    __syncthreads();

    //write Rx
    const int RxBaseIndex = (y * gridDim.x * 300) + (blockIdx.x * 300);
    for (int i = tid; i < 300; i += blockDim.x)
    {
        const short2 coords = c_RxCoordsP5[i];
        const int r = coords.x;
        const int c = coords.y;
        float sum = 0.0f;
#pragma unroll
        for (int w = 0; w < 8; w++)
            sum += __half2float(RxLocal[w * 32 + r][c]);
        Rx[RxBaseIndex + i] = sum;
    }
    __syncthreads();

    //reduction for rx (vectorized 128-bit plus half2 for maximum efficiency)
    half2* rxHalf2 = reinterpret_cast<half2*>(rxVec);
#pragma unroll
    for (int i = 0; i < 12; i++)
    {
        half2 sum = rxHalf2[i];
        for (int offset = 16; offset > 0; offset >>= 1)
        {
            int shfl_int = __shfl_down_sync(0xFFFFFFFF, reinterpret_cast<int&>(sum), offset);
            sum = __hadd2(sum, reinterpret_cast<half2&>(shfl_int));
        }
        rxHalf2[i] = sum;
    }
    if ((tid & 31) == 0)
    {
        half8* dstRow = reinterpret_cast<half8*>(&RxLocal[warpId][0]);
        dstRow[0] = rxVec[0];
        dstRow[1] = rxVec[1];
        dstRow[2] = rxVec[2];
    }
    __syncthreads();

    //first 24 threads sum the 8 warp results and write to rx
    const int rxBaseIndex = (y * gridDim.x * 24) + (blockIdx.x * 24);
    if (tid < 24) 
    {
        float sum = 0.0f;
#pragma unroll
        for (int w = 0; w < 8; w++)
            sum += __half2float(RxLocal[w][tid]);
        rx[rxBaseIndex + tid] = sum;
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
    for (int offset = 16; offset > 0; offset >>= 1)
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

        for (int offset = 16; offset > 0; offset >>= 1)
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