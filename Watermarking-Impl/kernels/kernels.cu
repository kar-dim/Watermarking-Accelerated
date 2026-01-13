#include "kernels.cuh"
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <mma.h>

using namespace nvcuda;

// clang-format off

// constant cache used to map linear index of the [8x8] Rx matrix for p=3 into [x,y] coordinates
// this helps to write back to global memory 36 Rx values per block instead of 64 (8x8) improving performance
__constant__ const short2 c_RxCoordsP3[36] = 
{
    {0,0},
    {1,0}, {1,1},
    {2,0}, {2,1}, {2,2},
    {3,0}, {3,1}, {3,2}, {3,3},
    {4,0}, {4,1}, {4,2}, {4,3}, {4,4},
    {5,0}, {5,1}, {5,2}, {5,3}, {5,4}, {5,5},
    {6,0}, {6,1}, {6,2}, {6,3}, {6,4}, {6,5}, {6,6},
    {7,0}, {7,1}, {7,2}, {7,3}, {7,4}, {7,5}, {7,6}, {7,7}
};
// constant cache used to map linear index of the [24x24] Rx matrix for p=5 into [x,y] coordinates
// this helps to write back to global memory 300 Rx values per block instead of 576 (24x24) improving performance
__constant__ const short2 c_RxCoordsP5[300] = {
    {0,0},
    {1,0}, {1,1},
    {2,0}, {2,1}, {2,2},
    {3,0}, {3,1}, {3,2}, {3,3},
    {4,0}, {4,1}, {4,2}, {4,3}, {4,4},
    {5,0}, {5,1}, {5,2}, {5,3}, {5,4}, {5,5},
    {6,0}, {6,1}, {6,2}, {6,3}, {6,4}, {6,5}, {6,6},
    {7,0}, {7,1}, {7,2}, {7,3}, {7,4}, {7,5}, {7,6}, {7,7},
    {8,0}, {8,1}, {8,2}, {8,3}, {8,4}, {8,5}, {8,6}, {8,7}, {8,8},
    {9,0}, {9,1}, {9,2}, {9,3}, {9,4}, {9,5}, {9,6}, {9,7}, {9,8}, {9,9},
    {10,0}, {10,1}, {10,2}, {10,3}, {10,4}, {10,5}, {10,6}, {10,7}, {10,8}, {10,9}, {10,10},
    {11,0}, {11,1}, {11,2}, {11,3}, {11,4}, {11,5}, {11,6}, {11,7}, {11,8}, {11,9}, {11,10}, {11,11},
    {12,0}, {12,1}, {12,2}, {12,3}, {12,4}, {12,5}, {12,6}, {12,7}, {12,8}, {12,9}, {12,10}, {12,11}, {12,12},
    {13,0}, {13,1}, {13,2}, {13,3}, {13,4}, {13,5}, {13,6}, {13,7}, {13,8}, {13,9}, {13,10}, {13,11}, {13,12}, {13,13},
    {14,0}, {14,1}, {14,2}, {14,3}, {14,4}, {14,5}, {14,6}, {14,7}, {14,8}, {14,9}, {14,10}, {14,11}, {14,12}, {14,13}, {14,14},
    {15,0}, {15,1}, {15,2}, {15,3}, {15,4}, {15,5}, {15,6}, {15,7}, {15,8}, {15,9}, {15,10}, {15,11}, {15,12}, {15,13}, {15,14}, {15,15},
    {16,0}, {16,1}, {16,2}, {16,3}, {16,4}, {16,5}, {16,6}, {16,7}, {16,8}, {16,9}, {16,10}, {16,11}, {16,12}, {16,13}, {16,14}, {16,15}, {16,16},
    {17,0}, {17,1}, {17,2}, {17,3}, {17,4}, {17,5}, {17,6}, {17,7}, {17,8}, {17,9}, {17,10}, {17,11}, {17,12}, {17,13}, {17,14}, {17,15}, {17,16}, {17,17},
    {18,0}, {18,1}, {18,2}, {18,3}, {18,4}, {18,5}, {18,6}, {18,7}, {18,8}, {18,9}, {18,10}, {18,11}, {18,12}, {18,13}, {18,14}, {18,15}, {18,16}, {18,17}, {18,18},
    {19,0}, {19,1}, {19,2}, {19,3}, {19,4}, {19,5}, {19,6}, {19,7}, {19,8}, {19,9}, {19,10}, {19,11}, {19,12}, {19,13}, {19,14}, {19,15}, {19,16}, {19,17}, {19,18}, {19,19},
    {20,0}, {20,1}, {20,2}, {20,3}, {20,4}, {20,5}, {20,6}, {20,7}, {20,8}, {20,9}, {20,10}, {20,11}, {20,12}, {20,13}, {20,14}, {20,15}, {20,16}, {20,17}, {20,18}, {20,19}, {20,20},
    {21,0}, {21,1}, {21,2}, {21,3}, {21,4}, {21,5}, {21,6}, {21,7}, {21,8}, {21,9}, {21,10}, {21,11}, {21,12}, {21,13}, {21,14}, {21,15}, {21,16}, {21,17}, {21,18}, {21,19}, {21,20}, {21,21},
    {22,0}, {22,1}, {22,2}, {22,3}, {22,4}, {22,5}, {22,6}, {22,7}, {22,8}, {22,9}, {22,10}, {22,11}, {22,12}, {22,13}, {22,14}, {22,15}, {22,16}, {22,17}, {22,18}, {22,19}, {22,20}, {22,21}, {22,22},
    {23,0}, {23,1}, {23,2}, {23,3}, {23,4}, {23,5}, {23,6}, {23,7}, {23,8}, {23,9}, {23,10}, {23,11}, {23,12}, {23,13}, {23,14}, {23,15}, {23,16}, {23,17}, {23,18}, {23,19}, {23,20}, {23,21}, {23,22}, {23,23}
};
// clang-format on

// STS.128
__device__ void me_p3_rxCalculate(half8* RxLocalVec8, const half8& vec, const half& center) {
    half8 tmp;
    tmp.a = vec.a * center;
    tmp.b = vec.b * center;
    tmp.c = vec.c * center;
    tmp.d = vec.d * center;
    tmp.e = vec.e * center;
    tmp.f = vec.f * center;
    tmp.g = vec.g * center;
    tmp.h = vec.h * center;
    *RxLocalVec8 = tmp;
}

__device__ void load_neighbor_row_funnel_p3(half& p0, half& p1, half& p2, const half* rowBase) {
    // shift Amount (0 or 16 bits), if threadIdx.x is odd, we shift right by 1 half (16 bits)
    const uint32_t shift = (threadIdx.x & 1) * 16;
    // load 2x 32-bit chunks (4 halves) using the aligned index
    // cast to uint* to load 32 bits at a time (equivalent to half2)
    const uint32_t* ptr = reinterpret_cast<const uint32_t*>(&rowBase[threadIdx.x & ~1]);
    uint32_t u0 = ptr[0]; // Loads halves [aligned, aligned+1]
    uint32_t u1 = ptr[1]; // Loads halves [aligned+2, aligned+3]
    // funnel shift (vectorized selection, this reconstructs the sliding window)
    uint32_t pair0 = __funnelshift_r(u0, u1, shift);
    uint32_t pairFinal = u1 >> shift;
    // unpack
    const half2 hPair0 = reinterpret_cast<half2&>(pair0);
    const half2 hFinal = reinterpret_cast<half2&>(pairFinal);
    // store
    p0 = hPair0.x;
    p1 = hPair0.y;
    p2 = hFinal.x;
}

__device__ void load_neighbor_row_funnel_p5(half& p0, half& p1, half& p2, half& p3, half& p4, const half* rowBase) {
    // shift Amount (0 or 16 bits), if threadIdx.x is odd, we shift right by 1 half (16 bits)
    const uint32_t shift = (threadIdx.x & 1) * 16;
    // load 3 x 32-bit chunks (6 halves) using the aligned index
    // cast to uint* to load 32 bits at a time (equivalent to half2)
    const uint32_t* ptr = reinterpret_cast<const uint32_t*>(&rowBase[threadIdx.x & ~1]);
    // funnel shift (vectorized selection, this reconstructs the sliding window)
    uint32_t pair0 = __funnelshift_r(ptr[0], ptr[1], shift);
    uint32_t pair1 = __funnelshift_r(ptr[1], ptr[2], shift);
    uint32_t pairFinal = ptr[2] >> shift;
    // unpack
    const half2 hPair0 = reinterpret_cast<half2&>(pair0);
    const half2 hPair1 = reinterpret_cast<half2&>(pair1);
    const half2 hFinal = reinterpret_cast<half2&>(pairFinal);
    // store
    p0 = hPair0.x;
    p1 = hPair0.y;
    p2 = hPair1.x;
    p3 = hPair1.y;
    p4 = hFinal.x;
}

__device__ void load_neighbor_vec_p5(half8* dst, const half blockValues[5][260], half& center) {
    half8 v0, v1, v2;
    load_neighbor_row_funnel_p5(v0.a, v0.b, v0.c, v0.d, v0.e, blockValues[0]);
    load_neighbor_row_funnel_p5(v0.f, v0.g, v0.h, v1.a, v1.b, blockValues[1]);
    load_neighbor_row_funnel_p5(v1.c, v1.d, center, v1.e, v1.f, blockValues[2]);
    load_neighbor_row_funnel_p5(v1.g, v1.h, v2.a, v2.b, v2.c, blockValues[3]);
    load_neighbor_row_funnel_p5(v2.d, v2.e, v2.f, v2.g, v2.h, blockValues[4]);
    // STS.128
    dst[0] = v0;
    dst[1] = v1;
    dst[2] = v2;
    dst[3] = {};
}

__global__ void me_p3(const float* __restrict__ input, float* __restrict__ Rx, float* __restrict__ rx, const unsigned int width, const unsigned int height) {
    constexpr int sharedMemStride = 24; // 16 + 8 for padding to minimize bank conflicts (padding is CRITICAL for performance)

    const int tid = threadIdx.x;
    const int x = blockIdx.x * 256 + tid;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int warpId = tid / 32;
    const int startRow = warpId * 32;
    const half halfScaleFactor = __float2half(0.00392156862f); // multiplication with stored value of (1/255) is faster than division by 255

    // shared memory for Rx, rx, scratch and all pixels utilized by the whole block
    __shared__ alignas(16) half RxLocal[256][sharedMemStride];
    __shared__ half blockValues[3][258];

    fillBlockStrip<3>(blockValues, input, width, height);
    __syncthreads();

    if (y >= height)
        return;

    // read the 3x3 window from shared memory
    half center;
    half8 localBlock;
    half8* RxLocalVec8 = reinterpret_cast<half8*>(&RxLocal[tid][0]);
    RxLocalVec8[1] = {};
    // if/else is better than always writing (even vectorized) to shared memory unnecessarily
    if (x < width) {
        load_neighbor_row_funnel_p3(localBlock.a, localBlock.b, localBlock.c, blockValues[0]);
        load_neighbor_row_funnel_p3(localBlock.d, center, localBlock.e, blockValues[1]);
        load_neighbor_row_funnel_p3(localBlock.f, localBlock.g, localBlock.h, blockValues[2]);
        RxLocalVec8[0] = localBlock;
    } else {
        RxLocalVec8[0] = {};
        localBlock = {};
        center = blockValues[1][tid + 1]; // center pixel
    }
    // compute rx, use half2 for faster reductions
    half8 rxVec;
    me_p3_rxCalculate(&rxVec, localBlock, center);
    half2* rxHalf2Ptr = reinterpret_cast<half2*>(&rxVec);
#pragma unroll
    for (int k = 0; k < 4; k++) {
        half2 val = rxHalf2Ptr[k];
        for (int offset = 16; offset > 0; offset >>= 1)
            val = __hadd2(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
        rxHalf2Ptr[k] = val;
    }

    // exchange (each warp leader)
    if ((tid & 31) == 0)
        RxLocalVec8[2] = rxVec;
    __syncthreads();

    // compute Rx with Tensor Cores
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> A;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> B;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> C;
    wmma::fill_fragment(C, 0.0f);
#pragma unroll
    for (int k0 = 0; k0 < 32; k0 += 16) {
        const half* tile_ptr = &RxLocal[startRow + k0][0];
        wmma::load_matrix_sync(A, tile_ptr, sharedMemStride);
        wmma::load_matrix_sync(B, tile_ptr, sharedMemStride);
        wmma::mma_sync(C, A, B, C);
    }

    // store matrix at warpId * 32 (row 0, 32, 64...)
    // so that we will overwrite only the first 16 rows of the warp's chunk
    // rows 16-31 of the chunk are NOT touched (preventing race conditions with neighbor warps)
    half* warpOutput = &RxLocal[warpId * 32][0];
    wmma::store_matrix_sync(warpOutput, C, sharedMemStride, wmma::mem_row_major);
    __syncthreads();

    // first 36 threads (warps 1 and partial from 2) write Rx
    if (tid < 36) {
        const short2 coords = c_RxCoordsP3[tid];
        const int r = coords.x;
        const int c = coords.y;
        float sum = 0.0f;
#pragma unroll
        for (int w = 0; w < 8; w++)
            sum += __half2float(RxLocal[w * 32 + r][c]);
        const int outputIndex = (y * gridDim.x * 36) + (blockIdx.x * 36);
        Rx[outputIndex + tid] = sum;
    }
    // use warp 3 (warp 2 is blocked for Rx) to write rx in parallel to Rx
    if (tid >= 64 && tid < 72) {
        const int rxTid = tid - 64; // remap id to [0,7]
        float sum = 0.0f;
#pragma unroll
        for (int w = 0; w < 8; w++)
            sum += __half2float(RxLocal[w * 32][16 + rxTid]);
        const int outputIndex = (y * gridDim.x * 8) + (blockIdx.x * 8);
        rx[outputIndex + rxTid] = sum;
    }
}

__global__ void me_p5(const float* __restrict__ input, float* __restrict__ Rx, float* __restrict__ rx, const unsigned int width, const unsigned int height) {
    constexpr int sharedMemStride = 40; // 32 + 8 for padding to minimize bank conflicts (padding is CRITICAL for performance)

    const int tid = threadIdx.x;
    const int x = blockIdx.x * 256 + tid;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int warpId = tid / 32;
    const int startRow = warpId * 32;

    // shared memory for Rx, rx, scratch and all pixels utilized by the whole block
    __shared__ alignas(16) half RxLocal[256][sharedMemStride];
    __shared__ half blockValues[5][260];

    fillBlockStrip<5>(blockValues, input, width, height);
    __syncthreads();

    if (y >= height)
        return;

    // read the 5x5 window from shared memory
    half centerVal;
    half2 center; // half -> half2 for vectorized ops later
    half8* localVec8 = reinterpret_cast<half8*>(&RxLocal[tid][0]);
    // if/else is better than always writing (even vectorized) to shared memory unnecessarily
    if (x < width) {
        load_neighbor_vec_p5(localVec8, blockValues, centerVal);
        center = __half2half2(centerVal);
    } else {
#pragma unroll
        for (int i = 0; i < 4; i++)
            localVec8[i] = {};
        center = __half2half2(blockValues[2][tid + 2]); // center pixel
    }

    // do not compute rx yet, compute Rx first
    // TENSOR CORE Rx ACCUMULATION (24x24 matrix -> 32x32 tiled)
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> A_low, A_high;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> B_low, B_high;
    // accumulators for the 32x32 grid
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> C00, C01, C10, C11;
    wmma::fill_fragment(C00, 0.0f);
    wmma::fill_fragment(C01, 0.0f);
    wmma::fill_fragment(C10, 0.0f);
    wmma::fill_fragment(C11, 0.0f);

    // loop for the 32 pixels in this warp
#pragma unroll
    for (int k0 = 0; k0 < 32; k0 += 16) {
        // pointer to the start of the 16-pixel batch for this warp
        const half* tilePtr = &RxLocal[startRow + k0][0];
        // load lower half
        wmma::load_matrix_sync(A_low, tilePtr, sharedMemStride);
        wmma::load_matrix_sync(B_low, tilePtr, sharedMemStride);
        // load upper half
        wmma::load_matrix_sync(A_high, tilePtr + 16, sharedMemStride);
        wmma::load_matrix_sync(B_high, tilePtr + 16, sharedMemStride);
        // compute 4 Matrices (2x2)
        wmma::mma_sync(C00, A_low, B_low, C00);   // top left
        wmma::mma_sync(C01, A_low, B_high, C01);  // top right
        wmma::mma_sync(C10, A_high, B_low, C10);  // bottom left
        wmma::mma_sync(C11, A_high, B_high, C11); // bottom right
    }

    // compute rx (vectorized 128-bit plus half2 for maximum efficiency)
    half8 rxVec[3];
#pragma unroll
    for (int i = 0; i < 3; i++) {
        half2* inPtr = reinterpret_cast<half2*>(&localVec8[i]);
        half2* resultPtr = reinterpret_cast<half2*>(&rxVec[i]);
#pragma unroll
        for (int i = 0; i < 4; i++)
            resultPtr[i] = __hmul2(inPtr[i], center);
    }

    // rx warp-level reduction
    // lane 0 will end up holding the sum of the entire warp
    half2* rxHalf2 = reinterpret_cast<half2*>(rxVec);
#pragma unroll
    for (int i = 0; i < 12; i++) {
        half2 sum = rxHalf2[i];
        for (int offset = 16; offset > 0; offset >>= 1) {
            int shflInt = __shfl_down_sync(0xFFFFFFFF, reinterpret_cast<int&>(sum), offset);
            sum = __hadd2(sum, reinterpret_cast<half2&>(shflInt));
        }
        rxHalf2[i] = sum;
    }

    // store Rx to shared mem, NOTE: we don't care about top-right, don't store C01 and waste speed
    half* warpOutput = &RxLocal[warpId * 32][0];
    wmma::store_matrix_sync(warpOutput, C00, sharedMemStride, wmma::mem_row_major);
    wmma::store_matrix_sync(warpOutput + 16 * sharedMemStride, C10, sharedMemStride, wmma::mem_row_major);
    wmma::store_matrix_sync(warpOutput + 16 * sharedMemStride + 16, C11, sharedMemStride, wmma::mem_row_major);

    // store rx partials to the unused memory zone (row 31)
    // row 31 is unused by the 24x24 matrix (chunk is 32x32) -> safe from overwrite
    if ((tid & 31) == 0) {
        half8* dst = reinterpret_cast<half8*>(&RxLocal[warpId * 32 + 31][0]);
        dst[0] = rxVec[0];
        dst[1] = rxVec[1];
        dst[2] = rxVec[2];
    }
    __syncthreads();

    // write rx entirely by warp 1
    if (tid < 32) {
        if (tid < 24) {
            float sum = 0.0f;
            // this is the unused memory zone we wrote before (row 31 of each chunk)
#pragma unroll
            for (int w = 0; w < 8; w++)
                sum += __half2float(RxLocal[w * 32 + 31][tid]);
            const int rxBaseIndex = (y * gridDim.x * 24) + (blockIdx.x * 24);
            rx[rxBaseIndex + tid] = sum;
        }
    }
    // warps 2-8 handle Rx in parallel (threads with id from 32 to 255) write 300 values
    else {
        const int workerIdx = tid - 32; // bring tid to [0,223]
        const int RxBaseIndex = (y * gridDim.x * 300) + (blockIdx.x * 300);

        for (int i = workerIdx; i < 300; i += 224) {
            const short2 coords = c_RxCoordsP5[i];
            float sum = 0.0f;
#pragma unroll
            for (int w = 0; w < 8; w++)
                sum += __half2float(RxLocal[w * 32 + coords.x][coords.y]);
            Rx[RxBaseIndex + i] = sum;
        }
    }
}

__global__ void calculate_partial_correlation(const float* __restrict__ e_u, const float* __restrict__ e_z, float* __restrict__ partialDots, float* __restrict__ partialNormU,
                                              float* __restrict__ partialNormZ, const unsigned int size) {
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + tid;
    const int warpId = tid / 32;

    // support for up to 1024/32 = 32 warps per block
    __shared__ float dotCache[32];
    __shared__ float normUCache[32];
    __shared__ float normZCache[32];

    float a = 0.0f, b = 0.0f;
    if (idx < size) {
        a = e_u[idx];
        b = e_z[idx];
    }

    float dotVal = a * b;
    float normUVal = a * a;
    float normZVal = b * b;

    // intra-warp reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        dotVal += __shfl_down_sync(0xFFFFFFFF, dotVal, offset);
        normUVal += __shfl_down_sync(0xFFFFFFFF, normUVal, offset);
        normZVal += __shfl_down_sync(0xFFFFFFFF, normZVal, offset);
    }

    // warp leaders write to shared memory
    if ((tid & 31) == 0) {
        dotCache[warpId] = dotVal;
        normUCache[warpId] = normUVal;
        normZCache[warpId] = normZVal;
    }
    __syncthreads();

    // final reduction by first warp
    if (tid < 32) {
        const bool validTid = tid < (blockDim.x + warpSize - 1) / 32;
        dotVal = validTid ? dotCache[tid] : 0.0f;
        normUVal = validTid ? normUCache[tid] : 0.0f;
        normZVal = validTid ? normZCache[tid] : 0.0f;

        for (int offset = 16; offset > 0; offset >>= 1) {
            dotVal += __shfl_down_sync(0xFFFFFFFF, dotVal, offset);
            normUVal += __shfl_down_sync(0xFFFFFFFF, normUVal, offset);
            normZVal += __shfl_down_sync(0xFFFFFFFF, normZVal, offset);
        }

        if (tid == 0) {
            partialDots[blockIdx.x] = dotVal;
            partialNormU[blockIdx.x] = normUVal;
            partialNormZ[blockIdx.x] = normZVal;
        }
    }
}

__global__ void calculate_final_correlation(const float* __restrict__ partialDots, const float* __restrict__ partialNormU, const float* __restrict__ partialNormZ, float* __restrict__ result,
                                            const unsigned int numBlocks) {
    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int warpId = tid / 32;
    const int numWarps = (blockDim.x + 31) / 32;

    // shared memory must match number of warps
    __shared__ float warpDot[32];
    __shared__ float warpU[32];
    __shared__ float warpZ[32];

    float localDot = 0.0f;
    float localU = 0.0f;
    float localZ = 0.0f;

    for (int i = tid; i < numBlocks; i += blockDim.x) {
        localDot += partialDots[i];
        localU += partialNormU[i];
        localZ += partialNormZ[i];
    }

    // intra-warp reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        localDot += __shfl_down_sync(0xFFFFFFFF, localDot, offset);
        localU += __shfl_down_sync(0xFFFFFFFF, localU, offset);
        localZ += __shfl_down_sync(0xFFFFFFFF, localZ, offset);
    }
    if (lane == 0) {
        warpDot[warpId] = localDot;
        warpU[warpId] = localU;
        warpZ[warpId] = localZ;
    }
    __syncthreads();

    // final warp reduces
    if (warpId == 0) {
        const bool validTid = tid < numWarps;
        localDot = validTid ? warpDot[lane] : 0.0f;
        localU = validTid ? warpU[lane] : 0.0f;
        localZ = validTid ? warpZ[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            localDot += __shfl_down_sync(0xFFFFFFFF, localDot, offset);
            localU += __shfl_down_sync(0xFFFFFFFF, localU, offset);
            localZ += __shfl_down_sync(0xFFFFFFFF, localZ, offset);
        }
        if (lane == 0) {
            float normU = sqrtf(localU);
            float normZ = sqrtf(localZ);
            result[0] = (normU > 0.0f && normZ > 0.0f) ? (localDot / (normU * normZ)) : 0.0f;
        }
    }
}

__global__ void nV12ToYUV420p(const uint8_t* __restrict__ uvSrc, const int uvPitch, uint8_t* __restrict__ uvDst, const int uvWidth, const int uvHeight) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= uvWidth * uvHeight)
        return;
    const int y = idx / uvWidth;
    const int x = idx % uvWidth;
    const uint8_t* src = uvSrc + y * uvPitch + 2 * x;
    uvDst[idx] = src[0];
    uvDst[uvWidth * uvHeight + idx] = src[1];
}

__global__ void pitchedToFloat(const uint8_t* __restrict__ input, float* __restrict__ output, const int width, const int height, const int pitch) {
    __shared__ float block[16][16 + 1]; //+1 to avoid bank conflicts
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    float convertedValue = 0.0f;
    if (x < width && y < height)
        convertedValue = static_cast<float>(input[y * pitch + x]);

    block[threadIdx.y][threadIdx.x] = convertedValue;
    __syncthreads();

    // write transposed data (coalesced writes to column-major output)
    const int dstX = blockIdx.y * blockDim.y + threadIdx.x;
    const int dstY = blockIdx.x * blockDim.x + threadIdx.y;
    if (dstX < height && dstY < width)
        output[dstY * height + dstX] = block[threadIdx.x][threadIdx.y];
}