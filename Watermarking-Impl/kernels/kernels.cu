#include "kernels.cuh"
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <mma.h>

using namespace nvcuda;

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

__device__ void load_neighbor_row_funnel_p7(half* dst, const half* rowBase) {
    // shift Amount (0 or 16 bits), if threadIdx.x is odd, we shift right by 1 half (16 bits)
    const uint32_t shift = (threadIdx.x & 1) * 16;
    // load 4 x 32-bit chunks (8 halves) using the aligned index
    // cast to uint* to load 32 bits at a time (equivalent to half2)
    const uint32_t* ptr = reinterpret_cast<const uint32_t*>(&rowBase[threadIdx.x & ~1]);
    // load 4 chunks (128 bits total read, covers 8 halves)
    uint32_t u0 = ptr[0];
    uint32_t u1 = ptr[1];
    uint32_t u2 = ptr[2];
    uint32_t u3 = ptr[3];
    // funnel shift (vectorized selection, this reconstructs the sliding window)
    uint32_t p0 = __funnelshift_r(u0, u1, shift);
    uint32_t p1 = __funnelshift_r(u1, u2, shift);
    uint32_t p2 = __funnelshift_r(u2, u3, shift);
    uint32_t p3 = u3 >> shift;
    // unpack
    reinterpret_cast<half2*>(dst)[0] = reinterpret_cast<half2&>(p0);
    reinterpret_cast<half2*>(dst)[1] = reinterpret_cast<half2&>(p1);
    reinterpret_cast<half2*>(dst)[2] = reinterpret_cast<half2&>(p2);
    // store
    dst[6] = reinterpret_cast<half2&>(p3).x;
}

__device__ void load_neighbor_row_funnel_p9(half* dst, const half* rowBase) {
    // shift Amount (0 or 16 bits), if threadIdx.x is odd, we shift right by 1 half (16 bits)
    const uint32_t shift = (threadIdx.x & 1) * 16;
    // load 5 x 32-bit chunks (10 halves) to cover the 9 pixels + alignment
    // cast to uint* to load 32 bits at a time (equivalent to half2)
    const uint32_t* ptr = reinterpret_cast<const uint32_t*>(&rowBase[threadIdx.x & ~1]);
    // load 5 chunks (160 bits total read, covers 10 halves)
    uint32_t u0 = ptr[0];
    uint32_t u1 = ptr[1];
    uint32_t u2 = ptr[2];
    uint32_t u3 = ptr[3];
    uint32_t u4 = ptr[4];
    // funnel shift (vectorized selection, this reconstructs the sliding window)
    uint32_t p0 = __funnelshift_r(u0, u1, shift);
    uint32_t p1 = __funnelshift_r(u1, u2, shift);
    uint32_t p2 = __funnelshift_r(u2, u3, shift);
    uint32_t p3 = __funnelshift_r(u3, u4, shift);
    uint32_t p4 = u4 >> shift; // Last chunk
    // unpack
    reinterpret_cast<half2*>(dst)[0] = reinterpret_cast<half2&>(p0);
    reinterpret_cast<half2*>(dst)[1] = reinterpret_cast<half2&>(p1);
    reinterpret_cast<half2*>(dst)[2] = reinterpret_cast<half2&>(p2);
    reinterpret_cast<half2*>(dst)[3] = reinterpret_cast<half2&>(p3);
    // store
    dst[8] = reinterpret_cast<half2&>(p4).x;
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

__device__ void load_neighbor_vec_p7(half8* dst, const half blockValues[7][262], half& center) {
    half rows[7][8]; // 8 cols for padding/alignment simplicity
#pragma unroll
    for (int i = 0; i < 7; i++)
        load_neighbor_row_funnel_p7(rows[i], blockValues[i]);

    center = rows[3][3];
    half* d = reinterpret_cast<half*>(dst);
    int idx = 0;
#pragma unroll
    for (int r = 0; r < 7; r++) {
        if (r == 3) {
            d[idx++] = rows[r][0];
            d[idx++] = rows[r][1];
            d[idx++] = rows[r][2];
            d[idx++] = rows[r][4];
            d[idx++] = rows[r][5];
            d[idx++] = rows[r][6];
        } else {
#pragma unroll
            for (int c = 0; c < 7; c++)
                d[idx++] = rows[r][c];
        }
    }
}

__device__ void load_neighbor_vec_p9(half8* dst, const half blockValues[9][136], half& center) {
    half rows[9][10];
#pragma unroll
    for (int i = 0; i < 9; i++)
        load_neighbor_row_funnel_p9(reinterpret_cast<half*>(rows[i]), blockValues[i]);
    center = rows[4][4];
    // pack 80 halves into 10 half8 vectors (skip center)
    half* d = reinterpret_cast<half*>(dst);
    int idx = 0;
#pragma unroll
    for (int r = 0; r < 9; r++) {
        if (r == 4) {
#pragma unroll
            for (int c = 0; c < 9; c++)
                if (c != 4)
                    d[idx++] = rows[r][c];
        } else {
#pragma unroll
            for (int c = 0; c < 9; c++)
                d[idx++] = rows[r][c];
        }
    }
}

__global__ void me_p3(const float* __restrict__ input, float* __restrict__ Rx, float* __restrict__ rx, const unsigned int width, const unsigned int height) {
    constexpr int sharedMemStride = 24; // 16 + 8 for padding to minimize bank conflicts (padding is CRITICAL for performance)

    const int tid = threadIdx.x;
    const int x = blockIdx.x * 256 + tid;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int warpId = tid / 32;
    const int startRow = warpId * 32;

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
        const half* tilePtr = &RxLocal[startRow + k0][0];
        wmma::load_matrix_sync(A, tilePtr, sharedMemStride);
        wmma::load_matrix_sync(B, tilePtr, sharedMemStride);
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
        const int2 coords = getPackedCoords(tid);
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
    // TENSOR CORE Rx ACCUMULATION (24x24 matrix -> 32x32 tiled, but we want the lower diagonal part only!
    // so we don't compute the upper-right -> skip one tensor core matrix multiply!
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> A_low, A_high;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> B_low, B_high;
    // accumulators for the 3 matrices
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> C00, C10, C11;
    wmma::fill_fragment(C00, 0.0f);
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
        // compute 3 Matrices (2x1)
        wmma::mma_sync(C00, A_low, B_low, C00); // top left
        // skip top right
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
        for (int j = 0; j < 4; j++)
            resultPtr[j] = __hmul2(inPtr[j], center);
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
            const int2 coords = getPackedCoords(i);
            float sum = 0.0f;
#pragma unroll
            for (int w = 0; w < 8; w++)
                sum += __half2float(RxLocal[w * 32 + coords.x][coords.y]);
            Rx[RxBaseIndex + i] = sum;
        }
    }
}

__global__ void me_p7(const float* __restrict__ input, float* __restrict__ Rx, float* __restrict__ rx, const unsigned int width, const unsigned int height) {
    constexpr int sharedMemStride = 56; // 48 + 8 for padding to minimize bank conflicts (padding is CRITICAL for performance)

    const int tid = threadIdx.x;
    const int x = blockIdx.x * 256 + tid;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int warpId = tid / 32;
    const int startRow = warpId * 32;

    // shared memory for Rx, rx, scratch and all pixels utilized by the whole block
    __shared__ alignas(16) half RxLocal[384][sharedMemStride];
    __shared__ half blockValues[7][262];

    fillBlockStrip<7>(blockValues, input, width, height);
    __syncthreads();

    if (y >= height)
        return;

    // read the 7x7 window from shared memory
    half centerVal;
    half2 center; // half -> half2 for vectorized ops later
    half8* localVec8 = reinterpret_cast<half8*>(&RxLocal[tid][0]);
    // if/else is better than always writing (even vectorized) to shared memory unnecessarily
    if (x < width) {
        load_neighbor_vec_p7(localVec8, blockValues, centerVal);
        center = __half2half2(centerVal);
    } else {
#pragma unroll
        for (int i = 0; i < 6; i++)
            localVec8[i] = {};
        center = __half2half2(blockValues[3][tid + 3]); // center pixel
    }

    // do not compute rx yet, compute Rx first
    // TENSOR CORE Rx ACCUMULATION
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> A[3];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> B[3];
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> C[6];
    // accumulators
#pragma unroll
    for (int i = 0; i < 6; i++)
        wmma::fill_fragment(C[i], 0.0f);

    // loop for the 32 pixels in this warp
#pragma unroll
    for (int k0 = 0; k0 < 32; k0 += 16) {
        // pointer to the start of the 16-pixel batch for this warp
        const half* tilePtr = &RxLocal[startRow + k0][0];
#pragma unroll
        // load A and B matrices (3 each)
        for (int i = 0; i < 3; i++) {
            wmma::load_matrix_sync(A[i], tilePtr + (i * 16), sharedMemStride);
            wmma::load_matrix_sync(B[i], tilePtr + (i * 16), sharedMemStride);
        }

        // lower triangular + diagonal calculation (6 tiles)
        wmma::mma_sync(C[0], A[0], B[0], C[0]);
        wmma::mma_sync(C[1], A[1], B[0], C[1]);
        wmma::mma_sync(C[2], A[1], B[1], C[2]);
        wmma::mma_sync(C[3], A[2], B[0], C[3]);
        wmma::mma_sync(C[4], A[2], B[1], C[4]);
        wmma::mma_sync(C[5], A[2], B[2], C[5]);
    }

    // compute rx (vectorized 128-bit plus half2 for maximum efficiency)
    half8 rxVec[6];
#pragma unroll
    for (int i = 0; i < 6; i++) {
        half2* inPtr = reinterpret_cast<half2*>(&localVec8[i]);
        half2* resultPtr = reinterpret_cast<half2*>(&rxVec[i]);
#pragma unroll
        for (int j = 0; j < 4; j++)
            resultPtr[j] = __hmul2(inPtr[j], center);
    }

    // rx warp-level reduction
    // lane 0 will end up holding the sum of the entire warp
    half2* rxHalf2 = reinterpret_cast<half2*>(rxVec);
#pragma unroll
    for (int i = 0; i < 24; i++) {
        half2 sum = rxHalf2[i];
        for (int offset = 16; offset > 0; offset >>= 1) {
            int shflInt = __shfl_down_sync(0xFFFFFFFF, reinterpret_cast<int&>(sum), offset);
            sum = __hadd2(sum, reinterpret_cast<half2&>(shflInt));
        }
        rxHalf2[i] = sum;
    }
    __syncthreads(); // here (p = 7 version) it is needed!

    // store Rx to shared mem
    const int outputRowStart = warpId * 48;
    half* warpOutput = &RxLocal[outputRowStart][0];
    wmma::store_matrix_sync(warpOutput, C[0], sharedMemStride, wmma::mem_row_major);
    wmma::store_matrix_sync(warpOutput + 16 * sharedMemStride, C[1], sharedMemStride, wmma::mem_row_major);
    wmma::store_matrix_sync(warpOutput + 16 * sharedMemStride + 16, C[2], sharedMemStride, wmma::mem_row_major);
    wmma::store_matrix_sync(warpOutput + 32 * sharedMemStride, C[3], sharedMemStride, wmma::mem_row_major);
    wmma::store_matrix_sync(warpOutput + 32 * sharedMemStride + 16, C[4], sharedMemStride, wmma::mem_row_major);
    wmma::store_matrix_sync(warpOutput + 32 * sharedMemStride + 32, C[5], sharedMemStride, wmma::mem_row_major);

    // Store rx partials (vertical strip in padding column 48)
    if ((tid & 31) == 0) {
#pragma unroll
        for (int i = 0; i < 6; i++) {
            half* vecPtr = reinterpret_cast<half*>(&rxVec[i]);
#pragma unroll
            for (int k = 0; k < 8; k++)
                RxLocal[outputRowStart + i * 8 + k][48] = vecPtr[k];
        }
    }
    __syncthreads();

    // write Rx by everyone
    const int RxBaseIndex = (y * gridDim.x * 1176) + (blockIdx.x * 1176);
    for (int k = tid; k < 1176; k += 256) {
        const int2 coords = getPackedCoords(k);
        float sum = 0.0f;
#pragma unroll
        for (int w = 0; w < 8; w++)
            sum += __half2float(RxLocal[w * 48 + coords.x][coords.y]);
        Rx[RxBaseIndex + k] = sum;
    }

    // write rx by first 48 threads (2 warps)
    if (tid < 48) {
        const int rxBaseIndex = (y * gridDim.x * 48) + (blockIdx.x * 48);
        float sum = 0.0f;
#pragma unroll
        for (int w = 0; w < 8; w++)
            sum += __half2float(RxLocal[w * 48 + tid][48]);
        rx[rxBaseIndex + tid] = sum;
    }
}

__global__ void me_p9(const float* __restrict__ input, float* __restrict__ Rx, float* __restrict__ rx, const unsigned int width, const unsigned int height) {
    constexpr int sharedMemStride = 88; // 80 + 8 for padding to minimize bank conflicts (padding is CRITICAL for performance)

    // NOTE: 128 threads per block, 256 will overflow shared memory!
    const int tid = threadIdx.x;
    const int x = blockIdx.x * 128 + tid;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int warpId = tid / 32;
    const int startRow = warpId * 32;

    // shared memory for Rx, rx, scratch and all pixels utilized by the whole block
    __shared__ alignas(16) half RxLocal[128][sharedMemStride];
    __shared__ alignas(16) half blockValues[9][136];

    // custom fillBlockStrip only for p=9 version
    constexpr int totalPixels = 9 * 136;
    constexpr int colStep = 14;
    constexpr int rowStep = 2;
    const int baseGlobalCol = (int)(blockIdx.x * 128) - 4;
    const int baseGlobalRow = (int)(blockIdx.y * 1) - 4;
    int r = tid % 9;
    int c = tid / 9;
    int idx = tid;
    while (idx < totalPixels) {
        if (c < 136) {
            const int globalCol = clamp(baseGlobalCol + c, 0, (int)width - 1);
            const int globalRow = clamp(baseGlobalRow + r, 0, (int)height - 1);
            blockValues[r][c] = __float2half(input[globalCol * height + globalRow] * 0.00392156862f);
        }
        idx += 128;
        c += colStep;
        r += rowStep;
        if (r >= 9) {
            r -= 9;
            c += 1;
        }
    }
    __syncthreads();

    if (y >= height)
        return;

    // read the 9x9 window from shared memory
    half centerVal;
    half2 center;
    half8* localVec8 = reinterpret_cast<half8*>(&RxLocal[tid][0]);
    localVec8[9] = {}; // clear padding
    // if/else is better than always writing (even vectorized) to shared memory unnecessarily
    if (x < width) {
        load_neighbor_vec_p9(localVec8, blockValues, centerVal);
        center = __half2half2(centerVal);
    } else {
#pragma unroll
        for (int i = 0; i < 10; i++)
            localVec8[i] = {};
        center = __half2half2(blockValues[4][tid + 4]); // center pixel
    }

    // do not compute rx yet, compute Rx first
    // TENSOR CORE Rx ACCUMULATION
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> A[5];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> B[5];
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> C[15];
    // accumulators
#pragma unroll
    for (int i = 0; i < 15; i++)
        wmma::fill_fragment(C[i], 0.0f);
    // loop for the 32 pixels in this warp
#pragma unroll
    for (int k0 = 0; k0 < 32; k0 += 16) {
        // pointer to the start of the 16-pixel batch for this warp
        const half* tilePtr = &RxLocal[startRow + k0][0];
#pragma unroll
        // load A and B matrices (5 each)
        for (int i = 0; i < 5; i++) {
            wmma::load_matrix_sync(A[i], tilePtr + (i * 16), sharedMemStride);
            wmma::load_matrix_sync(B[i], tilePtr + (i * 16), sharedMemStride);
        }
        // lower triangular + diagonal calculation (15 tiles)
        wmma::mma_sync(C[0], A[0], B[0], C[0]);
        wmma::mma_sync(C[1], A[1], B[0], C[1]);
        wmma::mma_sync(C[2], A[1], B[1], C[2]);
        wmma::mma_sync(C[3], A[2], B[0], C[3]);
        wmma::mma_sync(C[4], A[2], B[1], C[4]);
        wmma::mma_sync(C[5], A[2], B[2], C[5]);
        wmma::mma_sync(C[6], A[3], B[0], C[6]);
        wmma::mma_sync(C[7], A[3], B[1], C[7]);
        wmma::mma_sync(C[8], A[3], B[2], C[8]);
        wmma::mma_sync(C[9], A[3], B[3], C[9]);
        wmma::mma_sync(C[10], A[4], B[0], C[10]);
        wmma::mma_sync(C[11], A[4], B[1], C[11]);
        wmma::mma_sync(C[12], A[4], B[2], C[12]);
        wmma::mma_sync(C[13], A[4], B[3], C[13]);
        wmma::mma_sync(C[14], A[4], B[4], C[14]);
    }

    // compute rx (vectorized 128-bit plus half2 for maximum efficiency)
    half8 rxVec[10];
#pragma unroll
    for (int i = 0; i < 10; i++) {
        half2* inPtr = reinterpret_cast<half2*>(&localVec8[i]);
        half2* resultPtr = reinterpret_cast<half2*>(&rxVec[i]);
#pragma unroll
        for (int j = 0; j < 4; j++)
            resultPtr[j] = __hmul2(inPtr[j], center);
    }
    // rx warp-level reduction
    // lane 0 will end up holding the sum of the entire warp
    half2* rxHalf2 = reinterpret_cast<half2*>(rxVec);
#pragma unroll
    for (int i = 0; i < 40; i++) {
        half2 sum = rxHalf2[i];
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            int shflInt = __shfl_down_sync(0xFFFFFFFF, reinterpret_cast<int&>(sum), offset);
            sum = __hadd2(sum, reinterpret_cast<half2&>(shflInt));
        }
        rxHalf2[i] = sum;
    }
    __syncthreads();

    // from here we have DIFFERENT STRATEGY only for p=9 version
    // STREAMING OUTPUT (flush Registers to Global in chunks)
    // why we do this? because we want to increase OCCUPANCY, less shared memory per block
    // because it cannot physically fit for p=9... so we do the writing to global in chunks
    const int warpWindowStart = warpId * 32;
    half* warpOutput = &RxLocal[warpWindowStart][0];
    const int RxBaseIndex = (y * gridDim.x * 3240) + (blockIdx.x * 3240);

    // Rx chunk 1: rows 0-31 (tiles C0, C1, C2)
    wmma::store_matrix_sync(warpOutput + (0 * sharedMemStride) + 0, C[0], sharedMemStride, wmma::mem_row_major);
    wmma::store_matrix_sync(warpOutput + (16 * sharedMemStride) + 0, C[1], sharedMemStride, wmma::mem_row_major);
    wmma::store_matrix_sync(warpOutput + (16 * sharedMemStride) + 16, C[2], sharedMemStride, wmma::mem_row_major);
    __syncthreads();
    RxStreamPass<0, 528, 0>(tid, RxLocal, Rx, RxBaseIndex);

    // Rx chunk 2: rows 32-47 (tiles C3, C4, C5)
    __syncthreads();
    wmma::store_matrix_sync(warpOutput + (0 * sharedMemStride) + 0, C[3], sharedMemStride, wmma::mem_row_major);
    wmma::store_matrix_sync(warpOutput + (0 * sharedMemStride) + 16, C[4], sharedMemStride, wmma::mem_row_major);
    wmma::store_matrix_sync(warpOutput + (0 * sharedMemStride) + 32, C[5], sharedMemStride, wmma::mem_row_major);
    __syncthreads();

    // write indices 528 to 1175
    RxStreamPass<528, 1176, 32>(tid, RxLocal, Rx, RxBaseIndex);
    __syncthreads();
    // Rx chunk 3: rows 48-63 (tiles C6, C7, C8, C9)
    wmma::store_matrix_sync(warpOutput + (0 * sharedMemStride) + 0, C[6], sharedMemStride, wmma::mem_row_major);
    wmma::store_matrix_sync(warpOutput + (0 * sharedMemStride) + 16, C[7], sharedMemStride, wmma::mem_row_major);
    wmma::store_matrix_sync(warpOutput + (0 * sharedMemStride) + 32, C[8], sharedMemStride, wmma::mem_row_major);
    wmma::store_matrix_sync(warpOutput + (0 * sharedMemStride) + 48, C[9], sharedMemStride, wmma::mem_row_major);
    __syncthreads();

    // write indices 1176 to 2079
    RxStreamPass<1176, 2080, 48>(tid, RxLocal, Rx, RxBaseIndex);
    __syncthreads();
    // Rx chunk 4: rows 64-79 (tiles C10 to C14)
    wmma::store_matrix_sync(warpOutput + (0 * sharedMemStride) + 0, C[10], sharedMemStride, wmma::mem_row_major);
    wmma::store_matrix_sync(warpOutput + (0 * sharedMemStride) + 16, C[11], sharedMemStride, wmma::mem_row_major);
    wmma::store_matrix_sync(warpOutput + (0 * sharedMemStride) + 32, C[12], sharedMemStride, wmma::mem_row_major);
    wmma::store_matrix_sync(warpOutput + (0 * sharedMemStride) + 48, C[13], sharedMemStride, wmma::mem_row_major);
    wmma::store_matrix_sync(warpOutput + (0 * sharedMemStride) + 64, C[14], sharedMemStride, wmma::mem_row_major);
    __syncthreads();

    // write indices 2080 to 3239
    RxStreamPass<2080, 3240, 64>(tid, RxLocal, Rx, RxBaseIndex);

    // rx CHUNKS streaming (3 passes here)
    // reuse the window to sum 'rx' across warps, rx has 80 elements
    // window has 32 rows, why not use column 80 for storage!
    const int rxBaseIndex = (y * gridDim.x * 80) + (blockIdx.x * 80);
    rxStreamPass<0, 32>(tid, warpWindowStart, RxLocal, rxVec, rx, rxBaseIndex);
    rxStreamPass<32, 32>(tid, warpWindowStart, RxLocal, rxVec, rx, rxBaseIndex);
    rxStreamPass<64, 16>(tid, warpWindowStart, RxLocal, rxVec, rx, rxBaseIndex);
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
    const int lane = tid & 31;
    const int warpId = tid >> 5;
    const int numWarps = (blockDim.x + 31) >> 5;

    // shared memory must match number of warps
    __shared__ float warpDot[32];
    __shared__ float warpU[32];
    __shared__ float warpZ[32];

    float localDot = 0.0f;
    float localU = 0.0f;
    float localZ = 0.0f;

    // if pointers are 16-byte aligned we can use vectorized loads to massively speedup the kernel
    const uintptr_t rawDots = reinterpret_cast<const uintptr_t>(partialDots);
    const uintptr_t rawU = reinterpret_cast<const uintptr_t>(partialNormU);
    const uintptr_t rawZ = reinterpret_cast<const uintptr_t>(partialNormZ);
    if (((rawDots | rawU | rawZ) & 0xF) == 0) {
        const float4* vecDots = reinterpret_cast<const float4*>(partialDots);
        const float4* vecU = reinterpret_cast<const float4*>(partialNormU);
        const float4* vecZ = reinterpret_cast<const float4*>(partialNormZ);

        const int vecLoopLimit = numBlocks >> 2;
        // vectorized grid stride Loop
        for (int i = tid; i < vecLoopLimit; i += blockDim.x) {
            const float4 d = vecDots[i];
            const float4 u = vecU[i];
            const float4 z = vecZ[i];
            localDot += d.x + d.y + d.z + d.w;
            localU += u.x + u.y + u.z + u.w;
            localZ += z.x + z.y + z.z + z.w;
        }
        // tail elements
        for (int i = (vecLoopLimit << 2) + tid; i < numBlocks; i += blockDim.x) {
            localDot += partialDots[i];
            localU += partialNormU[i];
            localZ += partialNormZ[i];
        }
    }

    // non vectorized path (alignment not met), fallback to scalar loads
    else {
        for (int i = tid; i < numBlocks; i += blockDim.x) {
            localDot += partialDots[i];
            localU += partialNormU[i];
            localZ += partialNormZ[i];
        }
    }

    // intra-warp reduction
#pragma unroll
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
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            localDot += __shfl_down_sync(0xFFFFFFFF, localDot, offset);
            localU += __shfl_down_sync(0xFFFFFFFF, localU, offset);
            localZ += __shfl_down_sync(0xFFFFFFFF, localZ, offset);
        }
        if (lane == 0) {
            const float normU = sqrtf(localU);
            const float normZ = sqrtf(localZ);
            result[0] = (normU > 1e-12f && normZ > 1e-12f) ? (localDot / (normU * normZ)) : 0.0f;
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