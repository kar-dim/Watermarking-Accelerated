#include "kernels.cuh"
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <mma.h>

using namespace nvcuda;

// maps a linear index k to(row, col) coordinates for a packed lower triangular matrix
__device__ int2 getPackedCoords(const int k) {
    // inverse triangular number formula: r = floor((sqrt(1 + 8k) - 1) / 2)
    const int r = __float2int_rd(0.5f * (sqrtf(1.0f + 8.0f * k) - 1.0f));
    const int c = k - (r * (r + 1)) / 2;
    return make_int2(r, c);
}

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
    load_neighbor_row_funnel_p7(rows[0], blockValues[0]);
    load_neighbor_row_funnel_p7(rows[1], blockValues[1]);
    load_neighbor_row_funnel_p7(rows[2], blockValues[2]);
    load_neighbor_row_funnel_p7(rows[3], blockValues[3]);
    load_neighbor_row_funnel_p7(rows[4], blockValues[4]);
    load_neighbor_row_funnel_p7(rows[5], blockValues[5]);
    load_neighbor_row_funnel_p7(rows[6], blockValues[6]);
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

        // lower triangle calculation (6 tiles)
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

__global__ void cholesky_solver_p7(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ X, int* __restrict__ stopFlag) {
    const int laneId = threadIdx.x;

    // volatile to ensure visibility (because we use the faster __syncwarp)
    __shared__ volatile float sA[48][48 + 1]; // +1 to avoid bank conflicts
    __shared__ volatile float sB[48];

    // cooperative load (1176 elements -> 48x48 Shared)
    // check if A, B, and X are 16-byte aligned for vectorized loads
    const bool isAligned = ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B) | reinterpret_cast<uintptr_t>(X)) & 0xF) == 0;

    // if aligned: 1176 floats = 294 float4
    if (isAligned) {
        const float4* vecA = reinterpret_cast<const float4*>(A);
        // loop over 294 vectors
        for (int k = laneId; k < 294; k += 32) {
            float4 v = vecA[k];
            // unpack vector into Shared Memory
            const int baseIdx = k * 4;
            const int2 c0 = getPackedCoords(baseIdx + 0);
            sA[c0.x][c0.y] = v.x;
            const int2 c1 = getPackedCoords(baseIdx + 1);
            sA[c1.x][c1.y] = v.y;
            const int2 c2 = getPackedCoords(baseIdx + 2);
            sA[c2.x][c2.y] = v.z;
            const int2 c3 = getPackedCoords(baseIdx + 3);
            sA[c3.x][c3.y] = v.w;
        }
        // scalar path
    } else {
        for (int k = laneId; k < 1176; k += 32) {
            int2 c = getPackedCoords(k);
            sA[c.x][c.y] = A[k];
        }
    }

    // if aligned: 48 floats = 12 float4
    if (isAligned) {
        const float4* vecB = reinterpret_cast<const float4*>(B);
        if (laneId < 12) {
            const float4 v = vecB[laneId];
            sB[laneId * 4 + 0] = v.x;
            sB[laneId * 4 + 1] = v.y;
            sB[laneId * 4 + 2] = v.z;
            sB[laneId * 4 + 3] = v.w;
        }
        // scalar path
    } else {
        for (int k = laneId; k < 48; k += 32)
            sB[k] = B[k];
    }

    // initialize stop flag
    if (laneId == 0)
        *stopFlag = 0;
    __syncwarp();

    // in-place Cholesky Decomposition
    for (int k = 0; k < 48; k++) {
        // check diagonal and calculate sqrt
        float diag = sA[k][k];
        int abortFlag = 0;
        float invDiag = 0.0f;
        if (laneId == 0) {
            if (diag <= 1e-12f) {
                *stopFlag = 1; // write by 1 thread only
                abortFlag = 1;
            } else {
                invDiag = rsqrtf(diag);
                sA[k][k] = invDiag;
            }
        }

        // broadcast abort
        int abort_warp = __shfl_sync(0xFFFFFFFF, abortFlag, 0);
        if (abort_warp)
            return;

        // Broadcast L_kk
        float L_kk_inv = __shfl_sync(0xFFFFFFFF, invDiag, 0);

        for (int i = k + 1 + laneId; i < 48; i += 32)
            sA[i][k] = sA[i][k] * L_kk_inv;
        __syncwarp();

        // update trailing matrix
        for (int j = k + 1; j < 48; j += 2) {
            float L_jk0 = sA[j][k];
            float L_jk1 = (j + 1 < 48) ? sA[j + 1][k] : 0.0f;
            for (int i = j + laneId; i < 48; i += 32) {
                float Lik = sA[i][k];
                sA[i][j] = sA[i][j] - (Lik * L_jk0);
                if (j + 1 < 48 && i >= j + 1)
                    sA[i][j + 1] = sA[i][j + 1] - (Lik * L_jk1);
            }
        }
        __syncwarp();
    }

    // forward Substitution (solve L * y = b)
    for (int k = 0; k < 48; k++) {
        float val = sB[k];
        if (laneId == 0) {
            val *= sA[k][k];
            sB[k] = val;
        }
        float y_k = __shfl_sync(0xFFFFFFFF, val, 0);
        for (int i = k + 1 + laneId; i < 48; i += 32)
            sB[i] = sB[i] - (sA[i][k] * y_k);
        __syncwarp();
    }

    // backward Substitution (solve L^T * x = y)
    // solving U * x = y where U = L^T
    for (int k = 47; k >= 0; k--) {
        float val = sB[k];
        if (laneId == 0) {
            val *= sA[k][k];
            sB[k] = val;
        }
        float x_k = __shfl_sync(0xFFFFFFFF, val, 0);
        for (int i = laneId; i < k; i += 32)
            sB[i] = sB[i] - (sA[k][i] * x_k);
        __syncwarp();
    }

    // write Result
    if (isAligned) {
        float4* vecX = reinterpret_cast<float4*>(X);
        const float4* sbVec = (const float4*)((float*)sB);
        if (laneId < 12)
            vecX[laneId] = sbVec[laneId];
    } else {
        for (int k = laneId; k < 48; k += 32)
            X[k] = sB[k];
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