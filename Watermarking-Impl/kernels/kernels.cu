#include "kernels.cuh"
#include <cstdint>
#include <cub/cub.cuh>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <mma.h>

using namespace nvcuda;

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

// column version of the funnel loader (p=3)
__device__ void load_neighbor_row_funnel_p3_col(half& a, half& b, half& c, const half* rowBase, const int col) {
    const uint32_t shift = (col & 1) * 16;
    const uint32_t* ptr = reinterpret_cast<const uint32_t*>(&rowBase[col & ~1]);
    uint32_t p0 = __funnelshift_r(ptr[0], ptr[1], shift);
    uint32_t pF = ptr[1] >> shift;
    const half2 h0 = reinterpret_cast<half2&>(p0);
    const half2 hF = reinterpret_cast<half2&>(pF);
    a = h0.x;
    b = h0.y;
    c = hF.x;
}

// column version of the 3x3 helper vector loader
__device__ void load_neighbor_vec_p3_col(half8& dst, half& center, const half blockValues[3][258], const int col) {
    load_neighbor_row_funnel_p3_col(dst.a, dst.b, dst.c, blockValues[0], col);
    load_neighbor_row_funnel_p3_col(dst.d, center, dst.e, blockValues[1], col);
    load_neighbor_row_funnel_p3_col(dst.f, dst.g, dst.h, blockValues[2], col);
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

__device__ void load_neighbor_vec_p7(half8* dst, const half blockValues[7][134], half& center) {
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

__global__ void me_p3(const float* __restrict__ input, float* __restrict__ Rx, float* __restrict__ rx, const int width, const int height, const int totalBlocksY, const int taskTotal) {
    constexpr int inputStride = 40; // 20 floats -> 40 halves
    constexpr int outputStride = 20;

    const int tid = threadIdx.x;
    const int warpId = tid >> 5;
    const int startRow = warpId * 32;
    const int gridTotal = gridDim.x * gridDim.y;
    const int blockLinear = blockIdx.y * gridDim.x + blockIdx.x;

    __shared__ alignas(16) float RxLocal[128][20]; // note: for p=3 we process 256 pixels, but pack them into 128 vectors of size 2
    __shared__ alignas(16) half blockValues[3][258];
    __shared__ float rxStaging[4][8]; // 4 warps (128 threads) instead of 256
    __shared__ typename cub::WarpReduce<rxVecData<8>>::TempStorage temp_storage[4];

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_C;
    wmma::fill_fragment(acc_C, 0.0f);
    rxVecData<8> rxPersistent;
    rxPersistent.zero();

    // grid stride loop
    for (int taskIdx = blockLinear; taskIdx < taskTotal; taskIdx += gridTotal) {
        const int bx = taskIdx / totalBlocksY;
        const int by = taskIdx % totalBlocksY;

        if (bx >= width)
            continue;

        // all 256 threads help load the strip
        fillBlockStripVertical<3, 256>(blockValues, input, width, height, bx, by);
        __syncthreads();

        // load window (p=3 optimized)
        // ONLY for p=3 version: WMMA minimum size is 16x16 tensors, so we pack 2 pixels per thread to fill the tile
        // because the 3x3 neighborhood is small we would waste gpu resources if we only processed 1 pixel per thread (we would need to PAD 75% of tensor values!)
        // thread x packs pixel x (top) and pixel x+128 (bottom)
        if (tid < 128) {
            half8 vecTop, vecBot;
            half centerTop, centerBot;
            // load pixel A (tid)
            if ((by * 256 + tid) < height) {
                load_neighbor_row_funnel_p3(vecTop.a, vecTop.b, vecTop.c, blockValues[0]);
                load_neighbor_row_funnel_p3(vecTop.d, centerTop, vecTop.e, blockValues[1]);
                load_neighbor_row_funnel_p3(vecTop.f, vecTop.g, vecTop.h, blockValues[2]);
            } else {
                vecTop = {};
                centerTop = blockValues[1][tid + 1];
            }

            // load pixel B (tid + 128)
            if ((bx * 256 + tid + 128) < width) {
                load_neighbor_vec_p3_col(vecBot, centerBot, blockValues, tid + 128);
            } else {
                vecBot = {};
                centerBot = blockValues[1][tid + 128 + 1];
            }

            // rx accumulation (do both pixels)
            const half2 center2Top = __half2half2(centerTop);
            const half2 center2Bot = __half2half2(centerBot);
            const half2* ptrTop = reinterpret_cast<half2*>(&vecTop);
            const half2* ptrBot = reinterpret_cast<half2*>(&vecBot);

#pragma unroll
            for (int j = 0; j < 4; j++) {
                // pixel A
                float2 res = __half22float2(__hmul2(ptrTop[j], center2Top));
                rxPersistent.vals[j * 2] += res.x;
                rxPersistent.vals[j * 2 + 1] += res.y;
                // pixel B
                res = __half22float2(__hmul2(ptrBot[j], center2Bot));
                rxPersistent.vals[j * 2] += res.x;
                rxPersistent.vals[j * 2 + 1] += res.y;
            }

            // Rx accumulation (Tensor Cores) PACKED (2 pixels)
            // rowPtrVec[0] = top, rowPtrVec[1] = bottom
            half* rowPtr = reinterpret_cast<half*>(&RxLocal[tid][0]);
            half8* rowPtrVec = reinterpret_cast<half8*>(rowPtr);
            rowPtrVec[0] = vecTop;
            rowPtrVec[1] = vecBot; // here is our "packing trick" (one fully filled WMMA tile (16)
        }
        __syncthreads();

        // for WMMA: only first 128 threads (4 warps) need to run
        if (tid < 128) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> A;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> B;
#pragma unroll
            for (int k0 = 0; k0 < 32; k0 += 16) {
                const half* tilePtr = reinterpret_cast<half*>(&RxLocal[startRow + k0][0]);
                wmma::load_matrix_sync(A, tilePtr, inputStride);
                wmma::load_matrix_sync(B, tilePtr, inputStride);
                wmma::mma_sync(acc_C, A, B, acc_C);
            }
        }
        __syncthreads();
    }

    // write to global memory
    if (tid < 128) {
        float* warpOutput = &RxLocal[warpId * 32][0];
        wmma::store_matrix_sync(warpOutput, acc_C, outputStride, wmma::mem_row_major);
        // write rx (cub)
        writeRxVec<8>(rx, rxPersistent, temp_storage[warpId], &rxStaging[warpId][0]);
    }
    __syncthreads();

    // write Rx (top left + bottom right)
    if (tid < 36) {
        const int2 coords = getPackedCoords(tid);
        float sum = 0.0f;
#pragma unroll
        for (int w = 0; w < 4; w++) {
            // tile 1 (top left): pixel A results
            sum += RxLocal[w * 32 + coords.x][coords.y];
            // tile 4 (bottom right): pixel B results (offset: rows+8, cols+8)
            sum += RxLocal[w * 32 + 8 + coords.x][8 + coords.y];
        }
        atomicAdd(&Rx[tid], sum);
    }
}

__global__ void me_p5(const float* __restrict__ input, float* __restrict__ Rx, float* __restrict__ rx, const int width, const int height, const int totalBlocksY, const int taskTotal) {
    constexpr int inputStride = 72;  // WMMA stride: 36 floats = 72 halves
    constexpr int outputStride = 36; // output stride: match allocation (36) to avoid wrapping rows

    const int tid = threadIdx.x;
    const int warpId = tid >> 5;
    const int startRow = warpId * 32;
    const int gridTotal = gridDim.x * gridDim.y;
    const int blockLinear = blockIdx.y * gridDim.x + blockIdx.x;

    __shared__ alignas(16) float RxLocal[256][36];
    __shared__ alignas(16) half blockValues[5][260];
    __shared__ float rxStaging[8][24];
    __shared__ typename cub::WarpReduce<rxVecData<24>>::TempStorage temp_storage[8];

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_C00, acc_C10, acc_C11;
    wmma::fill_fragment(acc_C00, 0.0f);
    wmma::fill_fragment(acc_C10, 0.0f);
    wmma::fill_fragment(acc_C11, 0.0f);
    rxVecData<24> rxPersistent;
    rxPersistent.zero();

    // grid stride loop
    for (int taskIdx = blockLinear; taskIdx < taskTotal; taskIdx += gridTotal) {
        const int bx = taskIdx / totalBlocksY;
        const int by = taskIdx % totalBlocksY;

        if (bx >= width)
            continue;

        fillBlockStripVertical<5, 256>(blockValues, input, width, height, bx, by);
        __syncthreads();

        // load window
        half centerVal;
        half8 localVec8[3];

        if ((by * 256 + tid) < height) {
            load_neighbor_vec_p5(localVec8, blockValues, centerVal);
        } else {
#pragma unroll
            for (int i = 0; i < 3; i++)
                localVec8[i] = {};
            centerVal = blockValues[2][tid + 2];
        }
        half2 center = __half2half2(centerVal);

        // accumulate rx
#pragma unroll
        for (int i = 0; i < 3; i++) {
            half2* inPtr = reinterpret_cast<half2*>(&localVec8[i]);
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const float2 res = __half22float2(__hmul2(inPtr[j], center));
                rxPersistent.vals[i * 8 + j * 2 + 0] += res.x;
                rxPersistent.vals[i * 8 + j * 2 + 1] += res.y;
            }
        }

        half* rowPtr = reinterpret_cast<half*>(&RxLocal[tid][0]);
        half8* rowPtrVec = reinterpret_cast<half8*>(rowPtr);
        // stride is 36 floats -> 72 halves, we must zero the 4th vector (indices 24-31)
        // else the tile calculation will read garbage
#pragma unroll
        for (int i = 0; i < 3; i++)
            rowPtrVec[i] = localVec8[i];
        rowPtrVec[3] = {};
        __syncthreads();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> A_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> B_frag;
#pragma unroll
        for (int k0 = 0; k0 < 32; k0 += 16) {
            const half* tilePtr = reinterpret_cast<half*>(&RxLocal[startRow + k0][0]);
            wmma::load_matrix_sync(A_frag, tilePtr, inputStride);
            wmma::load_matrix_sync(B_frag, tilePtr, inputStride);
            wmma::mma_sync(acc_C00, A_frag, B_frag, acc_C00);
            wmma::load_matrix_sync(A_frag, tilePtr + 16, inputStride);
            wmma::mma_sync(acc_C10, A_frag, B_frag, acc_C10);
            wmma::load_matrix_sync(B_frag, tilePtr + 16, inputStride);
            wmma::mma_sync(acc_C11, A_frag, B_frag, acc_C11);
        }
        __syncthreads();
    }

    // write to global memory
    float* warpOutput = &RxLocal[warpId * 32][0];
    wmma::store_matrix_sync(warpOutput, acc_C00, outputStride, wmma::mem_row_major);
    wmma::store_matrix_sync(warpOutput + 16 * outputStride, acc_C10, outputStride, wmma::mem_row_major);
    wmma::store_matrix_sync(warpOutput + 16 * outputStride + 16, acc_C11, outputStride, wmma::mem_row_major);
    // write rx (cub) before the barrier to hide latency
    writeRxVec<24>(rx, rxPersistent, temp_storage[warpId], &rxStaging[warpId][0]);
    __syncthreads();

    // write Rx lambda
    auto writeRx = [&](const int k) {
        const int2 coords = getPackedCoords(k);
        float sum = 0.0f;
#pragma unroll
        for (int w = 0; w < 8; w++)
            sum += RxLocal[w * 32 + coords.x][coords.y];
        atomicAdd(&Rx[k], sum);
    };

    // pass 1: [0, 255]
    writeRx(tid);

    // pass 2: [256, 299]
    if (tid < 44) {
        writeRx(tid + 256);
    }
}

__global__ void me_p7(const float* __restrict__ input, float* __restrict__ Rx, float* __restrict__ rx, const int width, const int height, const int totalBlocksY, const int taskTotal) {
    constexpr int inputStride = 112;
    constexpr int outputStride = 56;

    const int tid = threadIdx.x;
    const int warpId = tid >> 5;
    const int startRow = warpId * 32;
    const int gridTotal = gridDim.x * gridDim.y;
    const int blockLinear = blockIdx.y * gridDim.x + blockIdx.x;

    __shared__ alignas(16) float RxLocal[192][56];
    __shared__ alignas(16) half blockValues[7][134];
    __shared__ float rxStaging[4][48];
    __shared__ typename cub::WarpReduce<rxVecData<48>>::TempStorage temp_storage[4];

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_Rx[6];
#pragma unroll
    for (int i = 0; i < 6; i++)
        wmma::fill_fragment(acc_Rx[i], 0.0f);
    rxVecData<48> rxPersistent;
    rxPersistent.zero();

    // grid stride loop
    for (int taskIdx = blockLinear; taskIdx < taskTotal; taskIdx += gridTotal) {
        const int bx = taskIdx / totalBlocksY;
        const int by = taskIdx % totalBlocksY;

        if (bx >= width)
            continue;

        fillBlockStripVertical<7, 128>(blockValues, input, width, height, bx, by);
        __syncthreads();

        if (by * blockDim.y + threadIdx.y >= height)
            continue;

        half centerVal;
        half8 localVec8[6];

        // load window
        if ((by * 128 + tid) < height) {
            load_neighbor_vec_p7(localVec8, blockValues, centerVal);
        } else {
#pragma unroll
            for (int i = 0; i < 6; ++i)
                localVec8[i] = {};
            centerVal = blockValues[3][tid + 3];
        }
        half2 center = __half2half2(centerVal);

        // accumulate rx
#pragma unroll
        for (int i = 0; i < 6; i++) {
            half2* inPtr = reinterpret_cast<half2*>(&localVec8[i]);
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const float2 res = __half22float2(__hmul2(inPtr[j], center));
                rxPersistent.vals[i * 8 + j * 2 + 0] += res.x;
                rxPersistent.vals[i * 8 + j * 2 + 1] += res.y;
            }
        }

        // accumulate Rx (Tensor Cores)
        half* rowPtr = reinterpret_cast<half*>(&RxLocal[tid][0]);
        half8* rowPtrVec = reinterpret_cast<half8*>(rowPtr);
#pragma unroll
        for (int i = 0; i < 6; i++)
            rowPtrVec[i] = localVec8[i];
        rowPtrVec[6] = {}; // zero padding
        __syncthreads();
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> A[3];
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> B[3];

#pragma unroll
        for (int k0 = 0; k0 < 32; k0 += 16) {
            const half* tilePtr = reinterpret_cast<half*>(&RxLocal[startRow + k0][0]);
#pragma unroll
            for (int i = 0; i < 3; i++) {
                wmma::load_matrix_sync(A[i], tilePtr + (i * 16), inputStride);
                wmma::load_matrix_sync(B[i], tilePtr + (i * 16), inputStride);
            }
            wmma::mma_sync(acc_Rx[0], A[0], B[0], acc_Rx[0]);
            wmma::mma_sync(acc_Rx[1], A[1], B[0], acc_Rx[1]);
            wmma::mma_sync(acc_Rx[2], A[1], B[1], acc_Rx[2]);
            wmma::mma_sync(acc_Rx[3], A[2], B[0], acc_Rx[3]);
            wmma::mma_sync(acc_Rx[4], A[2], B[1], acc_Rx[4]);
            wmma::mma_sync(acc_Rx[5], A[2], B[2], acc_Rx[5]);
        }
        __syncthreads();
    }

    // write to global memory
    float* warpOutput = &RxLocal[warpId * 48][0];
    wmma::store_matrix_sync(warpOutput, acc_Rx[0], outputStride, wmma::mem_row_major);
    wmma::store_matrix_sync(warpOutput + 16 * outputStride, acc_Rx[1], outputStride, wmma::mem_row_major);
    wmma::store_matrix_sync(warpOutput + 16 * outputStride + 16, acc_Rx[2], outputStride, wmma::mem_row_major);
    wmma::store_matrix_sync(warpOutput + 32 * outputStride, acc_Rx[3], outputStride, wmma::mem_row_major);
    wmma::store_matrix_sync(warpOutput + 32 * outputStride + 16, acc_Rx[4], outputStride, wmma::mem_row_major);
    wmma::store_matrix_sync(warpOutput + 32 * outputStride + 32, acc_Rx[5], outputStride, wmma::mem_row_major);
    // write rx (cub) before the barrier to hide latency
    writeRxVec<48>(rx, rxPersistent, temp_storage[warpId], &rxStaging[warpId][0]);
    __syncthreads();

    // write Rx, loop stride 128, sum over 4 warps
    for (int k = tid; k < 1176; k += 128) {
        const int2 coords = getPackedCoords(k);
        float sum = 0.0f;
#pragma unroll
        for (int w = 0; w < 4; w++)
            sum += RxLocal[w * 48 + coords.x][coords.y];
        atomicAdd(&Rx[k], sum);
    }
}

__global__ void me_p9(const float* __restrict__ input, float* __restrict__ Rx, float* __restrict__ rx, const int width, const int height, const int totalBlocksY, const int taskTotal) {
    constexpr int INPUT_STRIDE = 176; // WMMA stride is 176 (halves)
    constexpr int outputStride = 88;

    const int tid = threadIdx.x;
    const int warpId = tid >> 5;
    const int startRow = warpId * 32;
    const int gridTotal = gridDim.x * gridDim.y;
    const int blockLinear = blockIdx.y * gridDim.x + blockIdx.x;

    __shared__ alignas(16) float RxLocal[128][88];
    __shared__ alignas(16) half blockValues[9][136];
    __shared__ typename cub::WarpReduce<rxVecData<80>>::TempStorage temp_storage[4];

    // Init Accumulators (Float)
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_Rx[15];
#pragma unroll
    for (int i = 0; i < 15; i++)
        wmma::fill_fragment(acc_Rx[i], 0.0f);
    rxVecData<80> rxPersistent;
    rxPersistent.zero();

    // grid stride loop
    for (int taskIdx = blockLinear; taskIdx < taskTotal; taskIdx += gridTotal) {
        const int bx = taskIdx / totalBlocksY;
        const int by = taskIdx % totalBlocksY;

        if (bx >= width)
            continue;

        fillBlockStripVertical<9, 128>(blockValues, input, width, height, bx, by);
        __syncthreads();

        if (by * blockDim.y + threadIdx.y >= height)
            continue;

        // load window
        half centerVal;
        half8 localVec8[10];
        if ((by * 128 + tid) < height) {
            load_neighbor_vec_p9(localVec8, blockValues, centerVal);
        } else {
#pragma unroll
            for (int i = 0; i < 10; ++i)
                localVec8[i] = {};
            centerVal = blockValues[4][tid + 4];
        }
        half2 center = __half2half2(centerVal);

        // accumulate rx
#pragma unroll
        for (int i = 0; i < 10; i++) {
            half2* inPtr = reinterpret_cast<half2*>(&localVec8[i]);
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const float2 res = __half22float2(__hmul2(inPtr[j], center));
                rxPersistent.vals[i * 8 + j * 2 + 0] += res.x;
                rxPersistent.vals[i * 8 + j * 2 + 1] += res.y;
            }
        }

        half8* shmemPtr = reinterpret_cast<half8*>(&RxLocal[tid][0]);
#pragma unroll
        for (int i = 0; i < 10; i++)
            shmemPtr[i] = localVec8[i];
        shmemPtr[10] = {}; // zero padding
        __syncthreads();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> A[5];
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> B[5];

        const half* tilePtr = reinterpret_cast<half*>(&RxLocal[startRow][0]);
#pragma unroll
        for (int k0 = 0; k0 < 32; k0 += 16) {
            const half* currTile = tilePtr + (k0 * INPUT_STRIDE);
#pragma unroll
            for (int i = 0; i < 5; i++) {
                wmma::load_matrix_sync(A[i], currTile + (i * 16), INPUT_STRIDE);
                wmma::load_matrix_sync(B[i], currTile + (i * 16), INPUT_STRIDE);
            }
            wmma::mma_sync(acc_Rx[0], A[0], B[0], acc_Rx[0]);
            wmma::mma_sync(acc_Rx[1], A[1], B[0], acc_Rx[1]);
            wmma::mma_sync(acc_Rx[2], A[1], B[1], acc_Rx[2]);
            wmma::mma_sync(acc_Rx[3], A[2], B[0], acc_Rx[3]);
            wmma::mma_sync(acc_Rx[4], A[2], B[1], acc_Rx[4]);
            wmma::mma_sync(acc_Rx[5], A[2], B[2], acc_Rx[5]);
            wmma::mma_sync(acc_Rx[6], A[3], B[0], acc_Rx[6]);
            wmma::mma_sync(acc_Rx[7], A[3], B[1], acc_Rx[7]);
            wmma::mma_sync(acc_Rx[8], A[3], B[2], acc_Rx[8]);
            wmma::mma_sync(acc_Rx[9], A[3], B[3], acc_Rx[9]);
            wmma::mma_sync(acc_Rx[10], A[4], B[0], acc_Rx[10]);
            wmma::mma_sync(acc_Rx[11], A[4], B[1], acc_Rx[11]);
            wmma::mma_sync(acc_Rx[12], A[4], B[2], acc_Rx[12]);
            wmma::mma_sync(acc_Rx[13], A[4], B[3], acc_Rx[13]);
            wmma::mma_sync(acc_Rx[14], A[4], B[4], acc_Rx[14]);
        }
        __syncthreads();
    }

    // write to global memory
    float* warpOutput = &RxLocal[warpId * 32][0];

    // write Rx chunked
    // chunk 1
    wmma::store_matrix_sync(warpOutput + (0), acc_Rx[0], outputStride, wmma::mem_row_major);
    wmma::store_matrix_sync(warpOutput + (16 * outputStride), acc_Rx[1], outputStride, wmma::mem_row_major);
    wmma::store_matrix_sync(warpOutput + (16 * outputStride) + 16, acc_Rx[2], outputStride, wmma::mem_row_major);
    __syncthreads();
    RxStreamPass<0, 528, 0>(tid, RxLocal, Rx);
    __syncthreads();

    // chunk 2
    wmma::store_matrix_sync(warpOutput, acc_Rx[3], outputStride, wmma::mem_row_major);
    wmma::store_matrix_sync(warpOutput + 16, acc_Rx[4], outputStride, wmma::mem_row_major);
    wmma::store_matrix_sync(warpOutput + 32, acc_Rx[5], outputStride, wmma::mem_row_major);
    __syncthreads();
    RxStreamPass<528, 1176, 32>(tid, RxLocal, Rx);
    __syncthreads();

    // chunk 3
    wmma::store_matrix_sync(warpOutput, acc_Rx[6], outputStride, wmma::mem_row_major);
    wmma::store_matrix_sync(warpOutput + 16, acc_Rx[7], outputStride, wmma::mem_row_major);
    wmma::store_matrix_sync(warpOutput + 32, acc_Rx[8], outputStride, wmma::mem_row_major);
    wmma::store_matrix_sync(warpOutput + 48, acc_Rx[9], outputStride, wmma::mem_row_major);
    __syncthreads();
    RxStreamPass<1176, 2080, 48>(tid, RxLocal, Rx);
    __syncthreads();

    // chunk 4
    wmma::store_matrix_sync(warpOutput, acc_Rx[10], outputStride, wmma::mem_row_major);
    wmma::store_matrix_sync(warpOutput + 16, acc_Rx[11], outputStride, wmma::mem_row_major);
    wmma::store_matrix_sync(warpOutput + 32, acc_Rx[12], outputStride, wmma::mem_row_major);
    wmma::store_matrix_sync(warpOutput + 48, acc_Rx[13], outputStride, wmma::mem_row_major);
    wmma::store_matrix_sync(warpOutput + 64, acc_Rx[14], outputStride, wmma::mem_row_major);
    __syncthreads();
    RxStreamPass<2080, 3240, 64>(tid, RxLocal, Rx);

    // write rx (cub) after the barrier, we can't interleave (we are dangerously close to max limit, plus register pressure is already high)
    __syncthreads();
    writeRxVec<80>(rx, rxPersistent, temp_storage[warpId], warpOutput);
}

__global__ void compute_u_and_sumsq(const float* __restrict__ mask, const float* __restrict__ w, float* __restrict__ u, float* __restrict__ globalSumSq, const int N) {
    constexpr int blockSize = 768;

    // setup CUB for block reduction
    using BlockReduceT = cub::BlockReduce<float, blockSize>;
    __shared__ typename BlockReduceT::TempStorage temp_storage;

    const int tid = threadIdx.x;
    const int gridSize = blockDim.x * gridDim.x;

    // fuse calculation of u and local sum of squares
    float threadSumSq = 0.0f;
    int idx = blockIdx.x * blockDim.x + tid;

    // grid stride loop
    while (idx < N) {
        const float uVal = mask[idx] * w[idx];
        u[idx] = uVal;
        threadSumSq += uVal * uVal;
        idx += gridSize;
    }

    // cub block reduction of the sum of squares
    const float blockTotalSq = BlockReduceT(temp_storage).Sum(threadSumSq);
    // final global atomic add, only thread 0 of the block does this, we want to minimize atomics as much as possible
    if (tid == 0)
        atomicAdd(globalSumSq, blockTotalSq);
}

__global__ void apply_watermark_fused(const float* __restrict__ input, const float* __restrict__ u, const float* __restrict__ sumSqPtr, uint8_t* __restrict__ output, const float strengthNumerator,
                                      const int planeElements, const int numChannels) {
    const float uSumSquared = *sumSqPtr; // read the precomputed sum of squares from global memory (all threads read the same value, it is cached)
    const float strength = uSumSquared > 1e-12f ? strengthNumerator * rsqrtf(uSumSquared) : 0.0f;
    // grid stride loop over the PLANE (HxW) only (if 1 channel then it's the whole image)
    const int gridSize = blockDim.x * gridDim.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < planeElements) {
        const float uStr = u[idx] * strength;
        // channel loop, for grayscale it runs once, for RGB it runs 3 times
        for (int c = 0; c < numChannels; c++) {
            const int pixelIdx = idx + (c * planeElements);
            const float outputVal = clamp(input[pixelIdx] + uStr + 0.5f, 0.0f, 255.0f); // round (trick + 0.5 for truncation) + clamp
            output[pixelIdx] = (uint8_t)outputVal;                                      // cast to uint8
        }
        idx += gridSize;
    }
}

__global__ void calculate_partial_correlation(const float* __restrict__ e_u, const float* __restrict__ e_z, float* __restrict__ partialDots, float* __restrict__ partialNormU,
                                              float* __restrict__ partialNormZ, const unsigned int size) {
    constexpr int blockSize = 768;

    // we can use CUB to reduce with warp shuffles and reduce the boilerplate
    using BlockReduceT = cub::BlockReduce<CorrelationData, blockSize>;
    __shared__ typename BlockReduceT::TempStorage temp_storage;

    const int tid = threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    float sumDot = 0.0f;
    float sumNormU = 0.0f;
    float sumNormZ = 0.0f;

    // grid stride loop
    int idx = blockIdx.x * blockDim.x + tid;
    while (idx < size) {
        const float a = e_u[idx];
        const float b = e_z[idx];
        sumDot += a * b;
        sumNormU += a * a;
        sumNormZ += b * b;
        idx += stride;
    }

    // use CUB to reduce the sums within the block, each thread contributes its local sum and we get a block level sum at the end
    const CorrelationData threadData = {sumDot, sumNormU, sumNormZ};
    const CorrelationData blockSum = BlockReduceT(temp_storage).Sum(threadData);

    // thread 0 writes the result
    if (tid == 0) {
        partialDots[blockIdx.x] = blockSum.dot;
        partialNormU[blockIdx.x] = blockSum.normU;
        partialNormZ[blockIdx.x] = blockSum.normZ;
    }
}

__global__ void calculate_final_correlation(const float* __restrict__ partialDots, const float* __restrict__ partialNormU, const float* __restrict__ partialNormZ, float* __restrict__ result,
                                            const unsigned int numBlocks) {
    constexpr int blockSize = 1024;

    // we can use CUB to reduce with warp shuffles and reduce the boilerplate
    using BlockReduceT = cub::BlockReduce<CorrelationData, blockSize>;
    __shared__ typename BlockReduceT::TempStorage temp_storage;

    const int tid = threadIdx.x;

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
    } else {
        // non vectorized path (alignment not met), fallback to scalar loads
        for (int i = tid; i < numBlocks; i += blockDim.x) {
            localDot += partialDots[i];
            localU += partialNormU[i];
            localZ += partialNormZ[i];
        }
    }

    // use cub to reduce the block sums
    const CorrelationData threadData = {localDot, localU, localZ};
    const CorrelationData blockSum = BlockReduceT(temp_storage).Sum(threadData);

    // final math and write by first thread
    if (tid == 0) {
        const float normU = sqrtf(blockSum.normU);
        const float normZ = sqrtf(blockSum.normZ);
        result[0] = (normU > 1e-12f && normZ > 1e-12f) ? (blockSum.dot / (normU * normZ)) : 0.0f;
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