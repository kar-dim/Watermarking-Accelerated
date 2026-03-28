#include "kernels.cuh"
#include <cstdint>
#include <cub/cub.cuh>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <mma.h>

using namespace nvcuda;

__global__ void me_p3(const float* __restrict__ input, uint64_t* __restrict__ Rx, uint64_t* __restrict__ rx, const int width, const int height, const int totalBlocksY, const int taskTotal) {
    constexpr int IN_STRIDE = 40; // 20 floats -> 40 halves
    constexpr int OUT_STRIDE = 20;

    const int tid = threadIdx.x;
    const int warpId = tid >> 5;
    const int startRow = warpId * 32;
    const int gridTotal = gridDim.x * gridDim.y;
    const int blockLinear = blockIdx.y * gridDim.x + blockIdx.x;

    __shared__ alignas(16) float RxLocal[256][OUT_STRIDE]; // note: for p=3 we process 512 pixels, but pack them into 256 vectors of size 2
    __shared__ alignas(16) half blockValues[3][514];
    __shared__ alignas(16) float rxStaging[8][8]; // 8 warps
    __shared__ typename cub::WarpReduce<rxVecData<8>>::TempStorage temp_storage[8];

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
        fillBlockStripVertical<3, 512>(blockValues, input, width, height, bx, by);
        __syncthreads();

        // load window (p=3 optimized)
        // ONLY for p=3 version: thread 'tid' packs pixel 'tid' (top) AND pixel 'tid+256' (bottom)
        half8 vecTop, vecBot;
        half centerTop, centerBot;
        // load pixel A (tid)
        if ((by * 512 + tid) < height) {
            load_neighbor_vec<3>(&vecTop, blockValues, centerTop, tid);
        } else {
            vecTop = {};
            centerTop = blockValues[1][tid + 1];
        }

        // load pixel B (tid + 256)
        if ((by * 512 + tid + 256) < height) {
            load_neighbor_vec<3>(&vecBot, blockValues, centerBot, tid + 256);
        } else {
            vecBot = {};
            centerBot = blockValues[1][tid + 128 + 1];
        }

        // rx accumulation (do both pixels)
        accumulateRxVec<1>(&vecTop, rxPersistent.vals, __half2float(centerTop));
        accumulateRxVec<1>(&vecBot, rxPersistent.vals, __half2float(centerBot));

        // Rx accumulation (Tensor Cores) PACKED (2 pixels)
        // rowPtrVec[0] = top, rowPtrVec[1] = bottom
        half* rowPtr = reinterpret_cast<half*>(&RxLocal[tid][0]);
        half8* rowPtrVec = reinterpret_cast<half8*>(rowPtr);
        rowPtrVec[0] = vecTop;
        rowPtrVec[1] = vecBot; // here is our "packing trick" (one fully filled WMMA tile (16)
        __syncthreads();

        // all 256 threads do WMMA now
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> A;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> B;
#pragma unroll
        for (int k0 = 0; k0 < 32; k0 += 16) {
            const half* tilePtr = reinterpret_cast<half*>(&RxLocal[startRow + k0][0]);
            // load + compute directly
            wmma::load_matrix_sync(A, tilePtr, IN_STRIDE);
            wmma::load_matrix_sync(B, tilePtr, IN_STRIDE);
            wmma::mma_sync(acc_C, A, B, acc_C);
        }
        __syncthreads();
    }

    // write to global memory
    float* warpOutput = &RxLocal[warpId * 32][0];
    wmma::store_matrix_sync(warpOutput, acc_C, OUT_STRIDE, wmma::mem_row_major);
    // write rx (cub)
    writeRxVec<8>(rx, rxPersistent, temp_storage[warpId], &rxStaging[warpId][0]);
    __syncthreads();

    // write Rx (top left + bottom right)
    if (tid < 36) {
        const int2 coords = getPackedCoords(tid);
        float sum = 0.0f;
#pragma unroll
        for (int w = 0; w < 8; w++) {
            // tile 1 (top left): pixel A results
            sum += RxLocal[w * 32 + coords.x][coords.y];
            // tile 4 (bottom right): pixel B results (offset: rows+8, cols+8)
            sum += RxLocal[w * 32 + 8 + coords.x][8 + coords.y];
        }
        atomicAdd(Rx + tid, toScaledUint64(sum));
    }
}

__global__ void me_p5(const float* __restrict__ input, uint64_t* __restrict__ Rx, uint64_t* __restrict__ rx, const int width, const int height, const int totalBlocksY, const int taskTotal) {
    constexpr int IN_STRIDE = 72;
    constexpr int OUT_STRIDE = 36; // 36 + 0, no need to pad for bank conflicts!

    const int tid = threadIdx.x;
    const int warpId = tid >> 5;
    const int startRow = warpId * 32;
    const int gridTotal = gridDim.x * gridDim.y;
    const int blockLinear = blockIdx.y * gridDim.x + blockIdx.x;

    __shared__ alignas(16) float RxLocal[256][OUT_STRIDE];
    __shared__ alignas(16) half blockValues[5][260];
    __shared__ alignas(16) float rxStaging[8][24];
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
            load_neighbor_vec<5>(localVec8, blockValues, centerVal, tid);
        } else {
#pragma unroll
            for (int i = 0; i < 3; i++)
                localVec8[i] = {};
            centerVal = blockValues[2][tid + 2];
        }

        // accumulate rx
        accumulateRxVec<3>(localVec8, rxPersistent.vals, __half2float(centerVal));

        // accumulate Rx (Tensor Cores)
        half* rowPtr = reinterpret_cast<half*>(&RxLocal[tid][0]);
        half8* rowPtrVec = reinterpret_cast<half8*>(rowPtr);
#pragma unroll
        for (int i = 0; i < 3; i++)
            rowPtrVec[i] = localVec8[i];
        rowPtrVec[3] = {}; // zero padding
        __syncthreads();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> A[2];
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> B[2];
#pragma unroll
        for (int k0 = 0; k0 < 32; k0 += 16) {
            const half* tilePtr = reinterpret_cast<half*>(&RxLocal[startRow + k0][0]);
            // pipeline all memory loads
            wmma::load_matrix_sync(A[0], tilePtr, IN_STRIDE);
            wmma::load_matrix_sync(B[0], tilePtr, IN_STRIDE);
            wmma::load_matrix_sync(A[1], tilePtr + 16, IN_STRIDE);
            wmma::load_matrix_sync(B[1], tilePtr + 16, IN_STRIDE);
            // fire all Tensor Core math
            wmma::mma_sync(acc_C00, A[0], B[0], acc_C00);
            wmma::mma_sync(acc_C10, A[1], B[0], acc_C10);
            wmma::mma_sync(acc_C11, A[1], B[1], acc_C11);
        }
        __syncthreads();
    }

    // write to global memory
    float* warpOutput = &RxLocal[warpId * 32][0];
    wmma::store_matrix_sync(warpOutput, acc_C00, OUT_STRIDE, wmma::mem_row_major);
    wmma::store_matrix_sync(warpOutput + 16 * OUT_STRIDE, acc_C10, OUT_STRIDE, wmma::mem_row_major);
    wmma::store_matrix_sync(warpOutput + 16 * OUT_STRIDE + 16, acc_C11, OUT_STRIDE, wmma::mem_row_major);
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
        atomicAdd(Rx + k, toScaledUint64(sum));
    };
    // pass 1: [0, 255]
    writeRx(tid);
    // pass 2: [256, 299]
    if (tid < 44)
        writeRx(tid + 256);
}

__global__ void me_p7(const float* __restrict__ input, uint64_t* __restrict__ Rx, uint64_t* __restrict__ rx, const int width, const int height, const int totalBlocksY, const int taskTotal) {
    constexpr int IN_STRIDE = 120;
    constexpr int OUT_STRIDE = 60; // 56 + 4 to avoid bank conflicts

    const int tid = threadIdx.x;
    const int warpId = tid >> 5;
    const int startRow = warpId * 32;
    const int gridTotal = gridDim.x * gridDim.y;
    const int blockLinear = blockIdx.y * gridDim.x + blockIdx.x;

    __shared__ alignas(16) float RxLocal[192][OUT_STRIDE];
    __shared__ alignas(16) float rxStaging[4][48];
    __shared__ typename cub::WarpReduce<rxVecData<48>>::TempStorage temp_storage[4];
    half(*blockValues)[134] = reinterpret_cast<half(*)[134]>(&RxLocal[0][0]); // trick to reuse shared memory

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
            load_neighbor_vec<7>(localVec8, blockValues, centerVal, tid);
        } else {
#pragma unroll
            for (int i = 0; i < 6; i++)
                localVec8[i] = {};
            centerVal = blockValues[3][tid + 3];
        }

        // accumulate rx, note: for p=7 because we reuse shared memory we are forced to sync here!
        accumulateRxVec<6>(localVec8, rxPersistent.vals, __half2float(centerVal));
        __syncthreads();

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
            // pipeline all memory loads
#pragma unroll
            for (int i = 0; i < 3; i++) {
                wmma::load_matrix_sync(A[i], tilePtr + (i * 16), IN_STRIDE);
                wmma::load_matrix_sync(B[i], tilePtr + (i * 16), IN_STRIDE);
            }
            // fire all Tensor Core math
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
    wmma::store_matrix_sync(warpOutput, acc_Rx[0], OUT_STRIDE, wmma::mem_row_major);
    wmma::store_matrix_sync(warpOutput + 16 * OUT_STRIDE, acc_Rx[1], OUT_STRIDE, wmma::mem_row_major);
    wmma::store_matrix_sync(warpOutput + 16 * OUT_STRIDE + 16, acc_Rx[2], OUT_STRIDE, wmma::mem_row_major);
    wmma::store_matrix_sync(warpOutput + 32 * OUT_STRIDE, acc_Rx[3], OUT_STRIDE, wmma::mem_row_major);
    wmma::store_matrix_sync(warpOutput + 32 * OUT_STRIDE + 16, acc_Rx[4], OUT_STRIDE, wmma::mem_row_major);
    wmma::store_matrix_sync(warpOutput + 32 * OUT_STRIDE + 32, acc_Rx[5], OUT_STRIDE, wmma::mem_row_major);
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
        atomicAdd(Rx + k, toScaledUint64(sum));
    }
}

__global__ void me_p9(const float* __restrict__ input, uint64_t* __restrict__ Rx, uint64_t* __restrict__ rx, const int width, const int height, const int totalBlocksY, const int taskTotal) {
    constexpr int INPUT_STRIDE = 184;
    constexpr int OUT_STRIDE = 92; // 88 + 4 to avoid bank conflicts

    const int tid = threadIdx.x;
    const int warpId = tid >> 5;
    const int startRow = warpId * 32;
    const int gridTotal = gridDim.x * gridDim.y;
    const int blockLinear = blockIdx.y * gridDim.x + blockIdx.x;

    __shared__ alignas(16) float RxLocal[128][OUT_STRIDE];
    __shared__ typename cub::WarpReduce<rxVecData<80>>::TempStorage temp_storage[4];
    half(*blockValues)[136] = reinterpret_cast<half(*)[136]>(&RxLocal[0][0]); // trick to reuse shared memory

    // init Accumulators (float)
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
            load_neighbor_vec<9>(localVec8, blockValues, centerVal, tid);
        } else {
#pragma unroll
            for (int i = 0; i < 10; i++)
                localVec8[i] = {};
            centerVal = blockValues[4][tid + 4];
        }
        const float center = __half2float(centerVal);

        // accumulate rx, note: for p=9 because we reuse shared memory we are forced to sync here!
        accumulateRxVec<10>(localVec8, rxPersistent.vals, __half2float(centerVal));
        __syncthreads();

        // accumulate Rx (Tensor Cores)
        half8* shmemPtr = reinterpret_cast<half8*>(&RxLocal[tid][0]);
#pragma unroll
        for (int i = 0; i < 10; i++)
            shmemPtr[i] = localVec8[i];
        shmemPtr[10] = {}; // zero padding
        __syncthreads();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> A[5];
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> B[2];

        const half* tilePtr = reinterpret_cast<half*>(&RxLocal[startRow][0]);
#pragma unroll
        for (int k0 = 0; k0 < 32; k0 += 16) {
            const half* currTile = tilePtr + (k0 * INPUT_STRIDE);
            // pipeline all 5 A loads
#pragma unroll
            for (int i = 0; i < 5; i++)
                wmma::load_matrix_sync(A[i], currTile + (i * 16), INPUT_STRIDE);
            // load the first two B (col 0)
            wmma::load_matrix_sync(B[0], currTile + (0 * 16), INPUT_STRIDE);
            wmma::load_matrix_sync(B[1], currTile + (1 * 16), INPUT_STRIDE);
            // fire Tensor math for col 0
            wmma::mma_sync(acc_Rx[0], A[0], B[0], acc_Rx[0]);
            wmma::mma_sync(acc_Rx[1], A[1], B[0], acc_Rx[1]);
            wmma::mma_sync(acc_Rx[3], A[2], B[0], acc_Rx[3]);
            wmma::mma_sync(acc_Rx[6], A[3], B[0], acc_Rx[6]);
            wmma::mma_sync(acc_Rx[10], A[4], B[0], acc_Rx[10]);
            // B[0] is done, we can overwrite it with col 2 in the background
            wmma::load_matrix_sync(B[0], currTile + (2 * 16), INPUT_STRIDE);
            // fire Tensor math for col 1
            wmma::mma_sync(acc_Rx[2], A[1], B[1], acc_Rx[2]);
            wmma::mma_sync(acc_Rx[4], A[2], B[1], acc_Rx[4]);
            wmma::mma_sync(acc_Rx[7], A[3], B[1], acc_Rx[7]);
            wmma::mma_sync(acc_Rx[11], A[4], B[1], acc_Rx[11]);
            // B[1] is done, we can overwrite it with col 3 in the background
            wmma::load_matrix_sync(B[1], currTile + (3 * 16), INPUT_STRIDE);
            // fire Tensor math for col 2
            wmma::mma_sync(acc_Rx[5], A[2], B[0], acc_Rx[5]);
            wmma::mma_sync(acc_Rx[8], A[3], B[0], acc_Rx[8]);
            wmma::mma_sync(acc_Rx[12], A[4], B[0], acc_Rx[12]);
            // B[0] is done, we can overwrite it with col 4 in the background
            wmma::load_matrix_sync(B[0], currTile + (4 * 16), INPUT_STRIDE);
            // fire Tensor math for col 3
            wmma::mma_sync(acc_Rx[9], A[3], B[1], acc_Rx[9]);
            wmma::mma_sync(acc_Rx[13], A[4], B[1], acc_Rx[13]);
            // fire Tensor math for col 4
            wmma::mma_sync(acc_Rx[14], A[4], B[0], acc_Rx[14]);
        }
        __syncthreads();
    }

    // write to global memory
    float* warpOutput = &RxLocal[warpId * 32][0];

    // write Rx chunked
    // chunk 1
    wmma::store_matrix_sync(warpOutput + (0), acc_Rx[0], OUT_STRIDE, wmma::mem_row_major);
    wmma::store_matrix_sync(warpOutput + (16 * OUT_STRIDE), acc_Rx[1], OUT_STRIDE, wmma::mem_row_major);
    wmma::store_matrix_sync(warpOutput + (16 * OUT_STRIDE) + 16, acc_Rx[2], OUT_STRIDE, wmma::mem_row_major);
    __syncthreads();
    RxStreamPass<0, 528, 0>(tid, RxLocal, Rx);
    __syncthreads();

    // chunk 2
    wmma::store_matrix_sync(warpOutput, acc_Rx[3], OUT_STRIDE, wmma::mem_row_major);
    wmma::store_matrix_sync(warpOutput + 16, acc_Rx[4], OUT_STRIDE, wmma::mem_row_major);
    wmma::store_matrix_sync(warpOutput + 32, acc_Rx[5], OUT_STRIDE, wmma::mem_row_major);
    __syncthreads();
    RxStreamPass<528, 1176, 32>(tid, RxLocal, Rx);
    __syncthreads();

    // chunk 3
    wmma::store_matrix_sync(warpOutput, acc_Rx[6], OUT_STRIDE, wmma::mem_row_major);
    wmma::store_matrix_sync(warpOutput + 16, acc_Rx[7], OUT_STRIDE, wmma::mem_row_major);
    wmma::store_matrix_sync(warpOutput + 32, acc_Rx[8], OUT_STRIDE, wmma::mem_row_major);
    wmma::store_matrix_sync(warpOutput + 48, acc_Rx[9], OUT_STRIDE, wmma::mem_row_major);
    __syncthreads();
    RxStreamPass<1176, 2080, 48>(tid, RxLocal, Rx);
    __syncthreads();

    // chunk 4
    wmma::store_matrix_sync(warpOutput, acc_Rx[10], OUT_STRIDE, wmma::mem_row_major);
    wmma::store_matrix_sync(warpOutput + 16, acc_Rx[11], OUT_STRIDE, wmma::mem_row_major);
    wmma::store_matrix_sync(warpOutput + 32, acc_Rx[12], OUT_STRIDE, wmma::mem_row_major);
    wmma::store_matrix_sync(warpOutput + 48, acc_Rx[13], OUT_STRIDE, wmma::mem_row_major);
    wmma::store_matrix_sync(warpOutput + 64, acc_Rx[14], OUT_STRIDE, wmma::mem_row_major);
    __syncthreads();
    RxStreamPass<2080, 3240, 64>(tid, RxLocal, Rx);

    // write rx (cub) after the barrier, we can't interleave (we are dangerously close to max limit, plus register pressure is already high)
    __syncthreads();
    writeRxVec<80>(rx, rxPersistent, temp_storage[warpId], warpOutput);
}

__global__ void me_u_and_sumsq_fused(const float* __restrict__ errorSeq, const float* __restrict__ w, float* __restrict__ u, uint64_t* __restrict__ globalSumSq, const float* __restrict__ maxVal,
                                     const int N) {
    constexpr int blockSize = 768;

    using BlockReduceT = cub::BlockReduce<float, blockSize>;
    __shared__ typename BlockReduceT::TempStorage temp_storage;

    const int tid = threadIdx.x;
    const int gridSize = blockDim.x * gridDim.x;

    const float invDenom = 1.0f / ((*maxVal) + 1.0e-6f); // cached
    float threadSumSq = 0.0f;
    int idx = blockIdx.x * blockDim.x + tid;

    // grid stride loop
    while (idx < N) {
        // normalize error sequence (mask), calculate u, store globally AND calculate local sum of squares of u in one fused pass
        const float maskVal = errorSeq[idx] * invDenom;
        const float uVal = maskVal * w[idx];
        u[idx] = uVal;
        threadSumSq += uVal * uVal;
        idx += gridSize;
    }

    // block reduce with cub and atomic add to global sum by the leader
    const float blockTotalSq = BlockReduceT(temp_storage).Sum(threadSumSq);
    if (tid == 0)
        atomicAdd(globalSumSq, toScaledUint64(blockTotalSq));
}

__global__ void apply_watermark_fused(const float* __restrict__ input, const float* __restrict__ u, const uint64_t* __restrict__ sumSqPtr, uint8_t* __restrict__ output, const float strengthNumerator,
                                      const int planeElements, const int numChannels) {
    const float uSumSquared = toUnscaledFloat(*sumSqPtr); // read the precomputed sum of squares from global memory (all threads read the same value, it is cached)
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

__global__ void compute_abs_normalized_mask(const float* __restrict__ errorSeq, float* __restrict__ mask, const float* __restrict__ maxVal, const int N) {
    const int stride = blockDim.x * gridDim.x;
    const float invDenom = 1.0f / ((*maxVal) + 1.0e-6f); // cached
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < N) {
        // read, absolute, divide, and write value in one fused pass
        mask[idx] = fabsf(errorSeq[idx]) * invDenom;
        idx += stride;
    }
}

__global__ void calculate_final_correlation(const float* __restrict__ partialDots, const float* __restrict__ partialNormU, const float* __restrict__ partialNormZ, float* __restrict__ result,
                                            const int numBlocks) {
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

    __shared__ float tile[32][33]; // 32x32 tile, +1 to avoid bank conflicts

    // x and y are the coordinates in the original input image
    const int x = blockIdx.x * 32 + threadIdx.x;
    const int y = blockIdx.y * 32 + threadIdx.y;
    // loop 4 times to process 32 rows
#pragma unroll
    for (int i = 0; i < 32; i += 8) {
        float val = 0.0f;
        if (x < width && (y + i) < height)
            val = static_cast<float>(input[(y + i) * pitch + x]);
        tile[threadIdx.y + i][threadIdx.x] = val;
    }
    __syncthreads();

    // swap the grid coords to calculate the transposed destination
    const int dstX = blockIdx.y * 32 + threadIdx.x;
    const int dstY = blockIdx.x * 32 + threadIdx.y;
#pragma unroll
    for (int i = 0; i < 32; i += 8) {
        // read from transposed shared memory coordinates (+1 in the stride to avoid bank conflicts) and write to global memory transposed
        if (dstX < height && (dstY + i) < width)
            output[(dstY + i) * height + dstX] = tile[threadIdx.x][threadIdx.y + i];
    }
}