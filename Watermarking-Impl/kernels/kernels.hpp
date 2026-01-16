#pragma once
#include <string>
inline const std::string kernels = R"CLC(

#define PAD               (WINDOW_SIZE / 2)
#define N_PIXELS          (float) (WINDOW_SIZE * WINDOW_SIZE)
#define N_PIXELS_SQ       (N_PIXELS * N_PIXELS)
#define SHAREDSIZE        (16 + 2 * PAD)

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

inline void fillBlock(__global const float* restrict input, __local float* restrict sharedMem, const int width, const int height) {
    const int groupStartCol = (int)(get_group_id(1) * get_local_size(1));
    const int groupStartRow = (int)(get_group_id(0) * get_local_size(0));
    const int maxCol = width - 1;
    const int maxRow = height - 1;
    const int stride = (int)(get_local_size(0) * get_local_size(1));
    for (int i = (int)(get_local_id(1) * get_local_size(0) + get_local_id(0)); i < SHAREDSIZE * SHAREDSIZE; i += stride) {
        const int tileRow = i % SHAREDSIZE;
        const int tileCol = i / SHAREDSIZE;
        const int globalX = clamp(groupStartCol + tileCol - PAD, 0, maxCol);
        const int globalY = clamp(groupStartRow + tileRow - PAD, 0, maxRow);
        sharedMem[tileRow * (SHAREDSIZE + 1) + tileCol] = input[globalX * height + globalY];
    }
}

__kernel void nvf(__global const float* restrict input, __global float* restrict nvf, const unsigned int width, const unsigned int height) {	
	const int x = get_global_id(1);
    const int y = get_global_id(0);
    __local float region[SHAREDSIZE][SHAREDSIZE + 1];

	fillBlock(input, &region[0][0], width, height);
    barrier(CLK_LOCAL_MEM_FENCE);

    if (y >= height || x >= width)
        return;

    const int shX = get_local_id(1) + PAD;
    const int shY = get_local_id(0) + PAD;

	float sum = 0.0f, sumSq = 0.0f;
	for (int i = -PAD; i <= PAD; i++) {
		for (int j = -PAD; j <= PAD; j++) {
			const float pixelValue = region[shY + i][shX + j];
			sum += pixelValue;
			sumSq += pixelValue * pixelValue;
		}
	}
	const float numerator = (N_PIXELS * sumSq) - (sum * sum);
    const float output = native_divide(numerator, N_PIXELS_SQ + numerator);
	nvf[(x * height) + y] = fmax(output, 0.0f);
}

//use pointer arithmetic for dot product to help compilers optimize address calculations fast
inline float error_sequence_coeffs_filter(__local float* centerPtr, __constant float* coeffs) {
    #define P(r, c) centerPtr[(r) * (SHAREDSIZE + 1) + (c)] 
    float dot = 0.0f;
    int k = 0;
#pragma unroll
    for (int i = -PAD; i <= PAD; i++) {
#pragma unroll
        for (int j = -PAD; j <= PAD; j++) {
            if (i == 0 && j == 0)
                continue;
            dot += coeffs[k] * P(i, j);
            k++;
        }
    }
    return P(0, 0) - dot;
    #undef P
}

__kernel void error_sequence(
    __global const float* restrict input, 
    __global float* restrict x_,
    __constant float* restrict coeffs,
    const unsigned int width,
    const unsigned int height,
    const int calculateAbs,
    __global int* restrict stopFlag) {
    __local float region[SHAREDSIZE][SHAREDSIZE + 1];
    __local float* centerPtr = &region[get_local_id(0) + PAD][get_local_id(1) + PAD];
    fillBlock(input, &region[0][0], width, height);
    barrier(CLK_LOCAL_MEM_FENCE);

    const int x = get_global_id(1);
    const int y = get_global_id(0);
    if (x < width && y < height) {
        if (*stopFlag) {
            x_[x * height + y] = 0.0f;
            return;
        }
        const float output = error_sequence_coeffs_filter(centerPtr, coeffs);
        x_[x * height + y] = calculateAbs ? fabs(output) : output;
    }
}

__kernel void error_sequence_fused(
    __global const float* restrict inputA, 
    __global const float* restrict inputB,
    __global float* restrict x_,
    __constant float* restrict coeffs,
    const int width,
    const int height,
    __constant int* restrict stopFlag) {
    __local float region[SHAREDSIZE][SHAREDSIZE + 1];
    __local float* centerPtr = &region[get_local_id(0) + PAD][get_local_id(1) + PAD];
   
    const int groupStartRow = (int)(get_group_id(0) * get_local_size(0));
    const int groupStartCol = (int)(get_group_id(1) * get_local_size(1));
    const int maxRow = height - 1;
    const int maxCol = width - 1;
    const int stride = (int)(get_local_size(0) * get_local_size(1));
    for (int i = (int)(get_local_id(1) * get_local_size(0) + get_local_id(0)); i < SHAREDSIZE * SHAREDSIZE; i += stride) {
        const int tileRow = i % SHAREDSIZE;
        const int tileCol = i / SHAREDSIZE;
        const int globalX = clamp(groupStartCol + tileCol - PAD, 0, maxCol);
        const int globalY = clamp(groupStartRow + tileRow - PAD, 0, maxRow);
        const int idx = globalX * height + globalY;
        region[tileRow][tileCol] = inputA[idx] * inputB[idx];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int x = get_global_id(1);
    const int y = get_global_id(0);
    if (x < width && y < height) {
        if (*stopFlag) { 
            x_[x * height + y] = 0.0f; 
            return; 
        }
        x_[x * height + y] = error_sequence_coeffs_filter(centerPtr, coeffs);
    }
}

inline void me_p3_rxCalculate(__local half RxLocal[256][40], const int localId, const half x_0, const half x_1, const half x_2, const half x_3, const half x_4, const half x_5, const half x_6, const half x_7, const half x_8)
{
    __local half8* rowPtr = (__local half8*) &RxLocal[localId][0];
    *rowPtr = (half8)(x_0 * x_4, x_1 * x_4, x_2 * x_4, x_3 * x_4, x_5 * x_4, x_6 * x_4, x_7 * x_4, x_8 * x_4);
}

inline void me_p3_RxCalculate(__local half RxLocal[256][40], const int localId, const half x_0, const half x_1, const half x_2, const half x_3, const half x_5, const half x_6, const half x_7, const half x_8)
{
    __local half8* rowPtr = (__local half8*) &RxLocal[localId][0];
    rowPtr[0] = (half8)(x_0 * x_0, x_0 * x_1, x_0 * x_2, x_0 * x_3, x_0 * x_5, x_0 * x_6, x_0 * x_7, x_0 * x_8);
    rowPtr[1] = (half8)(x_1 * x_1, x_1 * x_2, x_1 * x_3, x_1 * x_5, x_1 * x_6, x_1 * x_7, x_1 * x_8, x_2 * x_2);
    rowPtr[2] = (half8)(x_2 * x_3, x_2 * x_5, x_2 * x_6, x_2 * x_7, x_2 * x_8, x_3 * x_3, x_3 * x_5, x_3 * x_6);
    rowPtr[3] = (half8)(x_3 * x_7, x_3 * x_8, x_5 * x_5, x_5 * x_6, x_5 * x_7, x_5 * x_8, x_6 * x_6, x_6 * x_7);
    rowPtr[4] = (half8)(x_6 * x_8, x_7 * x_7, x_7 * x_8, x_8 * x_8, 0.0h, 0.0h, 0.0h, 0.0h);
}

__kernel void me_p3(__global const float* restrict input,
    __global float* restrict Rx,
    __global float* restrict rx,
    const unsigned int width,
    const unsigned int height)

{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int outputIndex = (y *  get_global_size(0)) + x;
    const int localId = get_local_id(0);
    const float halfScaleFactor = 0.00392156862f;

    __local __attribute__((aligned(16))) half RxLocal[256][40];
    __local half blockValues[3][258];
    __local float rxPartial[32][8];

    if (y >= height)
        return;

    for (int i = localId; i < 3 * 258; i += get_local_size(0)) {
        const int tileCol = i / 3;
        const int tileRow = i % 3;
        const int globalX = clamp((int)(get_group_id(0) * get_local_size(0)) + tileCol - 1, 0, (int) width - 1);
        const int globalY = clamp((int)(get_group_id(1) * get_local_size(1)) + tileRow - 1, 0, (int) height - 1);
        vstore_half(input[globalX * height + globalY] * halfScaleFactor, 0, &blockValues[tileRow][tileCol]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    half x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8;
    const bool isValid = (x < width);
    if (isValid) {
        const int localX = localId + 1;
        x_0 = blockValues[0][localX - 1];
        x_1 = blockValues[0][localX];
        x_2 = blockValues[0][localX + 1];
        x_3 = blockValues[1][localX - 1];
        x_4 = blockValues[1][localX];
        x_5 = blockValues[1][localX + 1];
        x_6 = blockValues[2][localX - 1];
        x_7 = blockValues[2][localX];
        x_8 = blockValues[2][localX + 1];
        me_p3_rxCalculate(RxLocal, localId, x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
    }
    else
        vstore_half8((float8)(0.0f), 0, (__local half*)&RxLocal[localId][0]);
    barrier(CLK_LOCAL_MEM_FENCE);

    //OpenCL optimized rx summation
    const int col = localId % 8;
    const int rowStart = (localId / 8) * 8; 
    half psum = 0.0h;
#pragma unroll
    for (int r = 0; r < 8; r++)
        psum += RxLocal[rowStart + r][col];
    rxPartial[localId / 8][col] = (float)psum;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (localId < 8) {
        float sum = 0.0f;
#pragma unroll
        for (int i = 0; i < 32; i++)
            sum += rxPartial[i][localId];
        const int blockOffset = (y * get_num_groups(0) * 8) + (get_group_id(0) * 8);
        rx[blockOffset + localId] = sum;
    } //no barrier needed here

    if (isValid)
        me_p3_RxCalculate(RxLocal, localId, x_0, x_1, x_2, x_3, x_5, x_6, x_7, x_8);
    else {
#pragma unroll
        for(int v=0; v<5; v++)
            vstore_half8((float8)(0.0f), 0, (__local half*)&RxLocal[localId][v*8]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //OpenCL optimized Rx summation
    //parallel partial summation with 252 threads active
    const int flatId = localId;
    if (flatId < 252) {
        const int col = flatId % 36;
        const int chunk = flatId / 36;
        const int rowsPerChunk = 37;
        const int startRow = chunk * rowsPerChunk;
        const int endRow = min(startRow + rowsPerChunk, 256);
        
        float pSum = 0.0f;
        for (int i = startRow; i < endRow; i++)
            pSum += (float)RxLocal[i][col];
        //smartly reuse rxPartial local memory here (for partial sum)
        ((__local float*)rxPartial)[flatId] = pSum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //final summation by the first 36 threads
    if (flatId < 36) {
        float totalSum = 0.0f;
#pragma unroll
        for (int k = 0; k < 7; k++)
            totalSum += ((__local float*)rxPartial)[flatId + k * 36];
        const int blockOffset = (y * get_num_groups(0) * 36) + (get_group_id(0) * 36);
        Rx[blockOffset + flatId] = totalSum;
    }
}

__kernel void calculate_partial_correlation(
    __global const float* restrict e_u,
    __global const float* restrict e_z,
    __global float* restrict partialDots,
    __global float* restrict partialNormU,
    __global float* restrict partialNormZ,
    const unsigned int size) {

    const int tid = get_local_id(0);
    const int gid = get_global_id(0);
    const int groupId = get_group_id(0);

    __local float dotCache[256];
    __local float normUCache[256];
    __local float normZCache[256];

    float a = 0.0f, b = 0.0f;
    if (gid < size) {
        a = e_u[gid];
        b = e_z[gid];
    }

    dotCache[tid] = a * b;
    normUCache[tid] = a * a;
    normZCache[tid] = b * b;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            dotCache[tid] += dotCache[tid + s];
            normUCache[tid] += normUCache[tid + s];
            normZCache[tid] += normZCache[tid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (tid == 0) {
        partialDots[groupId] = dotCache[0];
        partialNormU[groupId] = normUCache[0];
        partialNormZ[groupId] = normZCache[0];
    }
}

__kernel void calculate_final_correlation(
    __global const float* restrict partialDots,
    __global const float* restrict partialNormU,
    __global const float* restrict partialNormZ,
    __global float* restrict result,
    const unsigned int numBlocks) {

    const int tid = get_local_id(0);
    const int localSize = get_local_size(0);

    float localDot = 0.0f;
    float localU = 0.0f;
    float localZ = 0.0f;

    __local float sumDot[1024];
    __local float sumU[1024];
    __local float sumZ[1024];

    for (int i = tid; i < numBlocks; i += localSize) {
        localDot += partialDots[i];
        localU += partialNormU[i];
        localZ += partialNormZ[i];
    }

    sumDot[tid] = localDot;
    sumU[tid] = localU;
    sumZ[tid] = localZ;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = localSize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sumDot[tid] += sumDot[tid + s];
            sumU[tid] += sumU[tid + s];
            sumZ[tid] += sumZ[tid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (tid == 0) {
        float final_dot = sumDot[0];
        float final_norm_u = sqrt(sumU[0]);
        float final_norm_z = sqrt(sumZ[0]);
        result[0] = (final_norm_u > 0.0f && final_norm_z > 0.0f) ? (final_dot / (final_norm_u * final_norm_z)) : 0.0f;
    }
}

__kernel void cholesky_solver_p3(__global const float* restrict A, 
                                 __global const float* restrict B,
                                 __global float* restrict X,
                                 __global int* restrict stopFlag) {
    // OpenCL upper packed Indexing:
    // maps Lower diagonal coords (r, c) where r >= c to the corresponding upper diagonal packed index (c, r).
    #define IDX(r, c) ((c * N) - (c * (c - 1)) / 2 + (r - c))

    const int N = 8; // p = 3 -> 8x8 system
    
    if (get_local_id(0) > 0 || get_group_id(0) > 0)
        return;

    float packed[36]; 
    float localB[8];

    #pragma unroll
    for (int k = 0; k < 36; k++)
        packed[k] = A[k];

    #pragma unroll
    for (int i = 0; i < N; i++)
        localB[i] = B[i];

    #pragma unroll
    for (int i = 0; i < N; i++) {
        #pragma unroll
        for (int j = 0; j <= i; j++) {
            float sum = 0.0f;
            #pragma unroll
            for (int k = 0; k < j; k++)
                sum += packed[IDX(i, k)] * packed[IDX(j, k)];

            if (i == j) {
                const float val = packed[IDX(i, i)] - sum;
                if (val <= 1e-12f) {
                    *stopFlag = 1;
                    goto exit; 
                }
                packed[IDX(i, i)] = sqrt(val);
            } else {
                packed[IDX(i, j)] = (packed[IDX(i, j)] - sum) / packed[IDX(j, j)];
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < N; i++) {
        float sum = 0.0f;
        #pragma unroll
        for (int k = 0; k < i; k++)
            sum += packed[IDX(i, k)] * localB[k];
        localB[i] = (localB[i] - sum) / packed[IDX(i, i)];
    }

    #pragma unroll
    for (int i = N - 1; i >= 0; i--) {
        float sum = 0.0f;
        #pragma unroll
        for (int k = i + 1; k < N; k++)
            sum += packed[IDX(k, i)] * localB[k];
        localB[i] = (localB[i] - sum) / packed[IDX(i, i)];
    }
    *stopFlag = 0;
exit:
    #pragma unroll
    for (int i = 0; i < N; i++)
        X[i] = localB[i];

    #undef IDX
}

)CLC";