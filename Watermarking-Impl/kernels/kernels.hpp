#pragma once
#include <string>
inline const std::string kernels = R"CLC(

#define PAD               (WINDOW_SIZE / 2)
#define NEIGHB_SIZE       ((WINDOW_SIZE * WINDOW_SIZE) - 1)
#define N_PIXELS          (float) (WINDOW_SIZE * WINDOW_SIZE)
#define N_PIXELS_SQ       (N_PIXELS * N_PIXELS)
#define SHAREDSIZE        (16 + 2 * PAD)

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

inline int2 getPackedCoords(int k) {
    const int r = (int)((sqrt(1.0f + 8.0f * k) - 1.0f) / 2.0f);
    const int c = k - (r * (r + 1)) / 2;
    return (int2)(r, c);
}

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
	nvf[(x * height) + y] = clamp(output, 0.0f, 255.0f);
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

)CLC"
R"CLC(

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

// OpenCL upper packed Indexing:
// maps Lower diagonal coords (r, c) where r >= c to the corresponding upper diagonal packed index (c, r)
#define SOLVER_IDX(r, c) ((c * NEIGHB_SIZE) - (c * (c - 1)) / 2 + (r - c))

#if WINDOW_SIZE == 3
__kernel void me(__global const float* restrict input,
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

    const int totalPixels = 3 * 258;
    const int colStep = 85;
    const int rowStep = 1;
    const int baseGlobalCol = (get_group_id(0) * 256) - 1; 
    const int baseGlobalRow = get_group_id(1) - 1;
    int r = localId % 3;
    int c = localId / 3;
    int idx = localId;
    while (idx < totalPixels) {
        int gCol = clamp(baseGlobalCol + c, 0, (int)width - 1);
        int gRow = clamp(baseGlobalRow + r, 0, (int)height - 1);
        vstore_half(input[gCol * height + gRow] * halfScaleFactor, 0, &blockValues[r][c]);
        idx += 256;
        c += colStep;
        r += rowStep;
        if (r >= 3) {
            r -= 3;
            c += 1;
        }
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

#elif WINDOW_SIZE == 5
__kernel void me(
    __global const float* restrict input,
    __global float* restrict Rx,
    __global float* restrict rx,
    const unsigned int width,
    const unsigned int height
) {
    const int gx = get_group_id(0);
    const int gy = get_group_id(1);
    const int localId = get_local_id(0);
    const float halfScaleFactor = 0.00392156862f;

    __local half blockValues[5][260]; 

    const int totalPixels = 5 * 260;
    const int colStep = 51;
    const int rowStep = 1;
    const int baseGlobalCol = (gx * 256) - 2; 
    const int baseGlobalRow = gy - 2;
    int r = localId % 5;
    int c = localId / 5;
    int idx = localId;
    while (idx < totalPixels) {
        int gCol = clamp(baseGlobalCol + c, 0, (int)width - 1);
        int gRow = clamp(baseGlobalRow + r, 0, (int)height - 1);
        vstore_half(input[gCol * height + gRow] * halfScaleFactor, 0, &blockValues[r][c]);
        idx += 256;
        c += colStep;
        r += rowStep;
        if (r >= 5) {
            r -= 5;
            c += 1;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // calculate Rx and rx coefficients
    // for p=5 we work differently than p=3 because of massive shared memory requirements
    // we process 324 total tasks (300 for Rx, 24 for rx)
    for (int k = localId; k < 324; k += 256) {
        float coeffSum = 0.0f;
        if (k < 300) {
            // which coefficient (r, c) we are calculating
            const int2 coords = getPackedCoords(k);
            const int r = coords.x; 
            const int c = coords.y; 
            // accumulate over 256 pixels
            for (int p = 0; p < 256; p++) {
                // map neighbor index to cache location (we also skip the center pixel at index 12)
                const int cacheRow = (r >= 12) ? r + 1 : r;
                const int cacheCol = (c >= 12) ? c + 1 : c;
                // get blockValues[neighbor_row][pixel_col + neighbor_col]
                const float val_a = (float)blockValues[cacheRow / 5][p + (cacheRow % 5)];
                const float val_b = (float)blockValues[cacheCol / 5][p + (cacheCol % 5)];
                coeffSum += val_a * val_b;
            }
            // write to global memory to the correct index
            const int outputIndex = (gy * get_num_groups(0) * 300) + (gx * 300) + SOLVER_IDX(r, c);
            Rx[outputIndex] = coeffSum;
            
        } else {
            // rx calculation
            const int nIdx = k - 300;
            const int cacheIdx = (nIdx >= 12) ? nIdx + 1 : nIdx;
            for (int p = 0; p < 256; p++) {
                const float val_n = (float)blockValues[cacheIdx / 5][p + (cacheIdx % 5)];
                // center pixel is always at local coord (2, p+2)
                coeffSum += val_n * (float)blockValues[2][p + 2];
            }
            const int outputIndex = (gy * get_num_groups(0) * 24) + (gx * 24) + nIdx;
            rx[outputIndex] = coeffSum;
        }
    }
}

)CLC"
R"CLC(

#elif WINDOW_SIZE == 7
__kernel void me(
    __global const float* restrict input,
    __global float* restrict Rx,
    __global float* restrict rx,
    const unsigned int width,
    const unsigned int height
) {
    const int gx = get_group_id(0);
    const int gy = get_group_id(1);
    const int localId = get_local_id(0);
    const float halfScaleFactor = 0.00392156862f;

    __local half blockValues[7][262]; 

    const int totalPixels = 7 * 262;
    const int colStep = 36;
    const int rowStep = 4;
    const int baseGlobalCol = (gx * 256) - 3; 
    const int baseGlobalRow = gy - 3;
    int r = localId % 7;
    int c = localId / 7;
    int idx = localId;
    while (idx < totalPixels) {
        int gCol = clamp(baseGlobalCol + c, 0, (int)width - 1);
        int gRow = clamp(baseGlobalRow + r, 0, (int)height - 1);
        vstore_half(input[gCol * height + gRow] * halfScaleFactor, 0, &blockValues[r][c]);
        idx += 256;
        c += colStep;
        r += rowStep;
        if (r >= 7) {
            r -= 7;
            c += 1;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int k = localId; k < 1224; k += 256) {
        float coeffSum = 0.0f;
        if (k < 1176) {
            const int2 coords = getPackedCoords(k);
            const int r = coords.x; 
            const int c = coords.y; 
            for (int p = 0; p < 256; p++) {
                const int cacheRow = (r >= 24) ? r + 1 : r;
                const int cacheCol = (c >= 24) ? c + 1 : c;
                const float val_a = (float)blockValues[cacheRow / 7][p + (cacheRow % 7)];
                const float val_b = (float)blockValues[cacheCol / 7][p + (cacheCol % 7)];
                coeffSum += val_a * val_b;
            }
            const int outputIndex = (gy * get_num_groups(0) * 1176) + (gx * 1176) + SOLVER_IDX(r, c);
            Rx[outputIndex] = coeffSum;
            
        } else {
            const int nIdx = k - 1176;
            const int cacheIdx = (nIdx >= 24) ? nIdx + 1 : nIdx;
            for (int p = 0; p < 256; p++) {
                const float val_n = (float)blockValues[cacheIdx / 7][p + (cacheIdx % 7)];
                coeffSum += val_n * (float)blockValues[3][p + 3];
            }
            const int outputIndex = (gy * get_num_groups(0) * 48) + (gx * 48) + nIdx;
            rx[outputIndex] = coeffSum;
        }
    }
}

#else
__kernel void me(
    __global const float* restrict input,
    __global float* restrict Rx,
    __global float* restrict rx,
    const unsigned int width,
    const unsigned int height
) {
    const int gx = get_group_id(0);
    const int gy = get_group_id(1);
    const int localId = get_local_id(0);
    const float halfScaleFactor = 0.00392156862f;

    __local half blockValues[9][264]; 

    const int totalPixels = 9 * 264;
    const int colStep = 28;
    const int rowStep = 4;
    const int baseGlobalCol = (gx * 256) - 4; 
    const int baseGlobalRow = gy - 4;
    int r = localId % 9;
    int c = localId / 9;
    int idx = localId;
    while (idx < totalPixels) {
        int gCol = clamp(baseGlobalCol + c, 0, (int)width - 1);
        int gRow = clamp(baseGlobalRow + r, 0, (int)height - 1);
        vstore_half(input[gCol * height + gRow] * halfScaleFactor, 0, &blockValues[r][c]);
        idx += 256;
        c += colStep;
        r += rowStep;
        if (r >= 9) {
            r -= 9;
            c += 1;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int k = localId; k < 3320; k += 256) {
        float coeffSum = 0.0f;
        if (k < 3240) {
            const int2 coords = getPackedCoords(k);
            const int r = coords.x; 
            const int c = coords.y; 
            for (int p = 0; p < 256; p++) {
                const int cacheRow = (r >= 40) ? r + 1 : r;
                const int cacheCol = (c >= 40) ? c + 1 : c;
                const float val_a = (float)blockValues[cacheRow / 9][p + (cacheRow % 9)];
                const float val_b = (float)blockValues[cacheCol / 9][p + (cacheCol % 9)];
                coeffSum += val_a * val_b;
            }
            const int outputIndex = (gy * get_num_groups(0) * 3240) + (gx * 3240) + SOLVER_IDX(r, c);
            Rx[outputIndex] = coeffSum;
            
        } else {
            const int nIdx = k - 3240;
            const int cacheIdx = (nIdx >= 40) ? nIdx + 1 : nIdx;
            for (int p = 0; p < 256; p++) {
                const float val_n = (float)blockValues[cacheIdx / 9][p + (cacheIdx % 9)];
                coeffSum += val_n * (float)blockValues[4][p + 4];
            }
            const int outputIndex = (gy * get_num_groups(0) * 80) + (gx * 80) + nIdx;
            rx[outputIndex] = coeffSum;
        }
    }
}
#endif

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

    const size_t rawDots = (size_t)partialDots;
    const size_t rawU = (size_t)partialNormU;
    const size_t rawZ = (size_t)partialNormZ;
    if (((rawDots | rawU | rawZ) & 0xF) == 0) {
        __global const float4* vecDots = (__global const float4*)partialDots;
        __global const float4* vecU = (__global const float4*)partialNormU;
        __global const float4* vecZ = (__global const float4*)partialNormZ;

        const int vecLoopLimit = numBlocks >> 2; 
        for (int i = tid; i < vecLoopLimit; i += localSize) {
            const float4 d = vecDots[i];
            const float4 u = vecU[i];
            const float4 z = vecZ[i];
            localDot += d.x + d.y + d.z + d.w;
            localU += u.x + u.y + u.z + u.w;
            localZ += z.x + z.y + z.z + z.w;
        }
        for (int i = (vecLoopLimit << 2) + tid; i < numBlocks; i += localSize) {
            localDot += partialDots[i];
            localU += partialNormU[i];
            localZ += partialNormZ[i];
        }
    } 
    else {
        for (int i = tid; i < numBlocks; i += localSize) {
            localDot += partialDots[i];
            localU += partialNormU[i];
            localZ += partialNormZ[i];
        }
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

)CLC"
R"CLC(

// naive very low latency 1-thread solver
// define this kernel ONLY for p=3 and p=5
// do NOT define this for p=7 and p=9, opencl driver will hang when building!!
#if WINDOW_SIZE == 3 && WINDOW_SIZE == 5
__kernel void cholesky_solver(__global const float* restrict A, 
                                 __global const float* restrict B,
                                 __global float* restrict X,
                                 __global int* restrict stopFlag) {
    // packed array size: N * (N + 1) / 2
    #define PACKED_SIZE ((NEIGHB_SIZE * (NEIGHB_SIZE + 1)) / 2)
    
    if (get_local_id(0) > 0 || get_group_id(0) > 0)
        return;

    float packed[PACKED_SIZE]; 
    float localB[NEIGHB_SIZE];

    const bool isAligned = ((((size_t)A | (size_t)B | (size_t)X) & 0xF) == 0);
    if (isAligned) {
        __global const float4* vecA = (__global const float4*)A;
        __global const float4* vecB = (__global const float4*)B;
        const int vecLimitA = (PACKED_SIZE + 3) / 4;
        
#pragma unroll
        for (int k = 0; k < vecLimitA; k++) {
            const float4 v = vecA[k];
            packed[k * 4 + 0] = v.x;
            packed[k * 4 + 1] = v.y;
            packed[k * 4 + 2] = v.z;
            packed[k * 4 + 3] = v.w;
        }

        const int vecLimitB = (NEIGHB_SIZE + 3) / 4;
#pragma unroll
        for (int i = 0; i < vecLimitB; i++) {
            const float4 v = vecB[i];
            localB[i * 4 + 0] = v.x;
            localB[i * 4 + 1] = v.y;
            localB[i * 4 + 2] = v.z;
            localB[i * 4 + 3] = v.w;
        }
    } 
    else {
#pragma unroll
        for (int k = 0; k < PACKED_SIZE; k++)
            packed[k] = A[k];
#pragma unroll
        for (int i = 0; i < NEIGHB_SIZE; i++)
            localB[i] = B[i];
    }

    // Cholesky decomposition and solving
    for (int i = 0; i < NEIGHB_SIZE; i++) {
#pragma unroll
        for (int j = 0; j <= i; j++) {
            float sum = 0.0f;
#pragma unroll
            for (int k = 0; k < j; k++)
                sum += packed[SOLVER_IDX(i, k)] * packed[SOLVER_IDX(j, k)];

            if (i == j) {
                const float val = packed[SOLVER_IDX(i, i)] - sum;
                if (val <= 1e-12f) {
                    *stopFlag = 1;
                    goto exit; 
                }
                packed[SOLVER_IDX(i, i)] = sqrt(val);
            } else {
                packed[SOLVER_IDX(i, j)] = (packed[SOLVER_IDX(i, j)] - sum) / packed[SOLVER_IDX(j, j)];
            }
        }
    }

#pragma unroll
    for (int i = 0; i < NEIGHB_SIZE; i++) {
        float sum = 0.0f;
#pragma unroll
        for (int k = 0; k < i; k++)
            sum += packed[SOLVER_IDX(i, k)] * localB[k];
        localB[i] = (localB[i] - sum) / packed[SOLVER_IDX(i, i)];
    }

#pragma unroll
    for (int i = NEIGHB_SIZE - 1; i >= 0; i--) {
        float sum = 0.0f;
#pragma unroll
        for (int k = i + 1; k < NEIGHB_SIZE; k++)
            sum += packed[SOLVER_IDX(k, i)] * localB[k];
        localB[i] = (localB[i] - sum) / packed[SOLVER_IDX(i, i)];
    }
    *stopFlag = 0;
exit:
    if (isAligned) {
        __global float4* vecX = (__global float4*)X;
        const int vecLimitB = NEIGHB_SIZE / 4;
#pragma unroll
        for (int i = 0; i < vecLimitB; i++) {
            float4 v;
            v.x = localB[i * 4 + 0];
            v.y = localB[i * 4 + 1];
            v.z = localB[i * 4 + 2];
            v.w = localB[i * 4 + 3];
            vecX[i] = v;
        }
        for (int i = vecLimitB * 4; i < NEIGHB_SIZE; i++)
            X[i] = localB[i];
    } else {
#pragma unroll
        for (int i = 0; i < NEIGHB_SIZE; i++)
            X[i] = localB[i];
    }
#undef PACKED_SIZE
}

#else
// parallel cholesky solver for p = 7 (N = 48) and p = 9 (N = 80), using one workgroup
__kernel void cholesky_solver(__global const float* restrict A, 
                              __global const float* restrict B,
                              __global float* restrict X,
                              __global int* restrict stopFlag) {

    __local float sA[NEIGHB_SIZE][NEIGHB_SIZE + 1]; // +1 to avoid bank conflicts
    __local float sB[NEIGHB_SIZE];

    const int tid = get_local_id(0);
    const int workers = get_local_size(0);

    // cooperative load of matrix A (Rx)
    // Rx is stored as col-major lower packed (SOLVER_IDX), we unpack it into sA[row][col]
    for (int c = 0; c < NEIGHB_SIZE; c++) {
        for (int r = c + tid; r < NEIGHB_SIZE; r += workers) {
            sA[r][c] = A[SOLVER_IDX(r, c)];
        }
    }

    // cooperative Load of Vector B (rx)
    for (int k = tid; k < NEIGHB_SIZE; k += workers)
        sB[k] = B[k];

    // initialize stopFlag (thread 0 only)
    if (tid == 0) 
        *stopFlag = 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    // cholesky decomposition (in-place on sA)
    for (int k = 0; k < NEIGHB_SIZE; k++) {
        // (thread 0 ) calculate the inverse diagonal to turn division into multiplication
        if (tid == 0) {
            const float diag = sA[k][k];
            if (diag <= 1e-12f) {
                *stopFlag = 1; 
                sA[k][k] = NAN; // signal!
            } else {
                sA[k][k] = 1 / sqrt(diag); // Store 1.0 / L_kk
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // check for abort
        const float invDiag = sA[k][k];
        if (isnan(invDiag))
            return;

        // scale column k (parallel)
        // L_ik = L_ik * (1/L_kk) for i > k
        for (int i = k + 1 + tid; i < NEIGHB_SIZE; i += workers)
            sA[i][k] *= invDiag;
        barrier(CLK_LOCAL_MEM_FENCE);

        // update trailing matrix (in parallel), parallelize i (row)
        for (int j = k + 1; j < NEIGHB_SIZE; j++) {
            const float L_jk = sA[j][k]; // Read L_jk once
            for (int i = j + tid; i < NEIGHB_SIZE; i += workers)
                sA[i][j] -= sA[i][k] * L_jk;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // forward Substitution (L * y = b)
    for (int k = 0; k < NEIGHB_SIZE; k++) {
        if (tid == 0)
            sB[k] *= sA[k][k]; // sA[k][k] is actually 1/L_kk stored from earlier
        barrier(CLK_LOCAL_MEM_FENCE);

        // broadcast y_k
        const float y_k = sB[k];

        // parallel update: b[i] -= L_ik * y_k
        for (int i = k + 1 + tid; i < NEIGHB_SIZE; i += workers)
            sB[i] -= sA[i][k] * y_k;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // backward Substitution (L^T * x = y)
    for (int k = NEIGHB_SIZE - 1; k >= 0; k--) {
        if (tid == 0)
            sB[k] *= sA[k][k]; 
        barrier(CLK_LOCAL_MEM_FENCE);

        // broadcast x_k
        const float x_k = sB[k];

        // parallel update: y[i] -= L_ki * x_k
        // L^T, row i is column i of L, we need L_ki, (we stored Lower), we access sA[k][i]
        for (int i = tid; i < k; i += workers)
            sB[i] -= sA[k][i] * x_k;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // store
    for (int i = tid; i < NEIGHB_SIZE; i += workers) {
        X[i] = sB[i];
    }
}
#endif

#undef SOLVER_IDX

)CLC";