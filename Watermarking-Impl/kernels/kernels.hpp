#pragma once
#include <string>
inline const std::string kernels = R"CLC(

#define PAD           (WINDOW_SIZE / 2)
#define SHAREDSIZE    (16 + 2 * PAD)

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

inline void fillBlock(
    __global const float* __restrict__ input,
    __local float* __restrict__ sharedMem,
    const int width,
    const int height)
{
    for (int i = get_local_id(1) * get_local_size(0) + get_local_id(0); i < SHAREDSIZE * SHAREDSIZE; i += get_local_size(0) * get_local_size(1))
    {
        const int tileRow = i / SHAREDSIZE;
        const int tileCol = i % SHAREDSIZE;
        const int globalX = clamp((int)(get_group_id(1) * get_local_size(1) + tileCol - PAD), 0, width - 1);
        const int globalY = clamp((int)(get_group_id(0) * get_local_size(0) + tileRow - PAD), 0, height - 1);
        sharedMem[tileRow * SHAREDSIZE + tileCol] = input[globalX * height + globalY];
    }
}

__kernel void nvf(__global const float* __restrict__ input, 
	__global float* __restrict__ nvf,
	const unsigned int width,
    const unsigned int height)
{	
	const int pad = WINDOW_SIZE / 2;
	const int pSquared = WINDOW_SIZE * WINDOW_SIZE;
	const int x = get_global_id(1);
    const int y = get_global_id(0);
    __local float region[16 + 2 * (WINDOW_SIZE/2)][16 + 2 * (WINDOW_SIZE/2)];

	fillBlock(input, &region[0][0], width, height);
    barrier(CLK_LOCAL_MEM_FENCE);

    if (y >= height || x >= width)
        return;

    const int shX = get_local_id(1) + pad;
    const int shY = get_local_id(0) + pad;

	float sum = 0.0f, sumSq = 0.0f;
	for (int i = -pad; i <= pad; i++)
	{
		for (int j = -pad; j <= pad; j++)
		{
			float pixelValue = region[shY + i][shX + j];
			sum += pixelValue;
			sumSq += pixelValue * pixelValue;
		}
	}
	float mean = sum / pSquared;
	float variance = (sumSq / pSquared) - (mean * mean);
	nvf[(x * height) + y] = fmax(variance / (1 + variance), 0.0f);
}

__kernel void error_sequence_p3(
    __global const float* __restrict__ input, 
    __global float* __restrict__ x_,
    __constant float* __restrict__ coeffs,
    const unsigned int width,
    const unsigned int height)
{
    const int x = get_global_id(1);
    const int y = get_global_id(0);
    __local float region[16 + 2][16 + 2];

    fillBlock(input, &region[0][0], width, height);
    barrier(CLK_LOCAL_MEM_FENCE);

    //calculate the dot product of the coefficients and the neighborhood for this pixel
    if (x < width && y < height) 
    {
        const int centerCol = get_local_id(1) + 1;
        const int centerRow = get_local_id(0) + 1;
        float dot = 0.0f;
        dot += coeffs[0] * region[centerRow - 1][centerCol - 1];
        dot += coeffs[1] * region[centerRow - 1][centerCol];
        dot += coeffs[2] * region[centerRow - 1][centerCol + 1];
        dot += coeffs[3] * region[centerRow][centerCol - 1];
        dot += coeffs[4] * region[centerRow][centerCol + 1];
        dot += coeffs[5] * region[centerRow + 1][centerCol - 1];
        dot += coeffs[6] * region[centerRow + 1][centerCol];
        dot += coeffs[7] * region[centerRow + 1][centerCol + 1];
		x_[(x * height + y)] = region[centerRow][centerCol] - dot;
    }
}

inline void me_p3_rxCalculate(__local half RxLocal[64][36], const int localId, const half x_0, const half x_1, const half x_2, const half x_3, const half x_4, const half x_5, const half x_6, const half x_7, const half x_8)
{
    vstore_half8((float8)(x_0 * x_4, x_1 * x_4, x_2 * x_4, x_3 * x_4, x_5 * x_4, x_6 * x_4, x_7 * x_4, x_8 * x_4), 0, &RxLocal[localId][0]);
}

inline void me_p3_RxCalculate(__local half RxLocal[64][36], const int localId, const half x_0, const half x_1, const half x_2, const half x_3, const half x_5, const half x_6, const half x_7, const half x_8)
{
    vstore_half8((float8)(x_0 * x_0, x_0 * x_1, x_0 * x_2, x_0 * x_3, x_0 * x_5, x_0 * x_6, x_0 * x_7, x_0 * x_8), 0, &RxLocal[localId][0]);
    vstore_half8((float8)(x_1 * x_1, x_1 * x_2, x_1 * x_3, x_1 * x_5, x_1 * x_6, x_1 * x_7, x_1 * x_8, x_2 * x_2), 0, &RxLocal[localId][8]);
    vstore_half8((float8)(x_2 * x_3, x_2 * x_5, x_2 * x_6, x_2 * x_7, x_2 * x_8, x_3 * x_3, x_3 * x_5, x_3 * x_6), 0, &RxLocal[localId][16]);
    vstore_half8((float8)(x_3 * x_7, x_3 * x_8, x_5 * x_5, x_5 * x_6, x_5 * x_7, x_5 * x_8, x_6 * x_6, x_6 * x_7), 0, &RxLocal[localId][24]);
    vstore_half4((float4)(x_6 * x_8, x_7 * x_7, x_7 * x_8, x_8 * x_8), 0, &RxLocal[localId][32]);
}

__kernel void me(__global const float* __restrict__ input,
    __global float* __restrict__ Rx,
    __global float* __restrict__ rx,
    __constant int* __restrict__ RxMappings,
    const unsigned int width,
    const unsigned int paddedWidth,
    const unsigned int height)

{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int outputIndex = (y * paddedWidth) + x;
    const int localId = get_local_id(0);
    const int widthLimit = paddedWidth == width ? 64 :get_group_id(0) == get_num_groups(0) - 1 ? 64 - (paddedWidth - width) : 64;

    __local half RxLocal[64][36];
    __local half blockValues[3][66];

    if (y >= height)
        return;

    for (int i = localId; i < 3 * 66; i += get_local_size(0))
    {
        const int tileCol = i / 3;
        const int tileRow = i % 3;
        const int globalX = clamp((int)(get_group_id(0) * get_local_size(0)) + tileCol - 1, 0, (int) width - 1);
        const int globalY = clamp((int)(get_group_id(1) * get_local_size(1)) + tileRow - 1, 0, (int) height - 1);
        vstore_half(input[globalX * height + globalY], 0, &blockValues[tileRow][tileCol]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    half x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8;
    if (x < width)
    {
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
    barrier(CLK_LOCAL_MEM_FENCE);

    //TODO can be optimized
    if (localId < 8)
    {
        float sum = 0.0f;
#pragma unroll
        for (int i = 0; i < widthLimit; i++)
            sum += RxLocal[i][localId];
        rx[(outputIndex / 8) + localId] = sum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (x < width)
        me_p3_RxCalculate(RxLocal, localId, x_0, x_1, x_2, x_3, x_5, x_6, x_7, x_8);
    barrier(CLK_LOCAL_MEM_FENCE);

    //TODO can be optimized
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < widthLimit; i++)
        sum += RxLocal[i][RxMappings[localId]];
    Rx[outputIndex] = sum;
}

__kernel void calculate_partial_correlation(
    __global const float* restrict e_u,
    __global const float* restrict e_z,
    __global float* restrict partialDots,
    __global float* restrict partialNormU,
    __global float* restrict partialNormZ,
    const unsigned int size)
{
    const int tid = get_local_id(0);
    const int gid = get_global_id(0);
    const int groupId = get_group_id(0);

    __local float dotCache[256];
    __local float normUCache[256];
    __local float normZCache[256];

    float a = 0.0f, b = 0.0f;
    if (gid < size) 
    {
        a = e_u[gid];
        b = e_z[gid];
    }

    dotCache[tid] = a * b;
    normUCache[tid] = a * a;
    normZCache[tid] = b * b;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = 128; s > 0; s >>= 1) 
    {
        if (tid < s) 
        {
            dotCache[tid] += dotCache[tid + s];
            normUCache[tid] += normUCache[tid + s];
            normZCache[tid] += normZCache[tid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (tid == 0) 
    {
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
    const unsigned int numBlocks)
{
    const int tid = get_local_id(0);
    float localDot = 0.0f;
    float localU = 0.0f;
    float localZ = 0.0f;

    __local float sumDot[1024];
    __local float sumU[1024];
    __local float sumZ[1024];

    for (int i = tid; i < numBlocks; i += 1024) 
    {
        localDot += partialDots[i];
        localU += partialNormU[i];
        localZ += partialNormZ[i];
    }

    sumDot[tid] = localDot;
    sumU[tid] = localU;
    sumZ[tid] = localZ;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = 512; s > 0; s >>= 1) 
    {
        if (tid < s) 
        {
            sumDot[tid] += sumDot[tid + s];
            sumU[tid] += sumU[tid + s];
            sumZ[tid] += sumZ[tid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (tid == 0) 
    {
        float final_dot = sumDot[0];
        float final_norm_u = sqrt(sumU[0]);
        float final_norm_z = sqrt(sumZ[0]);
        result[0] = (final_norm_u > 0.0f && final_norm_z > 0.0f) ? (final_dot / (final_norm_u * final_norm_z)) : 0.0f;
    }
}

)CLC";