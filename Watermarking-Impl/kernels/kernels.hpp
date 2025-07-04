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
    const unsigned int height,
	__local float region[16 + 2 * (WINDOW_SIZE/2)][16 + 2 * (WINDOW_SIZE/2)])
{	
	const int pad = WINDOW_SIZE / 2;
	const int pSquared = WINDOW_SIZE * WINDOW_SIZE;
	const int x = get_global_id(1);
    const int y = get_global_id(0);

	fillBlock(input, region, width, height);
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
	nvf[(x * height) + y] = variance / (1 + variance);
}

__kernel void scaled_neighbors_p3(
    __global const float* __restrict__ input, 
    __global float* __restrict__ x_,
    __constant float* __restrict__ coeffs,
    const unsigned int width,
    const unsigned int height,
    __local float region[16 + 2][16 + 2]) //hold the 18 x 18 region for this 16 x 16 block
{
    const int x = get_global_id(1);
    const int y = get_global_id(0);

    fillBlock(input, region, width, height);
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
		x_[(x * height + y)] = dot;
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
    const unsigned int height,
    __local half RxLocal[64][36], //64 local threads, 36 values each (8 for rx, this is a shared memory for both Rx,rx)
    __local half blockValues[3][66]) //64 local threads (+2 halos), 3 values each

{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int outputIndex = (y * paddedWidth) + x;
    const int localId = get_local_id(0);

    #pragma unroll
    for (int i = 0; i < 4; i++)
        vstore_half8((float8)(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f), 0, &RxLocal[localId][i * 8]);
    vstore_half4((float4)(0.0f, 0.0f, 0.0f, 0.0f), 0, &RxLocal[localId][32]);

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
        for (int i = 0; i < 64; i++)
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
    for (int i = 0; i < 64; i++)
        sum += RxLocal[i][RxMappings[localId]];
    Rx[outputIndex] = sum;
}
)CLC";