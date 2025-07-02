#pragma once
#include <string>
inline const std::string me_p3 = R"CLC(

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

//manual loop unrolled calculation of rx in local memory
void me_p3_rxCalculate(__local half RxLocal[64][36], const int localId, const half x_0, const half x_1, const half x_2, const half x_3, const half x_4, const half x_5, const half x_6, const half x_7, const half x_8)
{
    vstore_half8((float8)(x_0 * x_4, x_1 * x_4, x_2 * x_4, x_3 * x_4, x_5 * x_4, x_6 * x_4, x_7 * x_4, x_8 * x_4), 0, &RxLocal[localId][0]);
}

//manual loop unrolled calculation of Rx in local memory
void me_p3_RxCalculate(__local half RxLocal[64][36], const int localId, const half x_0, const half x_1, const half x_2, const half x_3, const half x_5, const half x_6, const half x_7, const half x_8)
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
        const int col = i / 3;
        const int row = i % 3;
        int globalX = get_group_id(0) * get_local_size(0) + col - 1;
        int globalY = get_group_id(1) * get_local_size(1) + row - 1;
        globalX = max(0, min(globalX, (int)(width - 1)));
        globalY = max(0, min(globalY, (int)(height - 1)));
        vstore_half(input[globalX * height + globalY], 0, &blockValues[row][col]);
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