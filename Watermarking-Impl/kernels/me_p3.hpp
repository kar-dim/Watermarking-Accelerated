#pragma once
#include <string>
inline const std::string me_p3 = R"CLC(

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

//manual loop unrolled calculation of rx in local memory
void me_p3_rxCalculate(__local half RxLocal[64][36], const int localId, const float x_0, const float x_1, const float x_2, const float x_3, const float currentPixel, const float x_5, const float x_6, const float x_7, const float x_8)
{
    vstore_half8((float8)(x_0 * currentPixel, x_1 * currentPixel, x_2 * currentPixel, x_3 * currentPixel, x_5 * currentPixel, x_6 * currentPixel, x_7 * currentPixel, x_8 * currentPixel), 0, &RxLocal[localId][0]);
}

//manual loop unrolled calculation of Rx in local memory
void me_p3_RxCalculate(__local half RxLocal[64][36], const int localId, const float x_0, const float x_1, const float x_2, const float x_3, const float x_5, const float x_6, const float x_7, const float x_8)
{
    vstore_half8((float8)(x_0 * x_0, x_0 * x_1, x_0 * x_2, x_0 * x_3, x_0 * x_5, x_0 * x_6, x_0 * x_7, x_0 * x_8), 0, &RxLocal[localId][0]);
    vstore_half8((float8)(x_1 * x_1, x_1 * x_2, x_1 * x_3, x_1 * x_5, x_1 * x_6, x_1 * x_7, x_1 * x_8, x_2 * x_2), 0, &RxLocal[localId][8]);
    vstore_half8((float8)(x_2 * x_3, x_2 * x_5, x_2 * x_6, x_2 * x_7, x_2 * x_8, x_3 * x_3, x_3 * x_5, x_3 * x_6), 0, &RxLocal[localId][16]);
    vstore_half8((float8)(x_3 * x_7, x_3 * x_8, x_5 * x_5, x_5 * x_6, x_5 * x_7, x_5 * x_8, x_6 * x_6, x_6 * x_7), 0, &RxLocal[localId][24]);
    vstore_half4((float4)(x_6 * x_8, x_7 * x_7, x_7 * x_8, x_8 * x_8), 0, &RxLocal[localId][32]);
}

__kernel void me(__read_only image2d_t image,
    __global float* __restrict__ Rx,
    __global float* __restrict__ rx,
    __constant int* __restrict__ RxMappings,
    __local half RxLocal[64][36]) //64 local threads, 36 values each (8 for rx, this is a shared memory for both Rx,rx)

{
    const int x = get_global_id(0), y = get_global_id(1);
    const int width = get_image_height(image); //image2d is transposed, so we read the opposite dimensions
    const int paddedWidth = get_global_size(0);
    const int localId = get_local_id(0);
    const int outputIndex = (y * paddedWidth) + x;

    //initialize shared memory, assign a portion for all threads for parallelism
    #pragma unroll
    for (int i = 0; i < 4; i++)
        vstore_half8((float8)(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f), 0, &RxLocal[localId][i * 8]);
    vstore_half4((float4)(0.0f, 0.0f, 0.0f, 0.0f), 0, &RxLocal[localId][32]);

    float x_0, x_1, x_2, x_3, currentPixel, x_5, x_6, x_7, x_8;
    if (x < width)
    {
        const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
        x_0 = read_imagef(image, sampler, (int2)(y - 1, x - 1)).x;
        x_1 = read_imagef(image, sampler, (int2)(y - 1, x)).x;
        x_2 = read_imagef(image, sampler, (int2)(y - 1, x + 1)).x;
        x_3 = read_imagef(image, sampler, (int2)(y, x - 1)).x;
        currentPixel = read_imagef(image, sampler, (int2)(y, x)).x;
        x_5 = read_imagef(image, sampler, (int2)(y, x + 1)).x;
        x_6 = read_imagef(image, sampler, (int2)(y + 1, x - 1)).x;
        x_7 = read_imagef(image, sampler, (int2)(y + 1, x)).x;
        x_8 = read_imagef(image, sampler, (int2)(y + 1, x + 1)).x;
        me_p3_rxCalculate(RxLocal, localId, x_0, x_1, x_2, x_3, currentPixel, x_5, x_6, x_7, x_8);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //simplified summation for rx
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

    //calculate 36 Rx values
    if (x < width)
        me_p3_RxCalculate(RxLocal, localId, x_0, x_1, x_2, x_3, x_5, x_6, x_7, x_8);
    barrier(CLK_LOCAL_MEM_FENCE);

    //simplified summation for Rx
    //TODO can be optimized
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 64; i++)
        sum += RxLocal[i][RxMappings[localId]];
    Rx[outputIndex] = sum;
}
)CLC";