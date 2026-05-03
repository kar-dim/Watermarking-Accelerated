#pragma once
#include <string>
inline const std::string utilityKernels = R"CLC(

// ITU-R 601 luma coefficients
#define K_LUMA_R 0.299f
#define K_LUMA_G 0.587f
#define K_LUMA_B 0.114f

// coalesced tiled transpose: row-major float to column-major float, multi-channel via z-dimension
__kernel void row_major_to_col_major_float(
    const __global float* restrict src,
    __global float* restrict dst,
    const int width,
    const int height)
{
    __local float tile[16][17];
    const int planeOffset = get_global_id(2) * width * height;
    const int col = get_group_id(0) * 16 + get_local_id(0);
    const int row = get_group_id(1) * 16 + get_local_id(1);
    if (col < width && row < height)
        tile[get_local_id(1)][get_local_id(0)] = src[planeOffset + row * width + col];
    barrier(CLK_LOCAL_MEM_FENCE);
    const int outRow = get_group_id(1) * 16 + get_local_id(0);
    const int outCol = get_group_id(0) * 16 + get_local_id(1);
    if (outRow < height && outCol < width)
        dst[planeOffset + outRow + outCol * height] = tile[get_local_id(0)][get_local_id(1)];
}

// coalesced tiled transpose: column-major uchar to row-major uchar, multi-channel via z-dimension
__kernel void col_major_to_row_major_u8(
    const __global uchar* restrict src,
    __global uchar* restrict dst,
    const int width,
    const int height)
{
    __local uchar tile[16][17];
    const int planeOffset = get_global_id(2) * width * height;
    const int col = get_group_id(0) * 16 + get_local_id(1);
    const int row = get_group_id(1) * 16 + get_local_id(0);
    if (col < width && row < height)
        tile[get_local_id(0)][get_local_id(1)] = src[planeOffset + col * height + row];
    barrier(CLK_LOCAL_MEM_FENCE);
    const int outRow = get_group_id(1) * 16 + get_local_id(1);
    const int outCol = get_group_id(0) * 16 + get_local_id(0);
    if (outRow < height && outCol < width)
        dst[planeOffset + outRow * width + outCol] = tile[get_local_id(1)][get_local_id(0)];
}

// coalesced tiled transpose: row-major uchar (with pitch) to column-major float
__kernel void pitched_to_float(
    const __global uchar* restrict input,
    __global float* restrict output,
    const int width,
    const int height,
    const int pitch)
{
    __local float tile[16][17];
    const int col = get_group_id(0) * 16 + get_local_id(0);
    const int row = get_group_id(1) * 16 + get_local_id(1);
    if (col < width && row < height)
        tile[get_local_id(1)][get_local_id(0)] = (float)input[row * pitch + col];
    barrier(CLK_LOCAL_MEM_FENCE);
    const int outRow = get_group_id(1) * 16 + get_local_id(0);
    const int outCol = get_group_id(0) * 16 + get_local_id(1);
    if (outRow < height && outCol < width)
        output[outCol * height + outRow] = tile[get_local_id(0)][get_local_id(1)];
}

// uint8 col-major (1 or 3 channel) to float col-major grayscale, with optional RGB weighting
__kernel void u8_to_float_gray(
    const __global uchar* restrict input,
    __global float* restrict output,
    const int planeSize,
    const int numChannels)
{
    const int stride = get_global_size(0);
    for (int i = get_global_id(0); i < planeSize; i += stride) {
        if (numChannels == 3)
            output[i] = (float)input[i] * K_LUMA_R + (float)input[i + planeSize] * K_LUMA_G + (float)input[i + 2 * planeSize] * K_LUMA_B;
        else
            output[i] = (float)input[i];
    }
}

// fused row-major 3-channel RGB to col-major grayscale with ITU-R 601 luma weights
__kernel void row_major_rgb_to_col_major_gray(
    const __global float* restrict src,
    __global float* restrict dst,
    const int width,
    const int height)
{
    __local float tile[16][17];
    const int planeSize = width * height;
    const int col = get_group_id(0) * 16 + get_local_id(0);
    const int row = get_group_id(1) * 16 + get_local_id(1);
    if (col < width && row < height) {
        const int rmIdx = row * width + col;
        tile[get_local_id(1)][get_local_id(0)] = src[rmIdx] * K_LUMA_R + src[planeSize + rmIdx] * K_LUMA_G + src[2 * planeSize + rmIdx] * K_LUMA_B;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    const int outRow = get_group_id(1) * 16 + get_local_id(0);
    const int outCol = get_group_id(0) * 16 + get_local_id(1);
    if (outRow < height && outCol < width)
        dst[outRow + outCol * height] = tile[get_local_id(0)][get_local_id(1)];
}

)CLC";
