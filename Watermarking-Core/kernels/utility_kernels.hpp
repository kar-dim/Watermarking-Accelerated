#pragma once
#include <string>

/*!
 *  \brief  OpenCL kernel source code for planar buffer transposes, RGB-to-gray conversions, and parallel reductions
 *  \author Dimitris Karatzas
 */

inline const std::string utilityKernels = R"CLC(

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

// coalesced tiled transpose: column-major planar uchar (1 or 3 channels) to row-major interleaved uchar (display layout)
__kernel void col_major_to_interleaved_u8(
    const __global uchar* restrict src,
    __global uchar* restrict dst,
    const int width,
    const int height,
    const int channels)
{
    __local uchar tile[3][16][17];
    const int planeSize = width * height;
    const int tileCol = get_group_id(0) * 16;
    const int tileRow = get_group_id(1) * 16;
    const int col = tileCol + get_local_id(1);
    const int row = tileRow + get_local_id(0);
    for (int channel = 0; channel < channels; channel++) {
        if (col < width && row < height)
            tile[channel][get_local_id(1)][get_local_id(0)] = src[channel * planeSize + col * height + row];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // the pixels of a tile row are contiguous in the output, consecutive work-items write consecutive bytes
    const int outRow = tileRow + get_local_id(1);
    if (outRow >= height)
        return;
    const int rowBytes = min(16, width - tileCol) * channels;
    __global uchar* output = dst + (outRow * width + tileCol) * channels;
    for (int byte = get_local_id(0); byte < rowBytes; byte += 16)
        output[byte] = tile[byte % channels][byte / channels][get_local_id(1)];
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

// the source pixel (row-major, as stored in the file) of the displayed pixel (x, y) for the EXIF orientations 1-8, the transforms of rotate() in utils.cpp
static inline int2 orientedSource(const int x, const int y, const int srcWidth, const int srcHeight, const int orientation) {
    switch (orientation) {
    case 2: return (int2)(srcWidth - 1 - x, y);
    case 3: return (int2)(srcWidth - 1 - x, srcHeight - 1 - y);
    case 4: return (int2)(x, srcHeight - 1 - y);
    case 5: return (int2)(y, x);
    case 6: return (int2)(y, srcHeight - 1 - x);
    case 7: return (int2)(srcWidth - 1 - y, srcHeight - 1 - x);
    case 8: return (int2)(srcWidth - 1 - y, x);
    default: return (int2)(x, y);
    }
}

// image upload: row-major planar uchar (1 or 3 channels, CImg layout) -> the displayed image (EXIF orientation 1-8): column-major planar uchar RGB (RGB only),
// column-major float luma and, when display is not null, the row-major interleaved uchar pixels (the original image preview). One coalesced tiled pass
__kernel void orient_row_major_to_col_major(
    const __global uchar* restrict src,
    __global uchar* restrict rgbDst,
    __global float* restrict grayDst,
    __global uchar* restrict display,
    const int srcWidth,
    const int srcHeight,
    const int channels,
    const int orientation)
{
    __local uchar tile[3][32][33];
    const bool swapAxes = orientation >= 5;
    const int width = swapAxes ? srcHeight : srcWidth;
    const int height = swapAxes ? srcWidth : srcHeight;
    const int tileCol = get_group_id(0) * 32;
    const int tileRow = get_group_id(1) * 32;
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);
    // the source pixels of a 32x32 displayed tile are a 32x32 square too, loaded with coalesced reads along the source rows (4 rows per work-item)
    const int2 first = orientedSource(tileCol, tileRow, srcWidth, srcHeight, orientation);
    const int2 last = orientedSource(tileCol + 31, tileRow + 31, srcWidth, srcHeight, orientation);
    const int srcCol0 = min(first.x, last.x);
    const int srcRow0 = min(first.y, last.y);
    const int srcCol = srcCol0 + lx;
    const int srcPlaneSize = srcWidth * srcHeight;
    #pragma unroll
    for (int i = 0; i < 32; i += 8) {
        const int srcRow = srcRow0 + ly + i;
        if (srcCol >= 0 && srcCol < srcWidth && srcRow >= 0 && srcRow < srcHeight) {
            const int srcIdx = srcRow * srcWidth + srcCol;
            if (channels == 3) {
                const uchar r = src[srcIdx];
                const uchar g = src[srcPlaneSize + srcIdx];
                const uchar b = src[2 * srcPlaneSize + srcIdx];
                tile[0][ly + i][lx] = r;
                tile[1][ly + i][lx] = g;
                tile[2][ly + i][lx] = b;
            } else {
                tile[0][ly + i][lx] = src[srcIdx];
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // inside a tile every orientation is a swap of the axes and/or a mirror of each axis: (x, y) in the displayed tile -> (column, row) in the source tile
    const int mirrorX = orientation == 2 || orientation == 3 || orientation == 7 || orientation == 8 ? 31 : 0;
    const int mirrorY = orientation == 3 || orientation == 4 || orientation == 6 || orientation == 7 ? 31 : 0;
    // column-major planes and luma: coalesced writes along the displayed columns
    const int planeSize = width * height;
    const int row = tileRow + lx;
    #pragma unroll
    for (int i = 0; i < 32; i += 8) {
        const int x = ly + i;
        const int col = tileCol + x;
        if (row < height && col < width) {
            const int tileX = (swapAxes ? lx : x) ^ mirrorX;
            const int tileY = (swapAxes ? x : lx) ^ mirrorY;
            const int idx = col * height + row;
            if (channels == 3) {
                const uchar r = tile[0][tileY][tileX];
                const uchar g = tile[1][tileY][tileX];
                const uchar b = tile[2][tileY][tileX];
                rgbDst[idx] = r;
                rgbDst[planeSize + idx] = g;
                rgbDst[2 * planeSize + idx] = b;
                grayDst[idx] = (float)r * K_LUMA_R + (float)g * K_LUMA_G + (float)b * K_LUMA_B;
            } else {
                grayDst[idx] = (float)tile[0][tileY][tileX];
            }
        }
    }
    // the displayed pixels, row-major interleaved: coalesced writes along the displayed rows
    if (display) {
        const bool rgb = channels == 3;
        const int rowBytes = min(32, width - tileCol) * channels;
        for (int y = ly; y < 32 && (tileRow + y) < height; y += 8) {
            __global uchar* output = display + ((tileRow + y) * width + tileCol) * channels;
            for (int byte = lx; byte < rowBytes; byte += 32) {
                const int x = rgb ? byte / 3 : byte;
                output[byte] = tile[byte - x * channels][(swapAxes ? x : y) ^ mirrorY][(swapAxes ? y : x) ^ mirrorX];
            }
        }
    }
}

)CLC";
