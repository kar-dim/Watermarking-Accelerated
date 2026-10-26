#pragma once
#include <string>

// HDR to SDR tonemap matching zscale and Mobius: bilinear chroma at siting, BT.2020 desaturation,
// BT.1886 gamma, BT.709 YCbCr with 4:2:0 chroma filter and rounding
inline const std::string hdrKernels = R"CLC(
// PQ EOTF and BT.1886 gamma tables on [0, 2] in steps of 1/1023
// values above 1 follow extended curves, output clipping comes last
inline float hdrCurve(__global const float2* table, float value) {
    float position = clamp(value, 0.0f, 2.0f) * 1023.0f;
    int lower = (int)position;
    float2 entry = table[lower];
    return fma(entry.y - entry.x, position - (float)lower, entry.x);
}

// 10-bit chroma samples covering 34x18 luma pixels with replicated edges
// 16-bit storage keeps workgroup local memory under 8 KB
#define CHROMA_TILE_WIDTH 19
#define CHROMA_TILE_HEIGHT 11

// bilinear chroma interpolation from local memory
inline float2 chromaAt(__local const ushort2 (*samples)[CHROMA_TILE_WIDTH], int left, int top, int x, int y, float siteX, float siteY) {
    float u = ((float)x - siteX) * 0.5f, v = ((float)y - siteY) * 0.5f;
    float u0 = floor(u), v0 = floor(v);
    int tileX = (int)u0 - left, tileY = (int)v0 - top;
    float2 top0 = mix(convert_float2(samples[tileY][tileX]), convert_float2(samples[tileY][tileX + 1]), u - u0);
    float2 bottom0 = mix(convert_float2(samples[tileY + 1][tileX]), convert_float2(samples[tileY + 1][tileX + 1]), u - u0);
    return mix(top0, bottom0, v - v0);
}

inline float3 hdrPixel(float y, float2 chroma, float a, float b, float k, __global const float2* pq, __global const float2* gamma) {
    float Y = fma(y, 1.0f / 876.0f, -64.0f / 876.0f);
    float Cb = fma(chroma.x, 1.0f / 896.0f, -512.0f / 896.0f);
    float Cr = fma(chroma.y, 1.0f / 896.0f, -512.0f / 896.0f);
    float3 rgb = (float3)(hdrCurve(pq, fma(1.4746f, Cr, Y)), hdrCurve(pq, fma(-0.16455f, Cb, fma(-0.57135f, Cr, Y))), hdrCurve(pq, fma(1.8814f, Cb, Y)));
    float luminance = fma(0.2627f, rgb.x, fma(0.6780f, rgb.y, 0.0593f * rgb.z));
    float desaturation = fmax(luminance - 2.0f, 1.0e-6f) / fmax(luminance, 1.0e-6f);
    rgb = fma((float3)(desaturation), (float3)(luminance) - rgb, rgb);
    float peak = fmax(fmax(rgb.x, rgb.y), rgb.z);
    float mapped = peak <= 0.3f ? peak : k * (peak + a) / (peak + b);
    rgb *= peak > 1.0e-6f ? mapped / peak : 0.0f;
    float3 linear709 = (float3)(fma(1.6605f, rgb.x, fma(-0.5876f, rgb.y, -0.0728f * rgb.z)),
                               fma(-0.1246f, rgb.x, fma(1.1329f, rgb.y, -0.0083f * rgb.z)),
                               fma(-0.0182f, rgb.x, fma(-0.1006f, rgb.y, 1.1187f * rgb.z)));
    return (float3)(hdrCurve(gamma, linear709.x), hdrCurve(gamma, linear709.y), hdrCurve(gamma, linear709.z));
}

inline float sdrLuma(float3 rgb) {
    return fma(0.2126f, rgb.x, fma(0.7152f, rgb.y, 0.0722f * rgb.z));
}

// 2:1 bilinear filter weight of luma tap at offset from chroma pair first pixel
inline float decimationWeight(int offset, float site) {
    return fmax(0.0f, 1.0f - fabs((float)offset - site) * 0.5f) * 0.5f;
}

// 16x8 workgroup covers 32x16 luma pixels with chroma filter apron
#define TILE_WIDTH 34
#define TILE_HEIGHT 18

__kernel __attribute__((reqd_work_group_size(16, 8, 1)))
void p010_hdr_to_sdr(__global const ushort* restrict source, __global uchar* restrict output, __global float* restrict floatLuma,
    __global const float2* pq, __global const float2* gamma, int width, int height, float a, float b, float k, int writeFloat,
    float siteX, float siteY, int shift) {
    __local ushort2 samples[CHROMA_TILE_HEIGHT][CHROMA_TILE_WIDTH];
    __local float2 chroma[TILE_HEIGHT][TILE_WIDTH];
    // extra column avoids bank conflicts during column-major read
    __local float lumaTile[16][33];
    const int pixels = width * height, chromaWidth = width / 2, chromaHeight = height / 2;
    __global const ushort* cb = source + pixels;
    __global const ushort* cr = cb + pixels / 4;
    const int thread = (int)(get_local_id(1) * 16 + get_local_id(0));
    // load source chroma once into local memory
    const int chromaLeft = (int)get_group_id(0) * 16 - 1, chromaTop = (int)get_group_id(1) * 8 - 1;
    for (int index = thread; index < CHROMA_TILE_WIDTH * CHROMA_TILE_HEIGHT; index += 128) {
        const int cx = clamp(chromaLeft + index % CHROMA_TILE_WIDTH, 0, chromaWidth - 1), cy = clamp(chromaTop + index / CHROMA_TILE_WIDTH, 0, chromaHeight - 1);
        samples[index / CHROMA_TILE_WIDTH][index % CHROMA_TILE_WIDTH] = (ushort2)(cb[cy * chromaWidth + cx] >> shift, cr[cy * chromaWidth + cx] >> shift);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    const int left = (int)get_group_id(0) * 32 - 1, top = (int)get_group_id(1) * 16 - 1;
    for (int index = thread; index < TILE_WIDTH * TILE_HEIGHT; index += 128) {
        const int tileX = index % TILE_WIDTH, tileY = index / TILE_WIDTH;
        const int x = clamp(left + tileX, 0, width - 1), y = clamp(top + tileY, 0, height - 1);
        const float3 rgb = hdrPixel((float)(source[y * width + x] >> shift), chromaAt(samples, chromaLeft, chromaTop, x, y, siteX, siteY), a, b, k, pq, gamma);
        const float luma = sdrLuma(rgb);
        chroma[tileY][tileX] = (float2)((rgb.z - luma) * (1.0f / 1.8556f), (rgb.x - luma) * (1.0f / 1.5748f));
        // workgroup pixels write luma output
        if (tileX >= 1 && tileX <= 32 && tileY >= 1 && tileY <= 16 && left + tileX < width && top + tileY < height) {
            const float scaled = clamp(fma(luma, 219.0f, 16.0f), 0.0f, 255.0f);
            output[y * width + x] = convert_uchar_sat_rte(scaled);
            lumaTile[tileY - 1][tileX - 1] = scaled;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // column-major float luma, consecutive workitems write consecutive rows
    if (writeFloat)
        for (int index = thread; index < 32 * 16; index += 128) {
            const int x = left + 1 + index / 16, y = top + 1 + index % 16;
            if (x < width && y < height) floatLuma[x * height + y] = lumaTile[index % 16][index / 16];
        }
    const int cx = get_global_id(0), cy = get_global_id(1);
    if (cx >= chromaWidth || cy >= chromaHeight) return;
    const int tileX = 2 * (int)get_local_id(0) + 1, tileY = 2 * (int)get_local_id(1) + 1;
    float2 sum = (float2)(0.0f);
    for (int dy = -1; dy <= 2; ++dy) {
        const float rowWeight = decimationWeight(dy, siteY);
        if (rowWeight == 0.0f) continue;
        float2 row = (float2)(0.0f);
        for (int dx = -1; dx <= 2; ++dx)
            row = fma((float2)(decimationWeight(dx, siteX)), chroma[tileY + dy][tileX + dx], row);
        sum = fma((float2)(rowWeight), row, sum);
    }
    output[pixels + cy * chromaWidth + cx] = convert_uchar_sat_rte(fma(sum.x, 224.0f, 128.0f));
    output[pixels + pixels / 4 + cy * chromaWidth + cx] = convert_uchar_sat_rte(fma(sum.y, 224.0f, 128.0f));
}
)CLC";
