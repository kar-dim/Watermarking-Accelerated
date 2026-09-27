#pragma once
#include <string>

// The OpenCL watermark generation. For bit-exactness we use fma(), correctly rounded sqrt and FP_CONTRACT OFF
inline const std::string generationKernels = R"CLC(
#pragma OPENCL FP_CONTRACT OFF
#define WM_INLINE static inline __attribute__((always_inline))

// ChaCha20 quarter round on the words a, b, c, d of the state: add, rotate, xor
WM_INLINE void quarterRound(uint* restrict x, const int a, const int b, const int c, const int d) {
    x[a] += x[b];
    x[d] = rotate(x[d] ^ x[a], 16u);
    x[c] += x[d];
    x[b] = rotate(x[b] ^ x[c], 12u);
    x[a] += x[b];
    x[d] = rotate(x[d] ^ x[a], 8u);
    x[c] += x[d];
    x[b] = rotate(x[b] ^ x[c], 7u);
}

// natural logarithm (Cephes polynomial)
WM_INLINE float logReference(const float x) {
    // frexp: mantissa in [0.5, 1) plus the matching exponent
    const uint bits = as_uint(x);
    float m = as_float((bits & 0x807FFFFFu) | 0x3F000000u);
    float e = (float)((int)((bits & 0x7F800000u) >> 23) - 126);

    // keep the mantissa near 1: if (m < sqrt(0.5)) { e -= 1; m = 2m - 1; } else { m -= 1; }
    const bool belowSqrtHalf = m < 0.707106781186547524f;
    e = belowSqrtHalf ? e - 1.0f : e;
    m = (belowSqrtHalf ? m + m : m) - 1.0f;

    const float z = m * m;
    float y = 7.0376836292E-2f;
    y = fma(y, m, -1.1514610310E-1f);
    y = fma(y, m, 1.1676998740E-1f);
    y = fma(y, m, -1.2420140846E-1f);
    y = fma(y, m, 1.4249322787E-1f);
    y = fma(y, m, -1.6668057665E-1f);
    y = fma(y, m, 2.0000714765E-1f);
    y = fma(y, m, -2.4999993993E-1f);
    y = fma(y, m, 3.3333331174E-1f);
    y = (y * m) * z;
    // recombine, using the hi/lo split of ln(2) for the exponent term
    y = fma(e, -2.12194440e-4f, y);
    y = fma(z, -0.5f, y);
    return fma(e, 0.693359375f, m + y);
}

// sine and cosine (Cephes minimax polynomials)
WM_INLINE float2 sinCosReference(const float x) {
    // reduce into an octant: j = ((int)(x * 4/pi) + 1) & ~1
    const int j = ((int)(x * 1.27323954473516f) + 1) & ~1;
    const float y = (float)j;

    // extended precision modular arithmetic: r = ((x - y*DP1) - y*DP2) - y*DP3
    float r = fma(y, -0.78515625f, x);
    r = fma(y, -2.4187564849853515625e-4f, r);
    r = fma(y, -3.77489497744594108e-8f, r);

    // cosine and sine polynomials of the reduced argument
    const float z = r * r;
    float cosPoly = 2.443315711809948E-005f;
    cosPoly = fma(cosPoly, z, -1.388731625493765E-003f);
    cosPoly = fma(cosPoly, z, 4.166664568298827E-002f);
    cosPoly = (cosPoly * z) * z;
    cosPoly = fma(z, -0.5f, cosPoly);
    cosPoly = cosPoly + 1.0f;
    float sinPoly = -1.9515295891E-4f;
    sinPoly = fma(sinPoly, z, 8.3321608736E-3f);
    sinPoly = fma(sinPoly, z, -1.6666654611E-1f);
    sinPoly = fma(sinPoly * z, r, r);

    // the octant bits decide which polynomial is sin and which is cos, and the sign of each (x = sin, y = cos)
    const bool sinFromSinPoly = (j & 2) == 0;
    const float sinValue = as_float(as_uint(sinFromSinPoly ? sinPoly : cosPoly) ^ ((uint)(j & 4) << 29));
    const float cosValue = as_float(as_uint(sinFromSinPoly ? cosPoly : sinPoly) ^ ((uint)(~(j - 2) & 4) << 29));
    return (float2)(sinValue, cosValue);
}

// correctly rounded sqrt from the builtin one (up to 3 ulp): y = RN(sqrt(x)) exactly when y * below(y) < x <= y * above(y)
// fma gives the exact sign of both differences. fma() is correctly rounded on every device (the OpenCL C specification)
WM_INLINE float sqrtCorrectlyRounded(const float x) {
    float y = sqrt(x);
    if (x == 0.0f) // keeps the sign of zero
        return y;
    // y is positive and finite: its neighbours are the next and the previous bit patterns
    for (int step = 0; step < 4; step++) {
        const float above = as_float(as_uint(y) + 1);
        const float below = as_float(as_uint(y) - 1);
        if (fma(-y, above, x) > 0.0f)
            y = above;
        else if (fma(-y, below, x) <= 0.0f)
            y = below;
        else
            break;
    }
    return y;
}

// two 24-bit random values -> two normally distributed values (Box-Muller transform)
WM_INLINE float2 boxMullerPair(const uint random1, const uint random2) {
    // uniform in (0, 1]: (random + 1) * 2^-24, exact
    const float u1 = fma((float)random1, 0x1.0p-24f, 0x1.0p-24f);
    const float u2 = fma((float)random2, 0x1.0p-24f, 0x1.0p-24f);
    const float radius = sqrtCorrectlyRounded(-2.0f * logReference(u1));
    const float2 sinCosTheta = sinCosReference(TWO_PI * u2);
    return (float2)(radius * sinCosTheta.y, radius * sinCosTheta.x);
}

// generates the watermark: one ChaCha20 block per work-item, 8 standard normal values (Box-Muller) rounded to half (toHalfBits, halfKernels)
__kernel void generate_watermark(const uint16 baseState, __global ushort* restrict watermark, const ulong numElements) {
    const ulong block = get_global_id(0);
    const ulong first = block * 8;
    if (first >= numElements)
        return;
    // ChaCha20 block "block": the 64-bit block counter in words 12-13, 20 rounds, then the final addition
    // the loops are unrolled: every index is a constant and the arrays stay in registers
    uint key[16];
    vstore16(baseState, 0, key);
    key[12] = (uint)block;
    key[13] = (uint)(block >> 32);
    uint x[16];
    #pragma unroll
    for (int i = 0; i < 16; i++)
        x[i] = key[i];
    #pragma unroll
    for (int i = 0; i < 10; i++) {
        quarterRound(x, 0, 4, 8, 12);
        quarterRound(x, 1, 5, 9, 13);
        quarterRound(x, 2, 6, 10, 14);
        quarterRound(x, 3, 7, 11, 15);
        quarterRound(x, 0, 5, 10, 15);
        quarterRound(x, 1, 6, 11, 12);
        quarterRound(x, 2, 7, 8, 13);
        quarterRound(x, 3, 4, 9, 14);
    }
    #pragma unroll
    for (int i = 0; i < 16; i++)
        x[i] += key[i];

    // the block is eight 64-bit values (little endian word pairs), pair j uses the top 24 bits of values 2j and 2j + 1: the high words >> 8
    const float2 pair0 = boxMullerPair(x[1] >> 8, x[3] >> 8);
    const float2 pair1 = boxMullerPair(x[5] >> 8, x[7] >> 8);
    const float2 pair2 = boxMullerPair(x[9] >> 8, x[11] >> 8);
    const float2 pair3 = boxMullerPair(x[13] >> 8, x[15] >> 8);
    const ushort8 halves = (ushort8)(toHalfBits(pair0.x), toHalfBits(pair0.y), toHalfBits(pair1.x), toHalfBits(pair1.y), toHalfBits(pair2.x), toHalfBits(pair2.y),
        toHalfBits(pair3.x), toHalfBits(pair3.y));
    if (first + 8 <= numElements) {
        vstore8(halves, block, watermark);
        return;
    }
    // the partial last block (at most 7 values)
    ushort halfArray[8];
    vstore8(halves, 0, halfArray);
    const ulong remaining = numElements - first;
    #pragma unroll
    for (int i = 0; i < 7; i++) {
        if (i < remaining)
            watermark[first + i] = halfArray[i];
    }
}

// Box-Muller transform of (x1, x2) pairs of 24-bit random values, the transform of generate_watermark
__kernel void box_muller_pairs(__global const uint* restrict randomPairs, __global float* restrict normals, const int pairs) {
    const int pair = get_global_id(0);
    if (pair < pairs)
        vstore2(boxMullerPair(randomPairs[2 * pair], randomPairs[(2 * pair) + 1]), pair, normals);
}
)CLC";
