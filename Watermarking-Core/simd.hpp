#pragma once

/*!
 *  \brief  SIMD instruction set requirement validation (AVX2/AVX-512) and intrinsics header inclusion
 *  \author Dimitris Karatzas
 */

// AVX-512 configurations also define __AVX2__, both builds need the AVX2 intrinsics
#if !defined(__AVX2__)
#error "AVX2 is required: build with AVX2 (default) or AVX-512 enabled"
#endif
#include <immintrin.h>

// one float vector register and the operations the CPU kernels need on it
// keep vectors in plain arrays: std::array<Vec, N> loses their alignment (clang ignores it on template arguments)
namespace Simd {
namespace Avx2 {
using Vec = __m256;
inline constexpr int lanes = 8;
inline Vec zero() { return _mm256_setzero_ps(); }
inline Vec broadcast(const float value) { return _mm256_set1_ps(value); }
inline Vec load(const float* source) { return _mm256_loadu_ps(source); }
inline void store(float* destination, const Vec value) { _mm256_storeu_ps(destination, value); }
inline Vec add(const Vec a, const Vec b) { return _mm256_add_ps(a, b); }
// a * b + c
inline Vec fmadd(const Vec a, const Vec b, const Vec c) { return _mm256_fmadd_ps(a, b, c); }
// c - a * b
inline Vec fnmadd(const Vec a, const Vec b, const Vec c) { return _mm256_fnmadd_ps(a, b, c); }
inline float sum(const Vec value) {
    const __m128 quad = _mm_add_ps(_mm256_castps256_ps128(value), _mm256_extractf128_ps(value, 1));
    const __m128 pair = _mm_add_ps(quad, _mm_movehl_ps(quad, quad));
    return _mm_cvtss_f32(_mm_add_ss(pair, _mm_movehdup_ps(pair)));
}
// each 128-bit lane of low moves down by FLOATS, the same lane of high fills its top
template <int FLOATS>
inline Vec shiftInLanes(const Vec high, const Vec low) {
    return _mm256_castsi256_ps(_mm256_alignr_epi8(_mm256_castps_si256(high), _mm256_castps_si256(low), 4 * FLOATS));
}
} // namespace Avx2

#if defined(__AVX512F__)
using Vec = __m512;
inline constexpr int lanes = 16;
inline Vec zero() { return _mm512_setzero_ps(); }
inline Vec broadcast(const float value) { return _mm512_set1_ps(value); }
inline Vec load(const float* source) { return _mm512_loadu_ps(source); }
inline void store(float* destination, const Vec value) { _mm512_storeu_ps(destination, value); }
inline Vec add(const Vec a, const Vec b) { return _mm512_add_ps(a, b); }
// a * b + c
inline Vec fmadd(const Vec a, const Vec b, const Vec c) { return _mm512_fmadd_ps(a, b, c); }
// c - a * b
inline Vec fnmadd(const Vec a, const Vec b, const Vec c) { return _mm512_fnmadd_ps(a, b, c); }
inline float sum(const Vec value) { return _mm512_reduce_add_ps(value); }
inline Avx2::Vec low(const Vec value) { return _mm512_castps512_ps256(value); }
inline Vec fmaddLow(const Avx2::Vec a, const Avx2::Vec b, const Vec accumulator) { return _mm512_insertf32x8(accumulator, Avx2::fmadd(a, b, low(accumulator)), 0); }
// each 128-bit lane of low moves down by FLOATS, the same lane of high fills its top
template <int FLOATS>
inline Vec shiftInLanes(const Vec high, const Vec low) {
    return _mm512_castsi512_ps(_mm512_alignr_epi8(_mm512_castps_si512(high), _mm512_castps_si512(low), 4 * FLOATS));
}
using Avx2::shiftInLanes;
#else
using namespace Avx2;
#endif

// the vector "offset" floats into a run of memory, quads[q] is the vector loaded 4 * q floats into it
// overlapping loads would split cache lines, the shifts work on registers
template <typename Vec>
inline Vec window(const Vec* quads, const int offset) {
    const Vec low = quads[offset / 4];
    switch (offset % 4) {
    case 1: return shiftInLanes<1>(quads[(offset / 4) + 1], low);
    case 2: return shiftInLanes<2>(quads[(offset / 4) + 1], low);
    case 3: return shiftInLanes<3>(quads[(offset / 4) + 1], low);
    default: return low;
    }
}
// loads that the windows at offsets [0, count) need
constexpr int windowQuads(const int count) { return ((count + 2) / 4) + 1; }
} // namespace Simd
