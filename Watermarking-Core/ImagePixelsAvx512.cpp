#include "ImagePixels.hpp"
#include <cstdint>
#include <immintrin.h>

#if !defined(__AVX512BW__)
#error "ImagePixelsAvx512.cpp must be compiled with AVX-512"
#endif

/*!
 * \brief AVX-512 image helpers called only after CPU and OS feature checks
 * \author Dimitris Karatzas
 */
void ImagePixels::Avx512::clearTransparent(uint8_t* rgb, const uint8_t* alpha, const size_t pixelCount) {
    constexpr size_t lanes = 64;
    const size_t blocks = pixelCount / lanes;
#if !defined(_USE_GPU_)
#pragma omp parallel for schedule(static)
#endif
    for (std::ptrdiff_t block = 0; block < static_cast<std::ptrdiff_t>(blocks); ++block) {
        const size_t offset = static_cast<size_t>(block) * lanes;
        const __mmask64 visible = _mm512_cmpneq_epi8_mask(_mm512_loadu_si512(alpha + offset), _mm512_setzero_si512());
        for (size_t channel = 0; channel < 3; ++channel) {
            uint8_t* plane = rgb + channel * pixelCount + offset;
            _mm512_storeu_si512(plane, _mm512_maskz_mov_epi8(visible, _mm512_loadu_si512(plane)));
        }
    }
    for (size_t pixel = blocks * lanes; pixel < pixelCount; ++pixel)
        for (size_t channel = 0; channel < 3; ++channel)
            rgb[channel * pixelCount + pixel] &= alpha[pixel] ? 0xff : 0;
}

void ImagePixels::Avx512::interleave(const uint8_t* source, uint8_t* preview, const size_t pixelCount, const int channels) {
    constexpr size_t lanes = 16;
    const size_t blocks = pixelCount / lanes;
    const uint8_t* green = source + pixelCount;
    const uint8_t* blue = green + pixelCount;
    const uint8_t* alpha = blue + pixelCount;
    const __m128i rgbMask = _mm_setr_epi8(0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, -1, -1, -1, -1);
    const __m512i packedIndices = _mm512_setr_epi32(0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 0, 0, 0, 0);
#pragma omp parallel for if (pixelCount > 1000000) schedule(static)
    for (std::ptrdiff_t block = 0; block < static_cast<std::ptrdiff_t>(blocks); ++block) {
        const size_t offset = static_cast<size_t>(block) * lanes;
        const __m512i r = _mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(source + offset)));
        const __m512i g = _mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(green + offset)));
        const __m512i b = _mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(blue + offset)));
        __m512i rgba = _mm512_or_si512(r, _mm512_or_si512(_mm512_slli_epi32(g, 8), _mm512_slli_epi32(b, 16)));
        if (channels == 4) {
            const __m512i a = _mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(alpha + offset)));
            rgba = _mm512_or_si512(rgba, _mm512_slli_epi32(a, 24));
            _mm512_storeu_si512(preview + offset * 4, rgba);
        } else {
            // Byte shuffles stop at 128-bit lanes, dword permute closes the gaps
            const __m512i packed = _mm512_shuffle_epi8(rgba, _mm512_broadcast_i32x4(rgbMask));
            _mm512_mask_storeu_epi32(preview + offset * 3, 0x0fff, _mm512_permutexvar_epi32(packedIndices, packed));
        }
    }
    for (size_t pixel = blocks * lanes; pixel < pixelCount; ++pixel)
        for (int channel = 0; channel < channels; ++channel)
            preview[pixel * channels + channel] = source[static_cast<size_t>(channel) * pixelCount + pixel];
}
