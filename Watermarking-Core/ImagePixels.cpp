#include "ImagePixels.hpp"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <intrin.h>

#if !defined(__AVX2__) || defined(__AVX512F__)
#error "ImagePixels.cpp must be compiled with AVX2"
#endif

/*!
 * \brief AVX2 image helpers and CPU feature checks for SIMD selection
 * \author Dimitris Karatzas
 */
namespace {
bool detectAvx512() {
    int info[4];
    __cpuid(info, 0);
    if (info[0] < 7)
        return false;
    __cpuidex(info, 1, 0);
    constexpr uint32_t xsaveAndAvx = (1u << 26) | (1u << 27) | (1u << 28);
    if ((static_cast<uint32_t>(info[2]) & xsaveAndAvx) != xsaveAndAvx)
        return false;
    // Windows must save the YMM, mask and ZMM state before these instructions are safe
    if ((_xgetbv(0) & 0xe6) != 0xe6)
        return false;
    __cpuidex(info, 7, 0);
    // The compiler may use any of F, DQ, CD, BW and VL in the AVX-512 source file
    constexpr uint32_t features = (1u << 16) | (1u << 17) | (1u << 28) | (1u << 30) | (1u << 31);
    return (static_cast<uint32_t>(info[1]) & features) == features;
}

struct Kernels {
    void (*clearTransparent)(uint8_t*, const uint8_t*, size_t);
    void (*interleave)(const uint8_t*, uint8_t*, size_t, int);
};

const Kernels& selectedKernels() {
    static const Kernels kernels =
        ImagePixels::supportsAvx512() ? Kernels{ImagePixels::Avx512::clearTransparent, ImagePixels::Avx512::interleave} : Kernels{ImagePixels::Avx2::clearTransparent, ImagePixels::Avx2::interleave};
    return kernels;
}
} // namespace

bool ImagePixels::supportsAvx512() {
    static const bool supported = detectAvx512();
    return supported;
}

void ImagePixels::clearTransparent(uint8_t* rgb, const uint8_t* alpha, const size_t pixelCount) { selectedKernels().clearTransparent(rgb, alpha, pixelCount); }

void ImagePixels::makePreview(const uint8_t* source, uint8_t* preview, const size_t pixelCount, const int channels) {
    if (channels == 1) {
        constexpr size_t chunkBytes = 64 * 1024;
        const size_t chunks = (pixelCount + chunkBytes - 1) / chunkBytes;
#pragma omp parallel for if (pixelCount > 1000000) schedule(static)
        for (std::ptrdiff_t chunk = 0; chunk < static_cast<std::ptrdiff_t>(chunks); ++chunk) {
            const size_t offset = static_cast<size_t>(chunk) * chunkBytes;
            std::memcpy(preview + offset, source + offset, std::min(chunkBytes, pixelCount - offset));
        }
    } else {
        selectedKernels().interleave(source, preview, pixelCount, channels);
    }
}

void ImagePixels::Avx2::clearTransparent(uint8_t* rgb, const uint8_t* alpha, const size_t pixelCount) {
    constexpr size_t lanes = 32;
    const size_t blocks = pixelCount / lanes;
    // GPU loaders run on prefetch threads, keep their alpha work serial
#if !defined(_USE_GPU_)
#pragma omp parallel for schedule(static)
#endif
    for (std::ptrdiff_t block = 0; block < static_cast<std::ptrdiff_t>(blocks); ++block) {
        const size_t offset = static_cast<size_t>(block) * lanes;
        const __m256i transparent = _mm256_cmpeq_epi8(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(alpha + offset)), _mm256_setzero_si256());
        for (size_t channel = 0; channel < 3; ++channel) {
            auto* pixels = reinterpret_cast<__m256i*>(rgb + channel * pixelCount + offset);
            _mm256_storeu_si256(pixels, _mm256_andnot_si256(transparent, _mm256_loadu_si256(pixels)));
        }
    }
    for (size_t pixel = blocks * lanes; pixel < pixelCount; ++pixel)
        for (size_t channel = 0; channel < 3; ++channel)
            rgb[channel * pixelCount + pixel] &= alpha[pixel] ? 0xff : 0;
}

void ImagePixels::Avx2::interleave(const uint8_t* source, uint8_t* preview, const size_t pixelCount, const int channels) {
    constexpr size_t lanes = 8;
    const size_t blocks = pixelCount / lanes;
    const uint8_t* green = source + pixelCount;
    const uint8_t* blue = green + pixelCount;
    const uint8_t* alpha = blue + pixelCount;
    const __m128i zero = _mm_setzero_si128();
    const __m128i rgbMask = _mm_setr_epi8(0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, -1, -1, -1, -1);
#pragma omp parallel for if (pixelCount > 1000000) schedule(static)
    for (std::ptrdiff_t block = 0; block < static_cast<std::ptrdiff_t>(blocks); ++block) {
        const size_t offset = static_cast<size_t>(block) * lanes;
        const __m128i redBytes = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(source + offset));
        const __m128i greenBytes = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(green + offset));
        const __m128i blueBytes = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(blue + offset));
        const __m128i rg = _mm_unpacklo_epi8(redBytes, greenBytes);
        const __m128i ba = _mm_unpacklo_epi8(blueBytes, channels == 4 ? _mm_loadl_epi64(reinterpret_cast<const __m128i*>(alpha + offset)) : zero);
        const __m128i first = _mm_unpacklo_epi16(rg, ba);
        const __m128i second = _mm_unpackhi_epi16(rg, ba);
        if (channels == 4) {
            _mm_storeu_si128(reinterpret_cast<__m128i*>(preview + offset * 4), first);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(preview + offset * 4 + 16), second);
        } else {
            const __m128i packedFirst = _mm_shuffle_epi8(first, rgbMask);
            const __m128i packedSecond = _mm_shuffle_epi8(second, rgbMask);
            const __m128i front = _mm_or_si128(packedFirst, _mm_slli_si128(packedSecond, 12));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(preview + offset * 3), front);
            _mm_storel_epi64(reinterpret_cast<__m128i*>(preview + offset * 3 + 16), _mm_srli_si128(packedSecond, 4));
        }
    }
    for (size_t pixel = blocks * lanes; pixel < pixelCount; ++pixel)
        for (int channel = 0; channel < channels; ++channel)
            preview[pixel * channels + channel] = source[static_cast<size_t>(channel) * pixelCount + pixel];
}
