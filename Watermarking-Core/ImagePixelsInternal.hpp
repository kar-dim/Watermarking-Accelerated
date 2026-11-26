#pragma once
#include <cstddef>
#include <cstdint>
#include <immintrin.h>

/*!
 *  \brief  Internal shared SIMD constants and scalar tail helpers for pixel loops
 *  \author Dimitris Karatzas
 */
namespace ImagePixelsInternal {
inline constexpr size_t kOmpPixelThreshold = 32768;

inline __m128i rgbPackMask() { return _mm_setr_epi8(0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, -1, -1, -1, -1); }

inline void interleaveTailScalar(const uint8_t* red, const uint8_t* green, const uint8_t* blue, const uint8_t* alpha, uint8_t* dst, const size_t start, const size_t pixelCount, const int channels) {
    for (size_t pixel = start; pixel < pixelCount; ++pixel) {
        dst[pixel * channels + 0] = red[pixel];
        dst[pixel * channels + 1] = green[pixel];
        dst[pixel * channels + 2] = blue[pixel];
        if (channels == 4)
            dst[pixel * channels + 3] = alpha[pixel];
    }
}

inline void clearTransparentTailScalar(uint8_t* rgb, const uint8_t* alpha, const size_t start, const size_t pixelCount) {
    for (size_t pixel = start; pixel < pixelCount; ++pixel)
        for (size_t channel = 0; channel < 3; ++channel)
            rgb[channel * pixelCount + pixel] &= alpha[pixel] ? 0xff : 0;
}
} // namespace ImagePixelsInternal
