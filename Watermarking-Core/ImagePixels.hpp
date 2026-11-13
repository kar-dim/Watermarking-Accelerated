#pragma once
#include <cstddef>
#include <cstdint>

/*!
 * \brief CPU image helpers with automatic SIMD selection
 * \author Dimitris Karatzas
 */
namespace ImagePixels {
bool supportsAvx512();
void clearTransparent(uint8_t* rgb, const uint8_t* alpha, size_t pixelCount);
void makePreview(const uint8_t* source, uint8_t* preview, size_t pixelCount, int channels);

// Direct interleave kernels accept RGB or RGBA planes
namespace Avx2 {
void clearTransparent(uint8_t* rgb, const uint8_t* alpha, size_t pixelCount);
void interleave(const uint8_t* source, uint8_t* preview, size_t pixelCount, int channels);
} // namespace Avx2

namespace Avx512 {
void clearTransparent(uint8_t* rgb, const uint8_t* alpha, size_t pixelCount);
void interleave(const uint8_t* source, uint8_t* preview, size_t pixelCount, int channels);
} // namespace Avx512
} // namespace ImagePixels
