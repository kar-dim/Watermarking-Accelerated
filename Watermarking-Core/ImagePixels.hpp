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
void clearTransparentSerial(uint8_t* rgb, const uint8_t* alpha, size_t pixelCount);
void makePreview(const uint8_t* source, uint8_t* preview, size_t pixelCount, int channels);
// serial row interleave safe on async save and background threads
void interleaveRow(const uint8_t* red, const uint8_t* green, const uint8_t* blue, const uint8_t* alpha, uint8_t* destination, size_t pixelCount, int channels);

// Direct interleave kernels accept RGB or RGBA planes
namespace Avx2 {
void clearTransparent(uint8_t* rgb, const uint8_t* alpha, size_t pixelCount);
void clearTransparentSerial(uint8_t* rgb, const uint8_t* alpha, size_t pixelCount);
void interleave(const uint8_t* source, uint8_t* preview, size_t pixelCount, int channels);
void interleaveRow(const uint8_t* red, const uint8_t* green, const uint8_t* blue, const uint8_t* alpha, uint8_t* destination, size_t pixelCount, int channels);
} // namespace Avx2

namespace Avx512 {
void clearTransparent(uint8_t* rgb, const uint8_t* alpha, size_t pixelCount);
void clearTransparentSerial(uint8_t* rgb, const uint8_t* alpha, size_t pixelCount);
void interleave(const uint8_t* source, uint8_t* preview, size_t pixelCount, int channels);
void interleaveRow(const uint8_t* red, const uint8_t* green, const uint8_t* blue, const uint8_t* alpha, uint8_t* destination, size_t pixelCount, int channels);
} // namespace Avx512
} // namespace ImagePixels
