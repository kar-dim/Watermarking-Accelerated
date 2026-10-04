#pragma once

#include <bit>
#include <cstddef>
#include <limits>
#include <stdexcept>

/*!
 *  \brief  Safe arithmetic checks and overflow guards for buffer and media dimensions
 *  \author Dimitris Karatzas
 */
// Shared size checks for host buffers, GPU allocations and media dimensions
namespace InternalUtils {
// Check multiplication before computing allocation sizes.
inline size_t checkedProduct(const size_t a, const size_t b) {
    if (b != 0 && a > std::numeric_limits<size_t>::max() / b)
        throw std::length_error("Buffer byte count exceeds the supported range");
    return a * b;
}

// Memory pools round allocations up to a representable power of two
inline size_t checkedPowerOfTwo(const size_t bytes) {
    constexpr size_t largest = size_t{1} << (std::numeric_limits<size_t>::digits - 1);
    if (bytes > largest)
        throw std::length_error("Buffer alignment exceeds the supported range");
    return std::bit_ceil(bytes);
}

// Kernels use signed int indices and add padding/alignment to their launch sizes
inline int checkedElements(const int rows, const int cols, const int channels = 1) {
    if (rows < 0 || cols < 0 || channels <= 0)
        throw std::invalid_argument("Invalid buffer dimensions");
    const size_t count = checkedProduct(checkedProduct(static_cast<size_t>(rows), static_cast<size_t>(cols)), static_cast<size_t>(channels));
    constexpr int limit = std::numeric_limits<int>::max() - 4096;
    if (rows > limit || cols > limit || count > static_cast<size_t>(limit))
        throw std::length_error("Buffer dimensions exceed the supported index range");
    return static_cast<int>(count);
}

// Allow room for RGBA expansion before decoding an image
inline void checkImageDimensions(const unsigned int rows, const unsigned int cols) {
    if (rows == 0 || cols == 0 || rows > static_cast<unsigned int>(std::numeric_limits<int>::max()) || cols > static_cast<unsigned int>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("Invalid image dimensions");
    checkedElements(static_cast<int>(rows), static_cast<int>(cols), 4);
}

// Include row padding when validating a video plane's index range
inline void checkVideoPitch(const int rows, const int rowBytes, const int pitchBytes) {
    if (rows <= 0 || rowBytes <= 0 || pitchBytes < rowBytes)
        throw std::invalid_argument("Invalid video plane stride");
    checkedElements(rows, pitchBytes);
}
} // namespace InternalUtils
