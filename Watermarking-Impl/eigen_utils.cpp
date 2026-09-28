#include "buffer.hpp"
#include "common_utils.hpp"
#include "eigen_rgb_array.hpp"
#include "cimg_init.h"
#include "eigen_utils.hpp"
#include "simd.hpp"
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <Eigen/Core>
#include <omp.h>
#include <optional>
#include <utility>
#include <vector>
#include <windows.h>

using namespace Eigen;

namespace {
constexpr int kBlock = 16;
// transposed column j of a block goes in register kColumnRegister[j] (4-bit bit reversal)
constexpr std::array<int, kBlock> kColumnRegister{0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15};

// loads a 16x16 byte block (16 rows "stride" bytes apart) and transposes it with 4 unpack stages (8/16/32/64 bit), col j goes in m[kColumnRegister[j]]
void loadTransposedBlock(const uint8_t* source, const size_t stride, __m128i m[kBlock]) {
    __m128i t[kBlock];
    for (int i = 0; i < kBlock; i++)
        m[i] = _mm_loadu_si128(reinterpret_cast<const __m128i*>(source + i * stride));
    for (int i = 0; i < kBlock / 2; i++) {
        t[i] = _mm_unpacklo_epi8(m[2 * i], m[(2 * i) + 1]);
        t[i + (kBlock / 2)] = _mm_unpackhi_epi8(m[2 * i], m[(2 * i) + 1]);
    }
    for (int i = 0; i < kBlock / 2; i++) {
        m[i] = _mm_unpacklo_epi16(t[2 * i], t[(2 * i) + 1]);
        m[i + (kBlock / 2)] = _mm_unpackhi_epi16(t[2 * i], t[(2 * i) + 1]);
    }
    for (int i = 0; i < kBlock / 2; i++) {
        t[i] = _mm_unpacklo_epi32(m[2 * i], m[(2 * i) + 1]);
        t[i + (kBlock / 2)] = _mm_unpackhi_epi32(m[2 * i], m[(2 * i) + 1]);
    }
    for (int i = 0; i < kBlock / 2; i++) {
        m[i] = _mm_unpacklo_epi64(t[2 * i], t[(2 * i) + 1]);
        m[i + (kBlock / 2)] = _mm_unpackhi_epi64(t[2 * i], t[(2 * i) + 1]);
    }
}

// the 8 low bytes as floats
__m256 lowBytesToFloat(const __m128i bytes) { return _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(bytes)); }

// 16 bytes -> 16 floats
void storeFloats(const __m128i bytes, float* destination) {
    _mm256_storeu_ps(destination, lowBytesToFloat(bytes));
    _mm256_storeu_ps(destination + 8, lowBytesToFloat(_mm_srli_si128(bytes, 8)));
}

// luma of a pixel, two FMAs (in order for the scalar AND the SIMD luma to round the same)
float luma(const uint8_t r, const uint8_t g, const uint8_t b) {
    return std::fma(static_cast<float>(b), CommonUtils::kLumaB, std::fma(static_cast<float>(r), CommonUtils::kLumaR, static_cast<float>(g) * CommonUtils::kLumaG));
}

// luma of 16 pixels
void storeLuma(const __m128i r, const __m128i g, const __m128i b, float* destination) {
    const __m256 kr = _mm256_set1_ps(CommonUtils::kLumaR);
    const __m256 kg = _mm256_set1_ps(CommonUtils::kLumaG);
    const __m256 kb = _mm256_set1_ps(CommonUtils::kLumaB);
    for (int half = 0; half < 2; half++) {
        const int shift = half * 8;
        const __m256 red = lowBytesToFloat(half ? _mm_srli_si128(r, 8) : r);
        const __m256 green = lowBytesToFloat(half ? _mm_srli_si128(g, 8) : g);
        const __m256 blue = lowBytesToFloat(half ? _mm_srli_si128(b, 8) : b);
        _mm256_storeu_ps(destination + shift, _mm256_fmadd_ps(blue, kb, _mm256_fmadd_ps(red, kr, _mm256_mul_ps(green, kg))));
    }
}

// row-major byte plane (height x width) -> its transpose (width x height, row-major) is the col-major plane of a row-major image and back
void transposePlane(const uint8_t* source, uint8_t* destination, const int height, const int width) {
    const int blockRows = height - (height % kBlock);
    const int blockCols = width - (width % kBlock);
    // each thread writes whole destination rows (16 source columns)
#pragma omp parallel for schedule(static)
    for (int col = 0; col < blockCols; col += kBlock) {
        int row = 0;
        for (; row < blockRows; row += kBlock) {
            __m128i m[kBlock];
            loadTransposedBlock(source + (static_cast<size_t>(row) * width) + col, width, m);
            for (int j = 0; j < kBlock; j++)
                _mm_storeu_si128(reinterpret_cast<__m128i*>(destination + (static_cast<size_t>(col + j) * height) + row), m[kColumnRegister[j]]);
        }
        for (; row < height; row++)
            for (int j = 0; j < kBlock; j++)
                destination[(static_cast<size_t>(col + j) * height) + row] = source[(static_cast<size_t>(row) * width) + col + j];
    }
    // tail columns that don't fill a block
    for (int col = blockCols; col < width; col++)
        for (int row = 0; row < height; row++)
            destination[(static_cast<size_t>(col) * height) + row] = source[(static_cast<size_t>(row) * width) + col];
}
} // namespace

namespace eigen_utils {
// column-major 8-bit RGB (+ optional alpha) -> row-major planar CImg
Gray8BufferIO eigenRgbToCimg(const EigenArrayU8RGB& arrayRgb, const std::vector<uint8_t>& alphaChannel) {
    const auto rows = arrayRgb[0].rows();
    const auto cols = arrayRgb[0].cols();
    const int channels = alphaChannel.empty() ? 3 : 4;
    Gray8BufferIO output(static_cast<unsigned int>(cols), static_cast<unsigned int>(rows), 1, channels);
    const size_t planeSize = static_cast<size_t>(rows) * cols;
    for (int channel = 0; channel < 3; channel++)
        transposePlane(arrayRgb[channel].data(), output.data() + (channel * planeSize), static_cast<int>(cols), static_cast<int>(rows));
    if (channels == 4)
        std::memcpy(output.data() + (3 * planeSize), alphaChannel.data(), planeSize);
    return output;
}

Gray8BufferIO eigenGrayToCimg(const Gray8Buffer& arrayGray) {
    const auto rows = arrayGray.rows();
    const auto cols = arrayGray.cols();
    Gray8BufferIO output(static_cast<unsigned int>(cols), static_cast<unsigned int>(rows));
    transposePlane(arrayGray.data(), output.data(), static_cast<int>(cols), static_cast<int>(rows));
    return output;
}

// row-major planar 8-bit RGB (CImg) -> column-major 8-bit RGB + float luma
std::pair<EigenArrayU8RGB, ArrayXXf> cimgToEigenRgbAndGray(const Gray8BufferIO& rgbImage) {
    const int rows = rgbImage.height();
    const int cols = rgbImage.width();
    const size_t planeSize = static_cast<size_t>(rows) * cols;
    const uint8_t* red = rgbImage.data();
    const uint8_t* green = red + planeSize;
    const uint8_t* blue = green + planeSize;
    EigenArrayU8RGB output = makeEigenRGBu8(rows, cols);
    ArrayXXf gray(rows, cols);
    uint8_t* outputRed = output[0].data();
    uint8_t* outputGreen = output[1].data();
    uint8_t* outputBlue = output[2].data();
    float* outputGray = gray.data();
    const auto copyPixel = [&](const size_t inputIndex, const size_t outputIndex) {
        const uint8_t r = red[inputIndex];
        const uint8_t g = green[inputIndex];
        const uint8_t b = blue[inputIndex];
        outputRed[outputIndex] = r;
        outputGreen[outputIndex] = g;
        outputBlue[outputIndex] = b;
        outputGray[outputIndex] = luma(r, g, b);
    };
    const int blockRows = rows - (rows % kBlock);
    const int blockCols = cols - (cols % kBlock);
    // 16x16 blocks of the 3 planes transposed in registers, each thread writes whole output columns
#pragma omp parallel for schedule(static)
    for (int col = 0; col < blockCols; col += kBlock) {
        int row = 0;
        for (; row < blockRows; row += kBlock) {
            const size_t inputOffset = (static_cast<size_t>(row) * cols) + col;
            __m128i r[kBlock];
            __m128i g[kBlock];
            __m128i b[kBlock];
            loadTransposedBlock(red + inputOffset, cols, r);
            loadTransposedBlock(green + inputOffset, cols, g);
            loadTransposedBlock(blue + inputOffset, cols, b);
            for (int j = 0; j < kBlock; j++) {
                const size_t outputOffset = (static_cast<size_t>(col + j) * rows) + row;
                const int k = kColumnRegister[j];
                _mm_storeu_si128(reinterpret_cast<__m128i*>(outputRed + outputOffset), r[k]);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(outputGreen + outputOffset), g[k]);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(outputBlue + outputOffset), b[k]);
                storeLuma(r[k], g[k], b[k], outputGray + outputOffset);
            }
        }
        for (; row < rows; row++)
            for (int j = 0; j < kBlock; j++)
                copyPixel((static_cast<size_t>(row) * cols) + col + j, (static_cast<size_t>(col + j) * rows) + row);
    }
    // tail columns that don't fill a block
    for (int col = blockCols; col < cols; col++)
        for (int row = 0; row < rows; row++)
            copyPixel((static_cast<size_t>(row) * cols) + col, (static_cast<size_t>(col) * rows) + row);
    return {std::move(output), std::move(gray)};
}

ImageBuffer cimgToEigenGray(const Gray8BufferIO& grayImage) {
    const int rows = grayImage.height();
    const int cols = grayImage.width();
    const uint8_t* input = grayImage.data();
    ArrayXXf output(rows, cols);
    float* outputData = output.data();
    const int blockRows = rows - (rows % kBlock);
    const int blockCols = cols - (cols % kBlock);
#pragma omp parallel for schedule(static)
    for (int col = 0; col < blockCols; col += kBlock) {
        int row = 0;
        for (; row < blockRows; row += kBlock) {
            __m128i m[kBlock];
            loadTransposedBlock(input + (static_cast<size_t>(row) * cols) + col, cols, m);
            for (int j = 0; j < kBlock; j++)
                storeFloats(m[kColumnRegister[j]], outputData + (static_cast<size_t>(col + j) * rows) + row);
        }
        for (; row < rows; row++)
            for (int j = 0; j < kBlock; j++)
                outputData[(static_cast<size_t>(col + j) * rows) + row] = input[(static_cast<size_t>(row) * cols) + col + j];
    }
    // tail columns that don't fill a block
    for (int col = blockCols; col < cols; col++)
        for (int row = 0; row < rows; row++)
            outputData[(static_cast<size_t>(col) * rows) + row] = input[(static_cast<size_t>(row) * cols) + col];
    return ImageBuffer(std::move(output));
}

// sets the number of OpenMP (watermarking) threads based on physical cores
// it is used only for video embedding, to improve performance by reducing
// context switching between openmp and ffmpeg's threads
void setThreadsToPhysicalCores() {
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    std::vector<uint8_t> buffer(len);
    auto info = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buffer.data());
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, info, &len))
        return;
    unsigned count = 0;
    char* ptr = reinterpret_cast<char*>(info);
    char* end = ptr + len;
    while (ptr < end) {
        auto p = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(ptr);
        if (p->Relationship == RelationProcessorCore)
            count++;
        ptr += p->Size;
    }
    omp_set_num_threads(count);
    Eigen::setNbThreads(count);
}
} // namespace eigen_utils
