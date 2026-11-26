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
#include <stdexcept>
#include <utility>
#include <vector>
#include <windows.h>

/*!
 *  \brief  Eigen-based matrix operations, SIMD shift matrix assembly, and solver implementations
 *  \author Dimitris Karatzas
 */

using namespace Eigen;

namespace {
constexpr int kBlock = 16;
constexpr size_t kOmpPixelThreshold = 32768;

__m128i reverseBytes16(const __m128i v) { return _mm_shuffle_epi8(v, _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)); }

// loads a 16x16 byte block (16 rows "stride" bytes apart) and transposes it with 4 unpack stages (8/16/32/64 bit), col j goes in m[Simd::kTransposeRegister[j]]
void loadTransposedBlock(const uint8_t* source, const size_t stride, __m128i m[kBlock]) {
    for (int i = 0; i < kBlock; i++)
        m[i] = _mm_loadu_si128(reinterpret_cast<const __m128i*>(source + i * stride));
    Simd::transposeU8_16x16(m);
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

void transposeBlockStripe(const uint8_t* source, uint8_t* destination, const int height, const int width, const int blockRows, const int col) {
    int row = 0;
    for (; row < blockRows; row += kBlock) {
        __m128i m[kBlock];
        loadTransposedBlock(source + (static_cast<size_t>(row) * width) + col, width, m);
        for (int j = 0; j < kBlock; j++)
            _mm_storeu_si128(reinterpret_cast<__m128i*>(destination + (static_cast<size_t>(col + j) * height) + row), m[Simd::kTransposeRegister[j]]);
    }
    for (; row < height; row++)
        for (int j = 0; j < kBlock; j++)
            destination[(static_cast<size_t>(col + j) * height) + row] = source[(static_cast<size_t>(row) * width) + col + j];
}

void transposeTailColumns(const uint8_t* source, uint8_t* destination, const int height, const int width, const int blockCols) {
    for (int col = blockCols; col < width; col++)
        for (int row = 0; row < height; row++)
            destination[(static_cast<size_t>(col) * height) + row] = source[(static_cast<size_t>(row) * width) + col];
}

// row-major byte plane (height x width) -> its transpose (width x height, row-major) is the col-major plane of a row-major image and back
void transposePlane(const uint8_t* source, uint8_t* destination, const int height, const int width, const bool parallel) {
    const int blockRows = height - (height % kBlock);
    const int blockCols = width - (width % kBlock);
    const bool useParallel = parallel && (static_cast<size_t>(height) * width >= kOmpPixelThreshold);
    // each thread writes whole destination rows (16 source columns)
#pragma omp parallel for if (useParallel) schedule(static)
    for (int col = 0; col < blockCols; col += kBlock)
        transposeBlockStripe(source, destination, height, width, blockRows, col);
    transposeTailColumns(source, destination, height, width, blockCols);
}
} // namespace

namespace eigen_utils {
// column-major 8-bit RGB (+ optional alpha) -> row-major planar CImg
Gray8BufferIO eigenRgbToCimg(const EigenArrayU8RGB& arrayRgb, const std::vector<uint8_t>& alphaChannel, const bool parallel) {
    const auto rows = arrayRgb[0].rows();
    const auto cols = arrayRgb[0].cols();
    const int height = static_cast<int>(cols);
    const int width = static_cast<int>(rows);
    const int blockRows = height - (height % kBlock);
    const int blockCols = width - (width % kBlock);
    const int colBlocks = blockCols / kBlock;
    const int channels = alphaChannel.empty() ? 3 : 4;
    Gray8BufferIO output(static_cast<unsigned int>(cols), static_cast<unsigned int>(rows), 1, channels);
    const size_t planeSize = static_cast<size_t>(rows) * cols;
    const bool useParallel = parallel && (planeSize >= kOmpPixelThreshold);
#pragma omp parallel for if (useParallel) schedule(static)
    for (int task = 0; task < 3 * colBlocks; task++) {
        const int channel = task / colBlocks;
        const int col = (task % colBlocks) * kBlock;
        transposeBlockStripe(arrayRgb[channel].data(), output.data() + (channel * planeSize), height, width, blockRows, col);
    }
    for (int channel = 0; channel < 3; channel++)
        transposeTailColumns(arrayRgb[channel].data(), output.data() + (channel * planeSize), height, width, blockCols);
    if (channels == 4) {
        if (alphaChannel.size() != planeSize)
            throw std::logic_error("Invalid alpha plane size");
        std::memcpy(output.data() + (3 * planeSize), alphaChannel.data(), planeSize);
    }
    return output;
}

Gray8BufferIO eigenGrayToCimg(const Gray8Buffer& arrayGray, const bool parallel) {
    const auto rows = arrayGray.rows();
    const auto cols = arrayGray.cols();
    Gray8BufferIO output(static_cast<unsigned int>(cols), static_cast<unsigned int>(rows));
    transposePlane(arrayGray.data(), output.data(), static_cast<int>(cols), static_cast<int>(rows), parallel);
    return output;
}

// row-major planar 8-bit RGB (CImg) -> column-major 8-bit RGB + float luma, fused with EXIF orientation 1..8
std::pair<EigenArrayU8RGB, ArrayXXf> cimgToEigenRgbAndGray(const Gray8BufferIO& rgbImage, const bool parallel, const int orientation) {
    const int srcW = rgbImage.width();
    const int srcH = rgbImage.height();
    const int orient = (orientation >= 1 && orientation <= 8) ? orientation : 1;
    const bool swapAxes = orient >= 5;
    const int rows = swapAxes ? srcW : srcH;
    const int cols = swapAxes ? srcH : srcW;
    const size_t planeSize = static_cast<size_t>(srcW) * srcH;
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
    const bool useParallel = parallel && (planeSize >= kOmpPixelThreshold);
    const int blockSrcW = srcW - (srcW % kBlock);
    if (swapAxes) {
        const bool flipRow = (orient == 6 || orient == 7);
        const bool flipCol = (orient == 7 || orient == 8);
#pragma omp parallel for if (useParallel) schedule(static)
        for (int dstCol = 0; dstCol < cols; dstCol++) {
            const int sy = flipRow ? (srcH - 1 - dstCol) : dstCol;
            const size_t rowBase = static_cast<size_t>(sy) * srcW;
            const size_t colBase = static_cast<size_t>(dstCol) * rows;
            int sx = 0;
            for (; sx < blockSrcW; sx += kBlock) {
                __m128i r = _mm_loadu_si128(reinterpret_cast<const __m128i*>(red + rowBase + sx));
                __m128i g = _mm_loadu_si128(reinterpret_cast<const __m128i*>(green + rowBase + sx));
                __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(blue + rowBase + sx));
                if (flipCol) {
                    r = reverseBytes16(r);
                    g = reverseBytes16(g);
                    b = reverseBytes16(b);
                }
                const int dstRow = flipCol ? (rows - sx - kBlock) : sx;
                const size_t outOff = colBase + dstRow;
                _mm_storeu_si128(reinterpret_cast<__m128i*>(outputRed + outOff), r);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(outputGreen + outOff), g);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(outputBlue + outOff), b);
                storeLuma(r, g, b, outputGray + outOff);
            }
            for (; sx < srcW; sx++) {
                const int dstRow = flipCol ? (rows - 1 - sx) : sx;
                copyPixel(rowBase + sx, colBase + dstRow);
            }
        }
        return {std::move(output), std::move(gray)};
    }
    const int blockSrcH = srcH - (srcH % kBlock);
    const bool mirrorX = (orient == 2 || orient == 3);
    const bool mirrorY = (orient == 3 || orient == 4);
#pragma omp parallel for if (useParallel) schedule(static)
    for (int sx0 = 0; sx0 < blockSrcW; sx0 += kBlock) {
        int sy = 0;
        for (; sy < blockSrcH; sy += kBlock) {
            const size_t inputOffset = (static_cast<size_t>(sy) * srcW) + sx0;
            __m128i r[kBlock];
            __m128i g[kBlock];
            __m128i b[kBlock];
            loadTransposedBlock(red + inputOffset, srcW, r);
            loadTransposedBlock(green + inputOffset, srcW, g);
            loadTransposedBlock(blue + inputOffset, srcW, b);
            const int dstRow = mirrorY ? (rows - sy - kBlock) : sy;
            for (int j = 0; j < kBlock; j++) {
                const int dstCol = mirrorX ? (cols - 1 - (sx0 + j)) : (sx0 + j);
                const size_t outputOffset = (static_cast<size_t>(dstCol) * rows) + dstRow;
                const int k = Simd::kTransposeRegister[j];
                const __m128i vr = mirrorY ? reverseBytes16(r[k]) : r[k];
                const __m128i vg = mirrorY ? reverseBytes16(g[k]) : g[k];
                const __m128i vb = mirrorY ? reverseBytes16(b[k]) : b[k];
                _mm_storeu_si128(reinterpret_cast<__m128i*>(outputRed + outputOffset), vr);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(outputGreen + outputOffset), vg);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(outputBlue + outputOffset), vb);
                storeLuma(vr, vg, vb, outputGray + outputOffset);
            }
        }
        for (; sy < srcH; sy++) {
            const int dstRow = mirrorY ? (rows - 1 - sy) : sy;
            for (int j = 0; j < kBlock; j++) {
                const int dstCol = mirrorX ? (cols - 1 - (sx0 + j)) : (sx0 + j);
                copyPixel((static_cast<size_t>(sy) * srcW) + sx0 + j, (static_cast<size_t>(dstCol) * rows) + dstRow);
            }
        }
    }
    for (int sx = blockSrcW; sx < srcW; sx++) {
        const int dstCol = mirrorX ? (cols - 1 - sx) : sx;
        for (int sy = 0; sy < srcH; sy++) {
            const int dstRow = mirrorY ? (rows - 1 - sy) : sy;
            copyPixel((static_cast<size_t>(sy) * srcW) + sx, (static_cast<size_t>(dstCol) * rows) + dstRow);
        }
    }
    return {std::move(output), std::move(gray)};
}

void pitchedRowMajorU8ToColMajorFloat(const uint8_t* source, const int rows, const int cols, const int srcPitch, float* destination, const bool parallel) {
    const int blockRows = rows - (rows % kBlock);
    const int blockCols = cols - (cols % kBlock);
    const bool useParallel = parallel && (static_cast<size_t>(rows) * cols >= kOmpPixelThreshold);
#pragma omp parallel for if (useParallel) schedule(static)
    for (int col = 0; col < blockCols; col += kBlock) {
        int row = 0;
        for (; row < blockRows; row += kBlock) {
            __m128i m[kBlock];
            loadTransposedBlock(source + (static_cast<size_t>(row) * srcPitch) + col, srcPitch, m);
            for (int j = 0; j < kBlock; j++)
                storeFloats(m[Simd::kTransposeRegister[j]], destination + (static_cast<size_t>(col + j) * rows) + row);
        }
        for (; row < rows; row++)
            for (int j = 0; j < kBlock; j++)
                destination[(static_cast<size_t>(col + j) * rows) + row] = source[(static_cast<size_t>(row) * srcPitch) + col + j];
    }
    // tail columns that don't fill a block
    for (int col = blockCols; col < cols; col++)
        for (int row = 0; row < rows; row++)
            destination[(static_cast<size_t>(col) * rows) + row] = source[(static_cast<size_t>(row) * srcPitch) + col];
}

ImageBuffer cimgToEigenGray(const Gray8BufferIO& grayImage, const bool parallel, const int orientation) {
    const int srcW = grayImage.width();
    const int srcH = grayImage.height();
    const int orient = (orientation >= 1 && orientation <= 8) ? orientation : 1;
    if (orient == 1) {
        ArrayXXf output(srcH, srcW);
        pitchedRowMajorU8ToColMajorFloat(grayImage.data(), srcH, srcW, srcW, output.data(), parallel);
        return ImageBuffer(std::move(output));
    }
    const bool swapAxes = orient >= 5;
    const int rows = swapAxes ? srcW : srcH;
    const int cols = swapAxes ? srcH : srcW;
    ArrayXXf output(rows, cols);
    const uint8_t* src = grayImage.data();
    float* dst = output.data();
    const bool useParallel = parallel && (static_cast<size_t>(srcW) * srcH >= kOmpPixelThreshold);
    const int blockSrcW = srcW - (srcW % kBlock);
    if (swapAxes) {
        const bool flipRow = (orient == 6 || orient == 7);
        const bool flipCol = (orient == 7 || orient == 8);
#pragma omp parallel for if (useParallel) schedule(static)
        for (int dstCol = 0; dstCol < cols; dstCol++) {
            const int sy = flipRow ? (srcH - 1 - dstCol) : dstCol;
            const size_t rowBase = static_cast<size_t>(sy) * srcW;
            const size_t colBase = static_cast<size_t>(dstCol) * rows;
            int sx = 0;
            for (; sx < blockSrcW; sx += kBlock) {
                __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + rowBase + sx));
                if (flipCol)
                    v = reverseBytes16(v);
                const int dstRow = flipCol ? (rows - sx - kBlock) : sx;
                storeFloats(v, dst + colBase + dstRow);
            }
            for (; sx < srcW; sx++) {
                const int dstRow = flipCol ? (rows - 1 - sx) : sx;
                dst[colBase + dstRow] = static_cast<float>(src[rowBase + sx]);
            }
        }
        return ImageBuffer(std::move(output));
    }
    const int blockSrcH = srcH - (srcH % kBlock);
    const bool mirrorX = (orient == 2 || orient == 3);
    const bool mirrorY = (orient == 3 || orient == 4);
#pragma omp parallel for if (useParallel) schedule(static)
    for (int sx0 = 0; sx0 < blockSrcW; sx0 += kBlock) {
        int sy = 0;
        for (; sy < blockSrcH; sy += kBlock) {
            __m128i m[kBlock];
            loadTransposedBlock(src + (static_cast<size_t>(sy) * srcW) + sx0, srcW, m);
            const int dstRow = mirrorY ? (rows - sy - kBlock) : sy;
            for (int j = 0; j < kBlock; j++) {
                const int dstCol = mirrorX ? (cols - 1 - (sx0 + j)) : (sx0 + j);
                const __m128i v = mirrorY ? reverseBytes16(m[Simd::kTransposeRegister[j]]) : m[Simd::kTransposeRegister[j]];
                storeFloats(v, dst + (static_cast<size_t>(dstCol) * rows) + dstRow);
            }
        }
        for (; sy < srcH; sy++) {
            const int dstRow = mirrorY ? (rows - 1 - sy) : sy;
            for (int j = 0; j < kBlock; j++) {
                const int dstCol = mirrorX ? (cols - 1 - (sx0 + j)) : (sx0 + j);
                dst[(static_cast<size_t>(dstCol) * rows) + dstRow] = static_cast<float>(src[(static_cast<size_t>(sy) * srcW) + sx0 + j]);
            }
        }
    }
    for (int sx = blockSrcW; sx < srcW; sx++) {
        const int dstCol = mirrorX ? (cols - 1 - sx) : sx;
        for (int sy = 0; sy < srcH; sy++) {
            const int dstRow = mirrorY ? (rows - 1 - sy) : sy;
            dst[(static_cast<size_t>(dstCol) * rows) + dstRow] = static_cast<float>(src[(static_cast<size_t>(sy) * srcW) + sx]);
        }
    }
    return ImageBuffer(std::move(output));
}

ImageBuffer u8RgbToFloatGray(const EigenArrayU8RGB& rgb, const bool parallel) {
    const int rows = static_cast<int>(rgb[0].rows());
    const int cols = static_cast<int>(rgb[0].cols());
    ArrayXXf output(rows, cols);
    const size_t total = static_cast<size_t>(rows) * cols;
    const size_t chunks = total / kBlock;
    const uint8_t* r = rgb[0].data();
    const uint8_t* g = rgb[1].data();
    const uint8_t* b = rgb[2].data();
    float* dst = output.data();
    const bool useParallel = parallel && (total >= kOmpPixelThreshold);
#pragma omp parallel for if (useParallel) schedule(static)
    for (int64_t chunk = 0; chunk < static_cast<int64_t>(chunks); chunk++) {
        const size_t offset = static_cast<size_t>(chunk) * kBlock;
        storeLuma(_mm_loadu_si128(reinterpret_cast<const __m128i*>(r + offset)), _mm_loadu_si128(reinterpret_cast<const __m128i*>(g + offset)),
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + offset)), dst + offset);
    }
    for (size_t i = chunks * kBlock; i < total; i++)
        dst[i] = luma(r[i], g[i], b[i]);
    return ImageBuffer(std::move(output));
}

ImageBuffer u8GrayToFloat(const Gray8Buffer& gray, const bool parallel) {
    const int rows = static_cast<int>(gray.rows());
    const int cols = static_cast<int>(gray.cols());
    ArrayXXf output(rows, cols);
    const size_t total = static_cast<size_t>(rows) * cols;
    const size_t chunks = total / kBlock;
    const uint8_t* src = gray.data();
    float* dst = output.data();
    const bool useParallel = parallel && (total >= kOmpPixelThreshold);
#pragma omp parallel for if (useParallel) schedule(static)
    for (int64_t chunk = 0; chunk < static_cast<int64_t>(chunks); chunk++) {
        const size_t offset = static_cast<size_t>(chunk) * kBlock;
        storeFloats(_mm_loadu_si128(reinterpret_cast<const __m128i*>(src + offset)), dst + offset);
    }
    for (size_t i = chunks * kBlock; i < total; i++)
        dst[i] = static_cast<float>(src[i]);
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
    if (count == 0)
        return;
    omp_set_num_threads(count);
    Eigen::setNbThreads(count);
}
} // namespace eigen_utils
