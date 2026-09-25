#include "buffer.hpp"
#include "eigen_rgb_array.hpp"
#include "eigen_utils.hpp"
#include "luma_coefficients.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <Eigen/Core>
#include <omp.h>
#include <optional>
#include <utility>
#include <vector>
#include <windows.h>

using namespace Eigen;

namespace eigen_utils {
Gray8BufferIO eigenRgbToCimg(const EigenArrayU8RGB& arrayRgb, const std::optional<Gray8BufferIO>& alphaChannel) {
    const auto rows = arrayRgb[0].rows();
    const auto cols = arrayRgb[0].cols();
    const int channels = alphaChannel.has_value() ? 4 : 3;
    Gray8BufferIO output(static_cast<unsigned int>(cols), static_cast<unsigned int>(rows), 1, channels);
#pragma omp parallel for schedule(static)
    for (int y = 0; y < rows; y++)
        for (int channel = 0; channel < 3; channel++)
            for (int x = 0; x < cols; x++)
                output(x, y, 0, channel) = arrayRgb[channel](y, x);
    if (channels == 4)
        std::memcpy(output.data() + (3 * cols * rows), alphaChannel->data(), cols * rows);
    return output;
}

Gray8BufferIO eigenGrayToCimg(const Gray8Buffer& arrayGray) {
    const auto rows = arrayGray.rows();
    const auto cols = arrayGray.cols();
    Gray8BufferIO output(static_cast<unsigned int>(cols), static_cast<unsigned int>(rows));
#pragma omp parallel for schedule(static)
    for (int y = 0; y < rows; y++)
        for (int x = 0; x < cols; x++)
            output(x, y) = arrayGray(y, x);
    return output;
}

// row-major planar float RGB (CImg) -> column-major 8-bit RGB (clamped and rounded) + float luma
std::pair<EigenArrayU8RGB, ArrayXXf> cimgToEigenRgbAndGray(const FloatBufferIO& rgbImage) {
    const int rows = rgbImage.height();
    const int cols = rgbImage.width();
    const size_t planeSize = static_cast<size_t>(rows) * cols;
    const float* red = rgbImage.data();
    const float* green = red + planeSize;
    const float* blue = green + planeSize;
    EigenArrayU8RGB output = makeEigenRGBu8(rows, cols);
    ArrayXXf gray(rows, cols);
    uint8_t* outputRed = output[0].data();
    uint8_t* outputGreen = output[1].data();
    uint8_t* outputBlue = output[2].data();
    const auto toByte = [](const float value) { return static_cast<uint8_t>(std::lround(std::clamp(value, 0.0f, 255.0f))); };
    float* outputGray = gray.data();
#pragma omp parallel for schedule(static)
    for (int col = 0; col < cols; ++col) {
        const size_t outputColumn = static_cast<size_t>(col) * rows;
        for (int row = 0; row < rows; ++row) {
            const size_t inputIndex = static_cast<size_t>(row) * cols + col;
            const size_t outputIndex = outputColumn + row;
            const uint8_t r = toByte(red[inputIndex]);
            const uint8_t g = toByte(green[inputIndex]);
            const uint8_t b = toByte(blue[inputIndex]);
            outputRed[outputIndex] = r;
            outputGreen[outputIndex] = g;
            outputBlue[outputIndex] = b;
            outputGray[outputIndex] = (static_cast<float>(r) * CommonUtils::kLumaR + static_cast<float>(g) * CommonUtils::kLumaG) + static_cast<float>(b) * CommonUtils::kLumaB;
        }
    }
    return {std::move(output), std::move(gray)};
}

ImageBuffer cimgToEigenGray(const FloatBufferIO& grayImage) {
    const int rows = grayImage.height();
    const int cols = grayImage.width();
    ArrayXXf output(rows, cols);
#pragma omp parallel for schedule(static)
    for (int x = 0; x < cols; x++)
        for (int y = 0; y < rows; y++)
            output(y, x) = grayImage(x, y);
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
