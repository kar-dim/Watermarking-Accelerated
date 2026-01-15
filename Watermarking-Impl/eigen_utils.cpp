#include "buffer.hpp"
#include "eigen_rgb_array.hpp"
#include "eigen_utils.hpp"
#include <cstdint>
#include <Eigen/Core>
#include <omp.h>
#include <optional>
#include <vector>
#include <windows.h>

using namespace Eigen;

namespace eigen_utils {
Gray8BufferIO eigenRgbToCimg(const EigenArrayU8RGB& arrayRgb, const std::optional<Gray8BufferIO>& alphaChannel) {
    const auto rows = arrayRgb[0].rows();
    const auto cols = arrayRgb[0].cols();
    const int channels = alphaChannel.has_value() ? 4 : 3;
    Gray8BufferIO output(static_cast<unsigned int>(cols), static_cast<unsigned int>(rows), 1, channels);
#pragma omp parallel for
    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            for (int channel = 0; channel < 3; channel++)
                output(x, y, 0, channel) = arrayRgb[channel](y, x);
            if (channels == 4)
                output(x, y, 0, 3) = (*alphaChannel)(x, y);
        }
    }
    return output;
}

Gray8BufferIO eigenGrayToCimg(const Gray8Buffer& arrayGray) {
    const auto rows = arrayGray.rows();
    const auto cols = arrayGray.cols();
    Gray8BufferIO output(static_cast<unsigned int>(cols), static_cast<unsigned int>(rows));
#pragma omp parallel for
    for (int y = 0; y < rows; y++)
        for (int x = 0; x < cols; x++)
            for (int channel = 0; channel < 3; channel++)
                output(x, y) = arrayGray(y, x);
    return output;
}

void cimgAlphaZero(FloatBufferIO& rgbImage, const Gray8BufferIO& alphaChannel) {
#pragma omp parallel for
    for (int y = 0; y < rgbImage.height(); y++) {
        for (int x = 0; x < rgbImage.width(); x++) {
            if (alphaChannel(x, y) == 0.0f) {
                for (int channel = 0; channel < 3; channel++)
                    rgbImage(x, y, 0, channel) = 0.0f; // set RGB channels to zero where alpha is zero
            }
        }
    }
}

EigenArrayRGB cimgToEigenRgb(const FloatBufferIO& rgbImage) {
    const int rows = rgbImage.height();
    const int cols = rgbImage.width();
    EigenArrayRGB output = {ArrayXXf(rows, cols), ArrayXXf(rows, cols), ArrayXXf(rows, cols)};
#pragma omp parallel for
    for (int x = 0; x < cols; x++)
        for (int y = 0; y < rows; y++)
            for (int channel = 0; channel < 3; channel++)
                output[channel](y, x) = rgbImage(x, y, 0, channel);
    return output;
}

ImageBuffer cimgToEigenGray(const FloatBufferIO& grayImage) {
    const int rows = grayImage.height();
    const int cols = grayImage.width();
    ArrayXXf output(rows, cols);
#pragma omp parallel for
    for (int x = 0; x < cols; x++)
        for (int y = 0; y < rows; y++)
            output(y, x) = grayImage(x, y);
    return ImageBuffer(output);
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
