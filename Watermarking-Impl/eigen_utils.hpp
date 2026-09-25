#pragma once
#include "buffer.hpp"
#include "eigen_rgb_array.hpp"
#include <optional>
#include <utility>

/*!
 *  \brief  Helper utility functions related to Eigen.
 *  \author Dimitris Karatzas
 */
namespace eigen_utils {
Gray8BufferIO eigenRgbToCimg(const EigenArrayU8RGB& arrayRgb, const std::optional<Gray8BufferIO>& alphaChannel);
Gray8BufferIO eigenGrayToCimg(const Gray8Buffer& arrayGray);
ImageBuffer cimgToEigenGray(const FloatBufferIO& grayImage);
std::pair<EigenArrayU8RGB, Eigen::ArrayXXf> cimgToEigenRgbAndGray(const FloatBufferIO& rgbImage);
void setThreadsToPhysicalCores();
inline EigenArrayU8RGB makeEigenRGBu8(int rows, int cols) { return {Gray8Buffer(rows, cols), Gray8Buffer(rows, cols), Gray8Buffer(rows, cols)}; }
} // namespace eigen_utils
