#pragma once
#include "buffer.hpp"
#include "eigen_rgb_array.hpp"
#include <cstdint>
#include <utility>
#include <vector>

// CImg is only declared to avoid including the whole CImg header here
namespace cimg_library {
template <typename T>
struct CImg;
} // namespace cimg_library
using Gray8BufferIO = cimg_library::CImg<uint8_t>;

/*!
 *  \brief  Helper utility functions related to Eigen.
 *  \author Dimitris Karatzas
 */
namespace eigen_utils {
Gray8BufferIO eigenRgbToCimg(const EigenArrayU8RGB& arrayRgb, const std::vector<uint8_t>& alphaChannel);
Gray8BufferIO eigenGrayToCimg(const Gray8Buffer& arrayGray);
ImageBuffer cimgToEigenGray(const Gray8BufferIO& grayImage);
std::pair<EigenArrayU8RGB, Eigen::ArrayXXf> cimgToEigenRgbAndGray(const Gray8BufferIO& rgbImage);
void setThreadsToPhysicalCores();
inline EigenArrayU8RGB makeEigenRGBu8(int rows, int cols) { return {Gray8Buffer(rows, cols), Gray8Buffer(rows, cols), Gray8Buffer(rows, cols)}; }
} // namespace eigen_utils
