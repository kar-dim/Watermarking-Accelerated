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
 *  \brief  Utility functions for Eigen operations
 *  \author Dimitris Karatzas
 */
namespace eigen_utils {
Gray8BufferIO eigenRgbToCimg(const EigenArrayU8RGB& arrayRgb, const std::vector<uint8_t>& alphaChannel, bool parallel = true);
Gray8BufferIO eigenGrayToCimg(const Gray8Buffer& arrayGray, bool parallel = true);
ImageBuffer cimgToEigenGray(const Gray8BufferIO& grayImage, bool parallel = true, int orientation = 1);
std::pair<EigenArrayU8RGB, Eigen::ArrayXXf> cimgToEigenRgbAndGray(const Gray8BufferIO& rgbImage, bool parallel = true, int orientation = 1);
ImageBuffer u8RgbToFloatGray(const EigenArrayU8RGB& rgb, bool parallel = true);
ImageBuffer u8GrayToFloat(const Gray8Buffer& gray, bool parallel = true);
void pitchedRowMajorU8ToColMajorFloat(const uint8_t* source, int rows, int cols, int srcPitch, float* destination, bool parallel = true);
void setThreadsToPhysicalCores();
inline EigenArrayU8RGB makeEigenRGBu8(int rows, int cols) { return {Gray8Buffer(rows, cols), Gray8Buffer(rows, cols), Gray8Buffer(rows, cols)}; }
} // namespace eigen_utils
