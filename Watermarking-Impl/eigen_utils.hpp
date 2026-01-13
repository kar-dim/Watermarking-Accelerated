#pragma once
#include "buffer.hpp"
#include "cimg_init.h"
#include "eigen_rgb_array.hpp"
#include <cstdint>
#include <optional>

/*!
 *  \brief  Helper utility functions related to Eigen.
 *  \author Dimitris Karatzas
 */
namespace eigen_utils {
cimg_library::CImg<uint8_t> eigenRgbToCimg(const EigenArrayU8RGB& arrayRgb, const std::optional<AlphaBuffer>& alphaChannel);
cimg_library::CImg<uint8_t> eigenGrayToCimg(const Gray8Buffer& arrayGray);
ImageBuffer cimgToEigenGray(const cimg_library::CImg<float>& grayImage);
void cimgAlphaZero(cimg_library::CImg<float>& rgbImage, const AlphaBuffer& alphaChannel);
EigenArrayRGB cimgToEigenRgb(const cimg_library::CImg<float>& rgbImage);
void setThreadsToPhysicalCores();
inline EigenArrayU8RGB makeEigenRGBu8(int rows, int cols) { return {Gray8Buffer(rows, cols), Gray8Buffer(rows, cols), Gray8Buffer(rows, cols)}; }
} // namespace eigen_utils
