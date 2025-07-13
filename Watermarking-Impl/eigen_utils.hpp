#pragma once
#include "buffer.hpp"
#include "cimg_init.h"
#include "eigen_rgb_array.hpp"
#include <optional>

enum IMAGE_TYPE
{
	JPG,
	PNG
};

/*!
 *  \brief  Helper utility functions related to Eigen.
 *  \author Dimitris Karatzas
 */
namespace eigen_utils
{
	cimg_library::CImg<float> eigenRgbToCimg(const EigenArrayRGB& imageRgb, const std::optional<BufferAlphaType>& alphaChannel);
	void cimgAlphaZero(cimg_library::CImg<float>& rgbImage, const cimg_library::CImg<float>& alphaChannel);
	EigenArrayRGB cimgToEigenRgb(const cimg_library::CImg<float>& rgbImage);
	Eigen::ArrayXXf eigenRgbToGray(const EigenArrayRGB& imageRgb, const float rWeight, const float gWeight, const float bWeight);
}
