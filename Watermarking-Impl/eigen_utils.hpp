#pragma once
#include "cimg_init.h"
#include "eigen_rgb_array.hpp"
#include <string>

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
	cimg_library::CImg<float> eigenRgbToCimg(const EigenArrayRGB& imageRgb);
	EigenArrayRGB cimgToEigenRgb(const cimg_library::CImg<float>& rgbImage);
	Eigen::ArrayXXf eigenRgbToGray(const EigenArrayRGB& imageRgb, const float rWeight, const float gWeight, const float bWeight);
	void saveWatermarkedImage(const std::string& imagePath, const std::string& suffix, const EigenArrayRGB& watermark, const IMAGE_TYPE type);
}
