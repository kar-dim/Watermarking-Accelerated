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
	cimg_library::CImg<float> eigenRgbToCimg(const EigenArrayRGB& imageRgb, const std::optional<AlphaBuffer>& alphaChannel);
	void cimgAlphaZero(cimg_library::CImg<float>& rgbImage, const cimg_library::CImg<float>& alphaChannel);
	EigenArrayRGB cimgToEigenRgb(const cimg_library::CImg<float>& rgbImage);
	void setThreadsToPhysicalCores();
	inline EigenArrayRGB makeEigenRGB(int rows, int cols)
	{
		return { Eigen::ArrayXXf(rows, cols), Eigen::ArrayXXf(rows, cols), Eigen::ArrayXXf(rows, cols) };
	}
}
