#pragma once
#include "buffer.hpp"
#include "cimg_init.h"
#include "eigen_rgb_array.hpp"
#include <optional>

/*!
 *  \brief  Helper utility functions related to Eigen.
 *  \author Dimitris Karatzas
 */
namespace eigen_utils
{
	cimg_library::CImg<float> eigenRgbToCimg(const EigenArrayRGB& imageRgb, const std::optional<AlphaBuffer>& alphaChannel);
	cimg_library::CImg<float> eigenGrayToCimg(const Eigen::ArrayXXf& arrayGray);
	ImageBuffer cimgToEigenGray(const cimg_library::CImg<float>& grayImage);
	void cimgAlphaZero(cimg_library::CImg<float>& rgbImage, const AlphaBuffer& alphaChannel);
	EigenArrayRGB cimgToEigenRgb(const cimg_library::CImg<float>& rgbImage);
	void setThreadsToPhysicalCores();
	inline EigenArrayRGB makeEigenRGB(int rows, int cols)
	{
		return { Eigen::ArrayXXf(rows, cols), Eigen::ArrayXXf(rows, cols), Eigen::ArrayXXf(rows, cols) };
	}
}
