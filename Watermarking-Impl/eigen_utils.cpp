#include "cimg_init.h"
#include "eigen_rgb_array.hpp"
#include "eigen_utils.hpp"
#include <Eigen/Dense>

using namespace cimg_library;
using namespace Eigen;

namespace eigen_utils 
{
	CImg<float> eigenRgbToCimg(const EigenArrayRGB& arrayRgb, const std::optional<CImg<float>>& alphaChannel)
	{
		const auto rows = arrayRgb[0].rows();
		const auto cols = arrayRgb[0].cols();
		const int channels = alphaChannel.has_value() ? 4 : 3;
		CImg<float> cimg_image(static_cast<unsigned int>(cols), static_cast<unsigned int>(rows), 1, channels);
		//a parallel pixel by pixel copy for loop is faster instead of three parallel (channel) bulk memory copies
		//because cimg and eigen use different memory layouts, and transposing is required which would make the copy much slower
	#pragma omp parallel for
		for (int y = 0; y < rows; ++y)
		{
			for (int x = 0; x < cols; ++x)
			{
				for (int channel = 0; channel < 3; channel++)
				{
					cimg_image(x, y, 0, channel) = arrayRgb[channel](y, x);
				}
				if (channels == 4)
					cimg_image(x, y, 0, 3) = (*alphaChannel)(x, y);
			}
		}
		return cimg_image;
	}

	void cimgAlphaZero(cimg_library::CImg<float>& rgbImage, const cimg_library::CImg<float>& alphaChannel)
	{
#pragma omp parallel for
		for (int y = 0; y < rgbImage.height(); ++y)
		{
			for (int x = 0; x < rgbImage.width(); ++x)
			{
				if (alphaChannel(x, y) == 0.0f) 
				{
					for (int channel = 0; channel < 3; channel++)
						rgbImage(x, y, 0, channel) = 0.0f; //set RGB channels to zero where alpha is zero
				}
			}
		}
	}

	EigenArrayRGB cimgToEigenRgb(const CImg<float>& rgbImage)
	{
		const int rows = rgbImage.height();
		const int cols = rgbImage.width();
		//a parallel pixel by pixel copy for loop is faster instead of three parallel (channel) bulk memory copies
		//because cimg and eigen use different memory layouts, and transposing is required which would make the copy much slower
		EigenArrayRGB rgb_array = { ArrayXXf(rows,cols), ArrayXXf(rows,cols), ArrayXXf(rows, cols) };
	#pragma omp parallel for
		for (int y = 0; y < rgbImage.height(); y++)
			for (int x = 0; x < rgbImage.width(); x++)
				for (int channel = 0; channel < 3; channel++)
					rgb_array[channel](y, x) = rgbImage(x, y, 0, channel);
		return rgb_array;
	}

	ArrayXXf eigenRgbToGray(const EigenArrayRGB& arrayRgb, const float rWeight, const float gWeight, const float bWeight)
	{
		return (arrayRgb[0] * rWeight) + (arrayRgb[1] * gWeight) + (arrayRgb[2] * bWeight);
	}
}
