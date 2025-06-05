#pragma once
#include "eigen_rgb_array.hpp"
#include <CImg.h>
#include <string>

enum IMAGE_TYPE
{
	JPG,
	PNG
};

cimg_library::CImg<float> eigen3dArrayToCimg(const EigenArrayRGB& imageRgb);
EigenArrayRGB cimgToEigen3dArray(const cimg_library::CImg<float>& rgbImage);
Eigen::ArrayXXf eigen3dArrayToGrayscaleArray(const EigenArrayRGB& imageRgb, const float rWeight, const float gWeight, const float bWeight);
void saveWatermarkedImage(const std::string& imagePath, const std::string& suffix, const EigenArrayRGB& watermark, const IMAGE_TYPE type);
