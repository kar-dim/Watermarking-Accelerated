#pragma once
#if defined(_USE_GPU_)
#include <arrayfire.h>
using ImageBuffer = af::array;
using Gray8Buffer = af::array;
using Gray16Buffer = af::array;
using AlphaBuffer = af::array;
#elif defined(_USE_EIGEN_)
#include <cstdint>
#include "cimg_init.h"
#include "ImageEigenBuffer.hpp"
using ImageBuffer = ImageEigenBuffer;
using Gray8Buffer = Eigen::Array<uint8_t, Eigen::Dynamic, Eigen::Dynamic>;
using Gray16Buffer = Eigen::Array<uint16_t, Eigen::Dynamic, Eigen::Dynamic>;
using AlphaBuffer = cimg_library::CImg<float>;
#endif