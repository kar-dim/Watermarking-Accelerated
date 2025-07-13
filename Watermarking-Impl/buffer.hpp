#pragma once
#if defined(_USE_GPU_)
#include <arrayfire.h>
using BufferType = af::array;
using GrayBuffer = af::array;
using BufferAlphaType = af::array;
#elif defined(_USE_EIGEN_)
#include <cstdint>
#include "cimg_init.h"
#include "EigenImage.hpp"
using BufferType = EigenImage;
using GrayBuffer = Eigen::Array<uint8_t, Eigen::Dynamic, Eigen::Dynamic>;
using BufferAlphaType = cimg_library::CImg<float>;
#endif