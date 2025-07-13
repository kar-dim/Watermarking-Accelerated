#pragma once
#if defined(_USE_GPU_)
#include <arrayfire.h>
using BufferType = af::array;
using GrayBuffer = af::array;
#elif defined(_USE_EIGEN_)
#include <cstdint>
#include "EigenImage.hpp"
using BufferType = EigenImage;
using GrayBuffer = Eigen::Array<uint8_t, Eigen::Dynamic, Eigen::Dynamic>;
#endif