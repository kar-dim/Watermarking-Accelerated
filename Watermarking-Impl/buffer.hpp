#pragma once
#if defined(_USE_CUDA_) || defined(_USE_OPENCL_)
#include <arrayfire.h>
using BufferType = af::array;
using GrayBuffer = af::array;
#elif defined(_USE_EIGEN_)
#include "EigenImage.hpp"
using BufferType = EigenImage;
using GrayBuffer = Eigen::Array<uint8_t, Eigen::Dynamic, Eigen::Dynamic>;
#endif