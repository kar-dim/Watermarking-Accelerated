#pragma once
#if defined(_USE_CUDA_)
#include "GpuArray.hpp"
#include "cimg_init.h"
using ImageBuffer = GpuArray<float>;
using ImageOutputBuffer = GpuArray<uint8_t>;
using Gray8Buffer = GpuArray<uint8_t>;
using Gray16Buffer = GpuArray<uint16_t>;
using Gray8BufferIO = cimg_library::CImg<uint8_t>;
using FloatBufferIO = cimg_library::CImg<float>;
#elif defined(_USE_OPENCL_)
#include <arrayfire.h>
using ImageBuffer = af::array;
using ImageOutputBuffer = af::array;
using Gray8Buffer = af::array;
using Gray16Buffer = af::array;
using Gray8BufferIO = af::array;
using FloatBufferIO = af::array;
#elif defined(_USE_EIGEN_)
#include <cstdint>
#include "cimg_init.h"
#include "ImageEigenBuffer.hpp"
#include "ImageEigenOutputBuffer.hpp"
#include <Eigen/Core>
using ImageBuffer = ImageEigenBuffer;
using ImageOutputBuffer = ImageEigenOutputBuffer;
using Gray8Buffer = Eigen::Array<uint8_t, Eigen::Dynamic, Eigen::Dynamic>;
using Gray8BufferIO = cimg_library::CImg<uint8_t>;
using Gray16Buffer = Eigen::Array<uint16_t, Eigen::Dynamic, Eigen::Dynamic>;
using FloatBufferIO = cimg_library::CImg<float>;
#endif