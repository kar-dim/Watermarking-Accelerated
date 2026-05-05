#pragma once
#include "cimg_init.h"
#include <cstdint>
using Gray8BufferIO = cimg_library::CImg<uint8_t>;
using FloatBufferIO = cimg_library::CImg<float>;
#if defined(_USE_CUDA_)
#include "CudaArray.hpp"
#include <cuda_runtime.h>
using QueueHandle = cudaStream_t;
using ImageBuffer = CudaArray<float>;
using ImageOutputBuffer = CudaArray<uint8_t>;
using Gray8Buffer = CudaArray<uint8_t>;
using Gray16Buffer = CudaArray<uint16_t>;
using FlagBuffer = CudaArray<int32_t>;
#elif defined(_USE_OPENCL_)
#include "OclArray.hpp"
#include "opencl_init.h"
using QueueHandle = cl_command_queue;
using ImageBuffer = OclArray<float>;
using ImageOutputBuffer = OclArray<uint8_t>;
using Gray8Buffer = OclArray<uint8_t>;
using Gray16Buffer = OclArray<uint16_t>;
using FlagBuffer = OclArray<int32_t>;
#elif defined(_USE_EIGEN_)
#include "ImageEigenBuffer.hpp"
#include "ImageEigenOutputBuffer.hpp"
#include <Eigen/Core>
using ImageBuffer = ImageEigenBuffer;
using ImageOutputBuffer = ImageEigenOutputBuffer;
using Gray8Buffer = Eigen::Array<uint8_t, Eigen::Dynamic, Eigen::Dynamic>;
using Gray16Buffer = Eigen::Array<uint16_t, Eigen::Dynamic, Eigen::Dynamic>;
#endif
