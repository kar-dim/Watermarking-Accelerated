#include "buffer.hpp"
#include "utils.hpp"
#include "WatermarkBase.hpp"
#include <memory>
#include <string>
#if defined(_USE_OPENCL_)
#include "WatermarkOCL.hpp"
#include <af/opencl.h>
#include <utility>
#elif defined(_USE_CUDA_)
#include "cuda_utils.hpp"
#include "WatermarkCuda.cuh"
#include <utility>
#elif defined(_USE_EIGEN_)
#include "cimg_init.h"
#include "eigen_utils.hpp"
#include "WatermarkEigen.hpp"
#endif
#include <stdexcept>

using std::string;

string Utilities::addSuffixBeforeExtension(const string& file, const string& suffix)
{
	auto dot = file.find_last_of('.');
	return dot == string::npos ? file + suffix : file.substr(0, dot) + suffix + file.substr(dot);
}

void Utilities::saveImage(const string& imagePath, const string& suffix, const BufferType& watermark)
{
#if defined(_USE_EIGEN_)
	const string watermarkedFile = Utilities::addSuffixBeforeExtension(imagePath, suffix);
	eigen3dArrayToCimg(watermark.getRGB()).save_png(watermarkedFile.c_str());
#elif defined(_USE_OPENCL_) || defined(_USE_CUDA_)
	af::saveImageNative(addSuffixBeforeExtension(imagePath, suffix).c_str(), watermark.as(u8));
#endif
}

std::unique_ptr<WatermarkBase> Utilities::createWatermarkObject(const unsigned int height, const unsigned int width, const string& randomMatrixPath, const int p, const float psnr)
{
	std::unique_ptr<WatermarkBase> watermarkObj;
#if defined(_USE_OPENCL_)
	watermarkObj = std::make_unique<WatermarkOCL>(height, width, randomMatrixPath, p, psnr);
#elif defined(_USE_CUDA_)
	watermarkObj = std::make_unique<WatermarkCuda>(height, width, randomMatrixPath, p, psnr);
#elif defined(_USE_EIGEN_)
	switch (p)
	{
	case 3:
		watermarkObj = std::make_unique<WatermarkEigen<3>>(height, width, randomMatrixPath, psnr); break;
	case 5:
		watermarkObj = std::make_unique<WatermarkEigen<5>>(height, width, randomMatrixPath, psnr); break;
	case 7:
		watermarkObj = std::make_unique<WatermarkEigen<7>>(height, width, randomMatrixPath, psnr); break;
	case 9:
		watermarkObj = std::make_unique<WatermarkEigen<9>>(height, width, randomMatrixPath, psnr); break;
	default:
		throw std::invalid_argument("Unsupported value for p. Allowed values: 3, 5, 7, 9.");
	}
#endif
	return watermarkObj;
}

//returns the maximum image size supported by the device (cols, rows)
#if defined(_USE_OPENCL_) || defined(_USE_CUDA_)
std::pair<unsigned int, unsigned int> Utilities::getMaxImageSize()
{
#if defined(_USE_OPENCL_)
	const cl::Device device(afcl::getDeviceId(), false);
	return std::pair<unsigned int, unsigned int>(
		static_cast<unsigned int>(device.getInfo<CL_DEVICE_IMAGE2D_MAX_WIDTH>()), 
		static_cast<unsigned int>(device.getInfo<CL_DEVICE_IMAGE2D_MAX_HEIGHT>())
	);
#elif defined(_USE_CUDA_)
	const auto properties = cuda_utils::getDeviceProperties();
	return std::pair<unsigned int, unsigned int>(properties.maxTexture2D[0], properties.maxTexture2D[1]);
#endif
}
#endif