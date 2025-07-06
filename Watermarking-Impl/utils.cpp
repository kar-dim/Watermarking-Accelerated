#include "buffer.hpp"
#include "utils.hpp"
#include "WatermarkBase.hpp"
#include <format>
#include <memory>
#include <stdexcept>
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
#include <algorithm>
#include <cctype>
#include "cimg_init.h"
#include "eigen_utils.hpp"
#include "WatermarkEigen.hpp"
#endif

using std::string;

string Utils::addSuffixBeforeExtension(const string& file, const string& suffix)
{
	const auto dot = file.find_last_of('.');
	if (dot == string::npos || dot == file.size() - 1)
		throw std::runtime_error("Filename has no valid extension: " + file);
	return file.substr(0, dot) + suffix + file.substr(dot);
}

void Utils::saveImage(const string& imagePath, const string& suffix, const BufferType& watermark)
{
#if defined(_USE_EIGEN_)
	const string watermarkedFile = Utils::addSuffixBeforeExtension(imagePath, suffix);
	string extension = watermarkedFile.substr(watermarkedFile.find_last_of('.') + 1);
	std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
	const auto rgbCimg = eigen_utils::eigenRgbToCimg(watermark.getRGB());
	if (extension == "png")  rgbCimg.save_png(watermarkedFile.c_str());
	else if (extension == "bmp")  rgbCimg.save_bmp(watermarkedFile.c_str());
	else if (extension == "jpg" || extension == "jpeg") rgbCimg.save_jpeg(watermarkedFile.c_str());
	else
		throw std::runtime_error("Unsupported image format: " + extension);
#elif defined(_USE_GPU_)
	af::saveImageNative(addSuffixBeforeExtension(imagePath, suffix).c_str(), watermark.as(u8));
#endif
}

std::unique_ptr<WatermarkBase> Utils::createWatermarkObject(const unsigned int height, const unsigned int width, const string& randomMatrixPath, const int p, const float psnr)
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

void Utils::checkError(const bool isError, const string& errorMsg) 
{ 
	if (isError) 
		throw std::runtime_error(errorMsg); 
};

//helper method to calculate execution time in FPS or in seconds
string Utils::formatExecutionTime(const bool showFps, const double seconds)
{
	return showFps ? std::format("FPS: {:.2f} FPS", 1.0 / seconds) : std::format("{:.6f} seconds", seconds);
}