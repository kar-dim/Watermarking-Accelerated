#include "utils.hpp"
#include <chrono>
#include <string>
#if defined(_USE_OPENCL_)
#include "opencl_utils.hpp"
#include <af/opencl.h>
#include <utility>
#elif defined(_USE_CUDA_)
#include "cuda_utils.hpp"
#include <utility>
#elif defined(_USE_EIGEN_)
#include "eigen_utils.hpp"
#endif
#include "buffer.hpp"
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

// Returns the maximum image size supported by the device (cols, rows)

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


namespace timer 
{
	void start() 
	{
		startTime = std::chrono::high_resolution_clock::now();
	}
	void end() 
	{
		currentTime = std::chrono::high_resolution_clock::now();
	}
	float elapsedSeconds() 
	{
		return static_cast<float>(std::chrono::duration_cast<std::chrono::microseconds>(currentTime - startTime).count() / 1000000.0f);
	}
}