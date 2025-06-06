#pragma once
#if defined(_USE_OPENCL_) || defined(_USE_CUDA_)
#include <arrayfire.h>
#include <utility>
#endif
#include <chrono>
#include <string>
#include "buffer.hpp"
#include "WatermarkBase.hpp"
#include <memory>
/*!
 *  \brief  Helper utility methods for testing the watermark algorithms
 *  \author Dimitris Karatzas
 */
class Utilities 
{
public:
	static std::string addSuffixBeforeExtension(const std::string& file, const std::string& suffix);
	static void saveImage(const std::string& imagePath, const std::string& suffix, const BufferType& watermark);
	static std::unique_ptr<WatermarkBase> createWatermarkObject(const unsigned int height, const unsigned int width, const std::string& randomMatrixPath, const int p, const float psnr);
#if defined(_USE_OPENCL_) || defined(_USE_CUDA_)
	static std::pair<unsigned int, unsigned int> getMaxImageSize();
#endif
};

/*!
 *  \brief  simple methods to calculate execution times
 *  \author Dimitris Karatzas
 */
namespace timer 
{
	static std::chrono::time_point<std::chrono::steady_clock> startTime, currentTime;
	void start();
	void end();
	float elapsedSeconds();
}