#pragma once
#if defined(_USE_OPENCL_) || defined(_USE_CUDA_)
#include <arrayfire.h>
#endif
#include <chrono>
#include <string>
#include <utility>
#if defined(_USE_OPENCL_)
#include "opencl_init.h"
#endif
#include "buffer.hpp"
/*!
 *  \brief  Helper utility methods for testing the watermark algorithms
 *  \author Dimitris Karatzas
 */
class Utilities 
{
public:
	static std::string addSuffixBeforeExtension(const std::string& file, const std::string& suffix);
	static void saveImage(const std::string& imagePath, const std::string& suffix, const BufferType& watermark);
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