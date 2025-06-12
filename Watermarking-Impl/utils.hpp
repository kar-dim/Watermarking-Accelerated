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
	template <typename Func>
	static double executionTime(Func&& func, const int loops = 1)
	{
		using clock = std::chrono::high_resolution_clock;
		using seconds = std::chrono::duration<double>;

		double totalSecs = 0.0;
		for (int i = 0; i < loops; i++)
		{
			const auto start = clock::now();
			std::forward<Func>(func)();
			const auto end = clock::now();
			totalSecs += seconds(end - start).count();
		}
		return totalSecs;
	}
#if defined(_USE_OPENCL_) || defined(_USE_CUDA_)
	static std::pair<unsigned int, unsigned int> getMaxImageSize();
#endif
};