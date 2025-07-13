#pragma once
#include "buffer.hpp"

#if defined(_USE_GPU_)
#include <arrayfire.h>
#include <utility>
#endif

#include <chrono>
#include <memory>
#include <string>
#include "WatermarkBase.hpp"
#include <optional>

/*!
 *  \brief  Helper utility methods.
 *  \author Dimitris Karatzas
 */
class Utils
{
public:
	static constexpr float rPercent = 0.299f;
	static constexpr float gPercent = 0.587f;
	static constexpr float bPercent = 0.114f;

	static std::string addSuffixBeforeExtension(const std::string& file, const std::string& suffix);
	static void saveImage(const std::string& imagePath, const std::string& suffix, const BufferType& watermark, const std::optional<BufferAlphaType>& alphaChannel);
	static std::unique_ptr<WatermarkBase> createWatermarkObject(const unsigned int height, const unsigned int width, const std::string& randomMatrixPath, const int p, const float psnr);
	//throws exception if an error condition is true
	static void checkError(const bool isError, const std::string& errorMsg);
	//helper method to calculate execution time in FPS or in seconds
	static std::string formatExecutionTime(const bool showFps, const double seconds);
	static void loadImage(BufferType& rgbImage, BufferType& image, const std::string& imageFile, std::optional<BufferAlphaType>& alphaChannel);

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
};