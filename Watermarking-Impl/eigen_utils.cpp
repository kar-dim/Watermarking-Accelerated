#include "buffer.hpp"
#include "cimg_init.h"
#include "eigen_rgb_array.hpp"
#include "eigen_utils.hpp"
#include <cstdint>
#include <Eigen/Dense>
#include <omp.h>
#include <optional>
#include <vector>
#include <windows.h>

using namespace cimg_library;
using namespace Eigen;

namespace eigen_utils 
{
	CImg<float> eigenRgbToCimg(const EigenArrayRGB& arrayRgb, const std::optional<AlphaBuffer>& alphaChannel)
	{
		const auto rows = arrayRgb[0].rows();
		const auto cols = arrayRgb[0].cols();
		const int channels = alphaChannel.has_value() ? 4 : 3;
		CImg<float> output(static_cast<unsigned int>(cols), static_cast<unsigned int>(rows), 1, channels);
	#pragma omp parallel for
		for (int y = 0; y < rows; ++y)
		{
			for (int x = 0; x < cols; ++x)
			{
				for (int channel = 0; channel < 3; channel++)
				{
					output(x, y, 0, channel) = arrayRgb[channel](y, x);
				}
				if (channels == 4)
					output(x, y, 0, 3) = (*alphaChannel)(x, y);
			}
		}
		return output;
	}

	void cimgAlphaZero(CImg<float>& rgbImage, const AlphaBuffer& alphaChannel)
	{
#pragma omp parallel for
		for (int y = 0; y < rgbImage.height(); y++)
		{
			for (int x = 0; x < rgbImage.width(); x++)
			{
				if (alphaChannel(x, y) == 0.0f) 
				{
					for (int channel = 0; channel < 3; channel++)
						rgbImage(x, y, 0, channel) = 0.0f; //set RGB channels to zero where alpha is zero
				}
			}
		}
	}

	EigenArrayRGB cimgToEigenRgb(const CImg<float>& rgbImage)
	{
		const int rows = rgbImage.height();
		const int cols = rgbImage.width();
		EigenArrayRGB output = { ArrayXXf(rows,cols), ArrayXXf(rows,cols), ArrayXXf(rows, cols) };
	#pragma omp parallel for
		for (int x = 0; x < rgbImage.width(); x++)
			for (int y = 0; y < rgbImage.height(); y++)
				for (int channel = 0; channel < 3; channel++)
					output[channel](y, x) = rgbImage(x, y, 0, channel);
		return output;
	}

	//sets the number of OpenMP (watermarking) threads based on physical cores
	//it is used only for video embedding, to improve performance by reducing
	//context switching between openmp and ffmpeg's threads
	void setThreadsToPhysicalCores()
	{
		DWORD len = 0;
		GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
		std::vector<uint8_t> buffer(len);
		auto info = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buffer.data());
		if (!GetLogicalProcessorInformationEx(RelationProcessorCore, info, &len)) 
			return;
		unsigned count = 0;
		char* ptr = reinterpret_cast<char*>(info);
		char* end = ptr + len;
		while (ptr < end) 
		{
			auto p = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(ptr);
			if (p->Relationship == RelationProcessorCore) count++;
			ptr += p->Size;
		}
		omp_set_num_threads(count);
		Eigen::setNbThreads(count);
	}
}
