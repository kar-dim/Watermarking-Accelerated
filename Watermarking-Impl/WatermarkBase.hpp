#pragma once

#include "buffer.hpp"
#include <cmath>
#include <fstream>
#if defined(_USE_GPU_)
#include <memory>
#endif
#include <stdexcept>
#include <string>

enum MASK_TYPE
{
	ME,
	NVF
};

/*!
 *  \brief  Functions for watermark computation and detection, Base class.
 *  \author Dimitris Karatzas
 */
class WatermarkBase 
{
private:
	static inline float computeStrengthFactor(const float psnr) { return 255.0f / std::sqrt(std::pow(10.0f, psnr / 10.0f)); }

public:
	WatermarkBase(const unsigned int rows, const unsigned int cols, const std::string& randomMatrixPath, const float psnr)
		: baseRows(rows), baseCols(cols), randomMatrix(loadRandomMatrix(randomMatrixPath)), strengthFactor(computeStrengthFactor(psnr))
	{ }

	WatermarkBase(const unsigned int rows, const unsigned int cols, const BufferType& randomMatrix, const float strengthFactor)
		: baseRows(rows), baseCols(cols), randomMatrix(randomMatrix), strengthFactor(strengthFactor)
	{ }

    virtual ~WatermarkBase() = default;

	//main watermark embedding method
	//it embeds the watermark computed fom "inputImage" (always grayscale)
	//into a new array based on "outputImage" (RGB or grayscale)
	virtual BufferType makeWatermark(const BufferType& inputImage, const BufferType& outputImage, float& watermarkStrength, const MASK_TYPE maskType) = 0;
	
	//the main mask detector function
	virtual float detectWatermark(const BufferType& inputImage, const MASK_TYPE maskType) = 0;

protected:
	template<unsigned int ALIGN>
	static constexpr unsigned int align(const unsigned int x) { return (x + (ALIGN - 1)) & ~(ALIGN - 1); }
	static constexpr bool maskCalcRequired = true;
	static constexpr bool maskCalcNotRequired = false;
	unsigned int baseRows, baseCols;
	BufferType randomMatrix;
	float strengthFactor;

	//helper method to load the random noise matrix W from the file specified.
	BufferType loadRandomMatrix(const std::string& randomMatrixPath) const
	{
		std::ifstream randomMatrixStream(randomMatrixPath.c_str(), std::ios::binary);
		if (!randomMatrixStream.is_open())
			throw std::runtime_error(std::string("Error opening '" + randomMatrixPath + "' file for Random noise W array\n"));
		randomMatrixStream.seekg(0, std::ios::end);
		const auto totalBytes = randomMatrixStream.tellg();
		randomMatrixStream.seekg(0, std::ios::beg);
		if (baseRows * baseCols * sizeof(float) != totalBytes)
			throw std::runtime_error(std::string("Error: W file total elements != image dimensions! W file total elements: " + std::to_string(totalBytes / (sizeof(float))) + ", Image width: " + std::to_string(baseCols) + ", Image height: " + std::to_string(baseRows) + "\n"));
#if defined(_USE_GPU_)
		std::unique_ptr<float> wPtr(new float[baseRows * baseCols]);
		randomMatrixStream.read(reinterpret_cast<char*>(wPtr.get()), totalBytes);
		return af::transpose(af::array(baseCols, baseRows, wPtr.get()));
#elif defined(_USE_EIGEN_)
		Eigen::ArrayXXf watermark(baseCols, baseRows);
		randomMatrixStream.read(reinterpret_cast<char*>(watermark.data()), totalBytes);
		return BufferType(watermark.transpose());
#endif
	}
};