#pragma once

#include "buffer.hpp"
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>

enum MASK_TYPE
{
	ME,
	NVF
};

#define ALIGN(x, ALIGN) (((x) + ((ALIGN) - 1)) & ~((ALIGN) - 1))
#define ME_MASK_CALCULATION_REQUIRED_NO false
#define ME_MASK_CALCULATION_REQUIRED_YES true

class WatermarkBase {
public:
	WatermarkBase(const unsigned int rows, const unsigned int cols, const std::string& randomMatrixPath, const int p, const float strengthFactor)
		: baseRows(rows), baseCols(cols), p(p), randomMatrix(loadRandomMatrix(randomMatrixPath)), strengthFactor(strengthFactor)
	{ }
	WatermarkBase(const unsigned int rows, const unsigned int cols, const BufferType& randomMatrix, const int p, const float strengthFactor)
		: baseRows(rows), baseCols(cols), p(p), randomMatrix(randomMatrix), strengthFactor(strengthFactor) 
	{ }
    virtual ~WatermarkBase() = default;
	virtual BufferType makeWatermark(const BufferType& inputImage, const BufferType& outputImage, float& watermarkStrength, const MASK_TYPE maskType) = 0;
	virtual float detectWatermark(const BufferType& watermarkedImage, const MASK_TYPE maskType) = 0;
protected:
	unsigned int baseRows, baseCols;
	int p;
	BufferType randomMatrix;
	float strengthFactor;

	//helper method to load the random noise matrix W from the file specified.
	//This is the random generated watermark generated from a Normal distribution generator with mean 0 and standard deviation 1
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
		std::unique_ptr<float> wPtr(new float[baseRows * baseCols]);
		randomMatrixStream.read(reinterpret_cast<char*>(wPtr.get()), totalBytes);
#if defined(_USE_EIGEN_)
		return BufferType(std::move(Eigen::Map<Eigen::ArrayXXf>(wPtr.get(), baseCols, baseRows).transpose().eval()));
#elif defined(_USE_CUDA_) || defined(_USE_OPENCL_)
		return af::transpose(af::array(baseCols, baseRows, wPtr.get()));
#endif
	}
};