#pragma once

#include "buffer.hpp"
#include "eigen_rgb_array.hpp"
#include "WatermarkBase.hpp"
#include <Eigen/Dense>
#include <string>

class WatermarkEigen : public WatermarkBase {
private:
	int pad;
	unsigned int paddedRows, paddedCols;
	int pSquared, halfNeighborsSize;
	Eigen::ArrayXXf padded;

	void createNeighbors(const Eigen::ArrayXXf& array, Eigen::VectorXf& x_, const int neighborSize, const int i, const int j) const;
	Eigen::ArrayXXf computeCustomMask(const Eigen::ArrayXXf& image, const Eigen::ArrayXXf& padded) const;
	Eigen::ArrayXXf computePredictionErrorMask(const Eigen::ArrayXXf& paddedImage, Eigen::ArrayXXf& errorSequence, Eigen::VectorXf& coefficients, const bool maskNeeded) const;
	Eigen::ArrayXXf computeErrorSequence(const Eigen::ArrayXXf& padded, const Eigen::VectorXf& coefficients) const;
	Eigen::ArrayXXf computeStrengthenedWatermark(const Eigen::ArrayXXf& inputImage, float& watermarkStrength, MASK_TYPE maskType);

public:
	WatermarkEigen(const unsigned int rows, const unsigned int cols, const std::string& randomMatrixPath, const int p, const float psnr);
	BufferType makeWatermark(const BufferType& inputImage, const BufferType& outputImage, float& watermarkStrength, MASK_TYPE maskType) override;
	float detectWatermark(const BufferType& watermarkedImage, MASK_TYPE type) override;
};