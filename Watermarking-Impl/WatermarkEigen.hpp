#pragma once

#include "buffer.hpp"
#include "eigen_rgb_array.hpp"
#include "WatermarkBase.hpp"
#include <Eigen/Dense>
#include <string>

class WatermarkEigen : public WatermarkBase 
{
private:
	int pad;
	unsigned int paddedRows, paddedCols;
	int pSquared, halfNeighborsSize;
	Eigen::ArrayXXf padded, mask, errorSequence, filteredEstimation, u, uStrengthened;
	Eigen::VectorXf coefficients;
	EigenArrayRGB watermarkedImage;
	void createNeighbors(const Eigen::ArrayXXf& array, Eigen::VectorXf& x_, const int neighborSize, const int i, const int j) const;
	void onReinitialize() override;
	void computeCustomMask(const Eigen::ArrayXXf& image);
	void computePredictionErrorMask(const bool maskNeeded);
	void computeErrorSequence(Eigen::ArrayXXf& outputErrorSequence);
	void computeStrengthenedWatermark(const Eigen::ArrayXXf& inputImage, float& watermarkStrength, MASK_TYPE maskType);

public:
	WatermarkEigen(const unsigned int rows, const unsigned int cols, const std::string& randomMatrixPath, const int p, const float psnr);
	WatermarkEigen(const WatermarkEigen& other) = default;
	WatermarkEigen(WatermarkEigen&& other) noexcept = default;
	WatermarkEigen& operator=(WatermarkEigen&& other) noexcept = default;
	WatermarkEigen& operator=(const WatermarkEigen& other) = default;
	BufferType makeWatermark(const BufferType& inputImage, const BufferType& outputImage, float& watermarkStrength, MASK_TYPE maskType) override;
	float detectWatermark(const BufferType& inputImage, MASK_TYPE type) override;
};