#pragma once

#include "buffer.hpp"
#include "eigen_rgb_array.hpp"
#include "WatermarkBase.hpp"
#include <Eigen/Dense>
#include <string>
#include <vector>

class WatermarkEigen : public WatermarkBase 
{
private:
	int pad;
	unsigned int paddedRows, paddedCols;
	int pSquared, localSize, halfNeighborsSize;
	Eigen::ArrayXXf padded, mask, errorSequence, filteredEstimation, u, uStrengthened;
	Eigen::VectorXf coefficients;
	EigenArrayRGB watermarkedImage;
	Eigen::VectorXf RxVec, rx;
	Eigen::MatrixXf Rx;
	std::vector<Eigen::VectorXf> RxVec_all, rx_all;
	void createNeighbors(const Eigen::ArrayXXf& array, Eigen::VectorXf& x_, const int neighborSize, const int i, const int j) const;
	void resetRxVectors();
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