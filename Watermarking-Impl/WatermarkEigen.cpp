#include "buffer.hpp"
#include "eigen_rgb_array.hpp"
#include "EigenImage.hpp"
#include "WatermarkBase.hpp"
#include "WatermarkEigen.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <Eigen/Dense>
#include <omp.h>
#include <string>
//#include <utility>
#include <vector>

using namespace Eigen;
using std::string;

//constructor to initialize all the necessary data
WatermarkEigen::WatermarkEigen(const unsigned int rows, const unsigned int cols, const string& randomMatrixPath, const int p, const float psnr) :
	WatermarkBase(rows, cols, randomMatrixPath, p, (255.0f / sqrt(pow(10.0f, psnr / 10.0f)))), pad(p / 2), paddedRows(rows + 2 * pad), 
	paddedCols(cols + 2 * pad), pSquared(p * p), halfNeighborsSize((pSquared - 1) / 2), padded(ArrayXXf::Zero(paddedRows, paddedCols)), 
	mask(rows, cols), errorSequence(rows, cols), filteredEstimation(rows, cols), u(rows,cols), uStrengthened(rows, cols), coefficients(pSquared - 1),
	watermarkedImage{ [](int r, int c) { return std::array<ArrayXXf, 3>{ ArrayXXf(r, c), ArrayXXf(r, c), ArrayXXf(r, c) }; }(rows, cols) }
{ }

//generate p x p neighbors
void WatermarkEigen::createNeighbors(const ArrayXXf& array, VectorXf& x_, const int neighborSize, const int i, const int j) const
{
	const auto x_temp = array.block(i - neighborSize, j - neighborSize, p, p).reshaped();
	//ignore the central pixel value
	x_.head(halfNeighborsSize) = x_temp.head(halfNeighborsSize);
	x_.tail(pSquared - halfNeighborsSize - 1) = x_temp.tail(halfNeighborsSize);
}

void WatermarkEigen::onReinitialize()
{
	pad = p / 2;
	paddedRows = (baseRows + 2 * pad);
	paddedCols = (baseCols + 2 * pad);
	pSquared = p * p;
	halfNeighborsSize = (pSquared - 1) / 2;
	padded = ArrayXXf::Zero(paddedRows, paddedCols);
	mask = ArrayXXf(baseRows, baseCols);
	errorSequence = ArrayXXf(baseRows, baseCols);
	filteredEstimation = ArrayXXf(baseRows, baseCols);
	u = ArrayXXf(baseRows, baseCols);
	uStrengthened = ArrayXXf(baseRows, baseCols);
	coefficients = VectorXf(pSquared - 1);
	std::generate(watermarkedImage.begin(), watermarkedImage.end(), [this] { return ArrayXXf(baseRows, baseCols); });
}

void WatermarkEigen::computeCustomMask(const ArrayXXf& image) 
{
	const int neighborsSize = (p - 1) / 2;
#pragma omp parallel for
	for (int j = pad; j < baseCols + pad; j++)
	{
		for (int i = pad; i < baseRows + pad; i++)
		{
			const auto neighb = padded.block(i - neighborsSize, j - neighborsSize, p, p);
			const float mean = neighb.mean();
			const float variance = (neighb - mean).square().sum() / pSquared;
			mask(i - pad, j - pad) = variance / (1.0f + variance);
		}
	}
}

BufferType WatermarkEigen::makeWatermark(const BufferType& inputImage, const BufferType& outputImage, float& watermarkStrength, MASK_TYPE type)
{
	computeStrengthenedWatermark(inputImage.getGray(), watermarkStrength, type);
	if (outputImage.isRGB())
	{
#pragma omp parallel for
		for (int channel = 0; channel < 3; channel++)
			watermarkedImage[channel] = (outputImage.getRGB()[channel] + uStrengthened).cwiseMax(0).cwiseMin(255);
		return watermarkedImage;
	}
	return BufferType((outputImage.getGray() + uStrengthened).cwiseMax(0).cwiseMin(255));
}

//compute the strengthened watermark, calcaulated by multiplying the mask with the strengthened watermark (random matrix)
void WatermarkEigen::computeStrengthenedWatermark(const ArrayXXf& inputImage, float& watermarkStrength, MASK_TYPE maskType)
{
	padded.block(pad, pad, inputImage.rows(), inputImage.cols()) = inputImage;
	if (maskType == MASK_TYPE::NVF)
		computeCustomMask(inputImage);
	else
		computePredictionErrorMask(MASK_CALC_REQUIRED);
	u = mask * randomMatrix.getGray();
	watermarkStrength = strengthFactor / sqrt(u.square().sum() / (baseRows * baseCols));
	uStrengthened = u * watermarkStrength;
}

//compute Prediction error mask
void WatermarkEigen::computePredictionErrorMask(const bool maskNeeded)
{
	MatrixXf Rx = ArrayXXf::Zero(pSquared - 1, pSquared - 1);
	VectorXf rx = VectorXf::Zero(pSquared - 1);
	const int numThreads = omp_get_max_threads();
	std::vector<MatrixXf> Rx_all(numThreads);
	std::vector<VectorXf> rx_all(numThreads);
	for (int i = 0; i < numThreads; i++) 
	{
		Rx_all[i] = Rx;
		rx_all[i] = rx;
	}
	const int neighborsSize = (p - 1) / 2;
#pragma omp parallel for
	for (int j = pad; j < baseCols + pad; j++)
	{
		VectorXf x_(pSquared - 1);
		for (int i = pad; i < baseRows + pad; i++)
		{
			//calculate p^2 - 1 neighbors
			createNeighbors(padded, x_, neighborsSize, i, j);
			//calculate Rx and rx
			Rx_all[omp_get_thread_num()].noalias() += x_ * x_.transpose();
			rx_all[omp_get_thread_num()].noalias() += x_ * padded(i, j);
		}
	}
	//reduction sums of Rx,rx of each thread
	for (int i = 0; i < numThreads; i++) 
	{
		Rx.noalias() += Rx_all[i];
		rx.noalias() += rx_all[i];
	}
	coefficients = Rx.fullPivLu().solve(rx);
	//calculate ex(i,j)
	computeErrorSequence(errorSequence);
	if (maskNeeded) 
	{
		auto errorSequenceAbs = errorSequence.abs();
		mask = errorSequenceAbs / errorSequenceAbs.maxCoeff();
	}
}

//computes the prediction error sequence of the padded input image
void WatermarkEigen::computeErrorSequence(ArrayXXf& outputErrorSequence)
{
	const int neighborsSize = (p - 1) / 2;
#pragma omp parallel for
	for (int j = 0; j < baseCols; j++)
	{
		VectorXf x_(pSquared - 1);
		const int jPad = j + pad;
		for (int i = 0; i < baseRows; i++)
		{
			const int iPad = i + pad;
			createNeighbors(padded, x_, neighborsSize, iPad, jPad);
			outputErrorSequence(i, j) = padded(iPad, jPad) - x_.dot(coefficients);
		}
	}
}

float WatermarkEigen::detectWatermark(const BufferType& inputImage, MASK_TYPE maskType)
{
	const auto &watermarkedBuffer = inputImage.getGray();
	//pad by using the preallocated block
	padded.block(pad, pad, watermarkedBuffer.rows(), watermarkedBuffer.cols()) = watermarkedBuffer;
	if (maskType == MASK_TYPE::NVF) 
	{
		computePredictionErrorMask(MASK_CALC_NOT_REQUIRED);
		computeCustomMask(watermarkedBuffer);
	}
	else
		computePredictionErrorMask(MASK_CALC_REQUIRED);
	
	padded.block(pad, pad, watermarkedBuffer.rows(), watermarkedBuffer.cols()) = (mask * randomMatrix.getGray());
	computeErrorSequence(filteredEstimation);
	float dot_ez_eu, d_ez, d_eu;
	
#pragma omp parallel sections
	{
#pragma omp section
		dot_ez_eu = errorSequence.cwiseProduct(filteredEstimation).sum();
#pragma omp section
		d_ez = std::sqrt(errorSequence.matrix().squaredNorm());
#pragma omp section
		d_eu = std::sqrt(filteredEstimation.matrix().squaredNorm());
	}
	return dot_ez_eu / (d_ez * d_eu);
}
