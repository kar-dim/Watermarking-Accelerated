#pragma once

#include "buffer.hpp"
#include "eigen_rgb_array.hpp"
#include "PredictionErrorMatrixData.hpp"
#include "WatermarkBase.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <Eigen/Dense>
#include <omp.h>
#include <string>

/*!
 *  \brief  Functions for watermark computation and detection, Eigen implementation.
 *  \author Dimitris Karatzas
 */
template<int p>
class WatermarkEigen final : public WatermarkBase 
{
private:
	static constexpr int pSquared = p * p;
	static constexpr int pad = p / 2;
	static constexpr int localSize = pSquared - 1;
	static constexpr int blockRadius = p / 2;
	static constexpr int halfNeighborsSize = localSize / 2;
	using LocalVector = Eigen::Matrix<float, localSize, 1>;
	using ArrayXXf = Eigen::ArrayXXf;

public:
	WatermarkEigen<p>(const unsigned int rows, const unsigned int cols, const std::string& randomMatrixPath, const float psnr) :
		WatermarkBase(rows, cols, randomMatrixPath, psnr), paddedRows(rows + 2 * pad),
		paddedCols(cols + 2 * pad), padded(ArrayXXf::Zero(paddedRows, paddedCols)),
		mask(rows, cols), errorSequence(rows, cols), filteredEstimation(rows, cols), u(rows, cols), uStrengthened(rows, cols),
		meMatrixData(omp_get_max_threads())
	{ }

	void makeWatermark(const BufferType& inputGrayImage, const BufferType& inputImage, BufferType& output, float& watermarkStrength, const MASK_TYPE maskType) override
	{
		computeStrengthenedWatermark(inputGrayImage.getGray(), watermarkStrength, maskType);
		if (inputImage.isRGB())
		{
#pragma omp parallel for
			for (int channel = 0; channel < 3; channel++)
				output.getRGB()[channel] = (inputImage.getRGB()[channel] + uStrengthened).cwiseMax(0).cwiseMin(255);
			return;
		}
		output = (inputImage.getGray() + uStrengthened).cwiseMax(0).cwiseMin(255);
	}

	float detectWatermark(const BufferType& inputImage, MASK_TYPE maskType) override
	{
		const auto& watermarkedBuffer = inputImage.getGray();
		//pad by using the preallocated block
		padded.block(pad, pad, watermarkedBuffer.rows(), watermarkedBuffer.cols()) = watermarkedBuffer;
		if (maskType == NVF)
		{
			computePredictionErrorData<maskCalcNotRequired>();
			computeCustomMask(watermarkedBuffer);
		}
		else
			computePredictionErrorData<maskCalcRequired>();

		padded.block(pad, pad, watermarkedBuffer.rows(), watermarkedBuffer.cols()) = (mask * randomMatrix.getGray());
		computeErrorSequence(filteredEstimation);
		float dot_ez_eu, d_ez, d_eu;

#pragma omp parallel sections
		{
#pragma omp section
			dot_ez_eu = errorSequence.cwiseProduct(filteredEstimation).sum();
#pragma omp section
			d_ez = errorSequence.matrix().norm();
#pragma omp section
			d_eu = filteredEstimation.matrix().norm();
		}
		return dot_ez_eu / (d_ez * d_eu);
	}
private:
	unsigned int paddedRows, paddedCols;
	ArrayXXf padded, mask, errorSequence, filteredEstimation, u, uStrengthened;
	PredictionErrorMatrixData<p> meMatrixData;

	//generate (p x p) - 1 neighbors
	void createNeighbors(const ArrayXXf& array, LocalVector& x_, const int i, const int j) const
	{
		const auto& block = array.block<p, p>(i - blockRadius, j - blockRadius).reshaped();
		//ignore the central pixel value
		x_.head(halfNeighborsSize) = block.head(halfNeighborsSize);
		x_.tail(pSquared - halfNeighborsSize - 1) = block.tail(halfNeighborsSize);
	}

	void computeCustomMask(const ArrayXXf& image)
	{
#pragma omp parallel for
		for (int j = pad; j < baseCols + pad; j++)
		{
			for (int i = pad; i < baseRows + pad; i++)
			{
				float sum = 0.0f, sumSq = 0.0f;
				for (int jj = -pad; jj <= pad; jj++)
				{
					for (int ii = -pad; ii <= pad; ii++)
					{
						float pixelValue = padded(i + ii, j + jj);
						sum += pixelValue;
						sumSq += pixelValue * pixelValue;
					}
				}
				float mean = sum / pSquared;
				float variance = (sumSq / pSquared) - (mean * mean);
				mask(i - pad, j - pad) = std::max(variance / (1.0f + variance), 0.0f);
			}
		}
	}
	//compute the strengthened watermark, calcalated by multiplying the mask with the strengthened watermark (random matrix)
	void computeStrengthenedWatermark(const ArrayXXf& inputImage, float& watermarkStrength, MASK_TYPE maskType)
	{
		padded.block(pad, pad, inputImage.rows(), inputImage.cols()) = inputImage;
		if (maskType == NVF)
			computeCustomMask(inputImage);
		else
			computePredictionErrorData<maskCalcRequired>();
		u = mask * randomMatrix.getGray();
		watermarkStrength = strengthFactor / sqrt(u.square().sum() / (baseRows * baseCols));
		uStrengthened = u * watermarkStrength;
	}

	//compute Prediction error data (coefficients, error sequence), and if needed, prediction error mask
	template<bool maskNeeded>
	void computePredictionErrorData()
	{
		meMatrixData.setZero();

#pragma omp parallel for
		for (int j = pad; j < baseCols + pad; j++)
		{
			LocalVector x_;
			for (int i = pad; i < baseRows + pad; i++)
			{
				//calculate p^2 - 1 neighbors
				createNeighbors(padded, x_, i, j);
				//calculate Rx optimized by using a vector representing the lower-triangular only instead of a matrix
				meMatrixData.computePredictionErrorMatrices(x_, padded(i, j), omp_get_thread_num());
			}
		}
		meMatrixData.computeCoefficients();
		//calculate ex(i,j)
		computeErrorSequence(errorSequence);
		if constexpr (maskNeeded)
		{
			auto errorSequenceAbs = errorSequence.abs();
			mask = errorSequenceAbs / errorSequenceAbs.maxCoeff();
		}
	}

	//computes the prediction error sequence of the padded input image
	void computeErrorSequence(ArrayXXf& outputErrorSequence)
	{
		const auto& coefficients = meMatrixData.getCoefficients();
#pragma omp parallel for
		for (int j = 0; j < baseCols; j++)
		{
			LocalVector x_;
			const int jPad = j + pad;
			for (int i = 0; i < baseRows; i++)
			{
				const int iPad = i + pad;
				createNeighbors(padded, x_, iPad, jPad);
				outputErrorSequence(i, j) = padded(iPad, jPad) - x_.dot(coefficients);
			}
		}
	}
};