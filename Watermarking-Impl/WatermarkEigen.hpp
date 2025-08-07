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
	static constexpr int tileSize = 64; //Tile size per thread (rows per tile)
	using LocalVector = Eigen::Matrix<float, localSize, 1>;
	using TileMatrix = Eigen::Matrix<float, localSize, tileSize>;
	using ArrayXXf = Eigen::ArrayXXf;

public:
	WatermarkEigen<p>(const unsigned int rows, const unsigned int cols, const std::string& randomMatrixPath, const float psnr) :
		WatermarkBase(rows, cols, randomMatrixPath, psnr), padded(ArrayXXf::Zero(rows + 2 * pad, cols + 2 * pad)),
		mask(rows, cols), errorSequence(rows, cols), filteredEstimation(rows, cols), uStrengthened(rows, cols),
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
	ArrayXXf padded, mask, errorSequence, filteredEstimation, uStrengthened;
	PredictionErrorMatrixData<p> meMatrixData;

	//generate (p x p) - 1 neighbors for a tile
	void fillLocalTile(const Eigen::ArrayXXf& padded, int i, int j, TileMatrix& tile, int tileRows)
	{
		constexpr int center = pad;
		for (int a = 0; a < tileRows; ++a)
		{
			int k = 0;
			for (int dj = 0; dj < p; ++dj)
			{
				for (int di = 0; di < p; ++di)
				{
					if (di == center && dj == center)
						continue;
					tile(k++, a) = padded(i + a + di - center, j + dj - center);
				}
			}
		}
	}

	//helper method for custom mask sums accumulation
	template <bool ADD = true>
	inline void computeCustomMaskSums(const float pixelValue, float& sum, float& sumSq)
	{
		if constexpr (ADD)
		{
			sum += pixelValue;
			sumSq += pixelValue * pixelValue;
		}
		else
		{
			sum -= pixelValue;
			sumSq -= pixelValue * pixelValue;
		}
	}

	//helper method for custom mask calculation per pixel
	inline void computeCustomMaskPixel(const float sum, const float sumSq, const int i, const int j)
	{
		float mean = sum / pSquared;
		float variance = (sumSq / pSquared) - (mean * mean);
		mask(i, j) = std::max(variance / (1.0f + variance), 0.0f);
	}

	void computeCustomMask(const ArrayXXf& image)
	{
#pragma omp parallel for
		for (int j = pad; j < baseCols + pad; j++)
		{
			float sum = 0.0f, sumSq = 0.0f;
			//initialize window and process first pixel in this column
			for (int jj = -pad; jj <= pad; jj++)
				for (int ii = -pad; ii <= pad; ii++)
					computeCustomMaskSums<true>(padded(pad + ii, j + jj), sum, sumSq);
			computeCustomMaskPixel(sum, sumSq, 0, j - pad);

			//slide window down for remaining pixels in this column
			for (int i = pad + 1; i < baseRows + pad; i++)
			{
				//remove top row and add bottom row of window
				for (int jj = -pad; jj <= pad; jj++)
					computeCustomMaskSums<false>(padded(i - pad - 1, j + jj), sum, sumSq);
				for (int jj = -pad; jj <= pad; jj++)
					computeCustomMaskSums<true>(padded(i + pad, j + jj), sum, sumSq);
				computeCustomMaskPixel(sum, sumSq, i - pad, j - pad);
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
		const auto u = mask * randomMatrix.getGray();
		watermarkStrength = strengthFactor / sqrt(u.square().sum() / (baseRows * baseCols));
		uStrengthened = u * watermarkStrength;
	}

	//compute Prediction error data (coefficients, error sequence), and if needed, prediction error mask
	template<bool maskNeeded>
	void computePredictionErrorData()
	{
		meMatrixData.setZero();

#pragma omp parallel for
		for (int j = pad; j < baseCols + pad; ++j)
		{
			TileMatrix tile;
			for (int i = pad; i < baseRows + pad; i += tileSize)
			{
				const int tileRows = std::min(static_cast<int>(baseRows + pad - i), tileSize);
				fillLocalTile(padded, i, j, tile, tileRows);
				for (int tileRow = 0; tileRow < tileRows; tileRow++)
					meMatrixData.computePredictionErrorMatrices(tile.col(tileRow), padded(i + tileRow, j), omp_get_thread_num());
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
		for (int j = pad; j < baseCols + pad; ++j)
		{
			TileMatrix tile;
			for (int i = pad; i < baseRows + pad; i += tileSize)
			{
				int tileRows = std::min(static_cast<int>(baseRows + pad - i), tileSize);
				fillLocalTile(padded, i, j, tile, tileRows);
				for (int tileRow = 0; tileRow < tileRows; tileRow++)
					outputErrorSequence(i - pad + tileRow, j - pad) = padded(i + tileRow, j) - tile.col(tileRow).dot(coefficients);
			}
		}
	}
};