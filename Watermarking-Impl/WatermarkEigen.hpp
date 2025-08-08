#pragma once

#include "buffer.hpp"
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
	enum class Op { ADD, SUB };
	static constexpr int pSquared = p * p;
	static constexpr int pad = p / 2;
	static constexpr int localSize = pSquared - 1;
	static constexpr int blockRadius = p / 2;
	static constexpr int halfNeighborsSize = localSize / 2;
	static constexpr int tileSize = 64;
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

	//helper method to clamp the pixel value to the image boundaries if out of bounds
	inline float clampedValue(const ArrayXXf& img, int r, int c, const int rows, const int cols) 
	{
		return img(std::clamp(r, 0, rows - 1), std::clamp(c, 0, cols - 1));
	}

	//helper method for custom mask sums accumulation
	template <Op OP>
	inline void computeCustomMaskSums(const float pixelValue, float& sum, float& sumSq)
	{
		if constexpr (OP == Op::ADD)
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

	//main method to compute the custom mask
	void computeCustomMask(const ArrayXXf& image)
	{
		const int rows = static_cast<int>(baseRows);
		const int cols = static_cast<int>(baseCols);
		//helper lambda to process the border pixels (clamp if out of bounds)
		auto processBorder = [&](const int iStart, const int iEnd, int const jStart, const int jEnd)
		{
#pragma omp parallel for collapse(2) schedule(dynamic, 8)
			for (int j = jStart; j < jEnd; j++)
			{ 
				for (int i = iStart; i < iEnd; i++)
				{
					float sum = 0.0f, sumSq = 0.0f;
					for (int jj = -pad; jj <= pad; jj++)
						for (int ii = -pad; ii <= pad; ii++)
							computeCustomMaskSums<Op::ADD>(clampedValue(image, i + ii, j + jj, rows, cols), sum, sumSq);
					computeCustomMaskPixel(sum, sumSq, i, j);
				}
			}
		};

		//1) Center region
		if (rows > 2 * pad && cols > 2 * pad) 
		{
#pragma omp parallel for
			for (int j = pad; j < cols - pad; j++)
			{
				float sum = 0.0f, sumSq = 0.0f;
				for (int jj = -pad; jj <= pad; jj++) 
					for (int ii = -pad; ii <= pad; ii++) 
						computeCustomMaskSums<Op::ADD>(image(pad + ii, j + jj), sum, sumSq);
				computeCustomMaskPixel(sum, sumSq, pad, j);
				//slide window down for remaining center rows in this column
				for (int i = pad + 1; i < rows - pad; ++i)
				{
					//remove top row and add new bottom row
					for (int jj = -pad; jj <= pad; jj++) 
						computeCustomMaskSums<Op::SUB>(image(i - pad - 1, j + jj), sum, sumSq);
					for (int jj = -pad; jj <= pad; jj++)
						computeCustomMaskSums<Op::ADD>(image(i + pad, j + jj), sum, sumSq);
					computeCustomMaskPixel(sum, sumSq, i, j);
				}
			}
		}

		//2) BORDER region
		//process all pixels outside the core region with clamped sampling.
		//top border
		processBorder(0, std::min(pad, rows), 0, cols);
		//bottom border
		if (rows > pad)
			processBorder(std::max(rows - pad, 0), rows, 0, cols);
		//left border
		if (cols > 0)
			processBorder(pad, rows - pad, 0, std::min(pad, cols));
		//right border
		if (cols > pad)
			processBorder(pad, rows - pad, std::max(cols - pad, 0), cols);
	}

	//compute the strengthened watermark, calculated by multiplying the mask with the strengthened watermark (random matrix)
	void computeStrengthenedWatermark(const ArrayXXf& inputImage, float& watermarkStrength, MASK_TYPE maskType)
	{
		if (maskType == NVF)
			computeCustomMask(inputImage);
		else 
		{
			padded.block(pad, pad, inputImage.rows(), inputImage.cols()) = inputImage;
			computePredictionErrorData<maskCalcRequired>();
		}
		const auto u = mask * randomMatrix.getGray();
		watermarkStrength = strengthFactor / std::sqrt(u.square().sum() / (baseRows * baseCols));
		uStrengthened = u * watermarkStrength;
	}

	//helper method to calculate a tile and apply a function to each
	//used in prediction error calculations
	template <typename Func>
	inline void applyToTilesParallel(Func&& func)
	{
		const int rowsLimit = static_cast<int>(baseRows + pad);
		constexpr int center = pad;
#pragma omp parallel
		{
			TileMatrix tile;
			const int threadId = omp_get_thread_num();
#pragma omp for
			for (int j = pad; j < baseCols + pad; ++j)
			{
				for (int i = pad; i < baseRows + pad; i += tileSize)
				{
					//generate (p x p) - 1 neighbors for each tile
					const int tileRows = std::min(rowsLimit - i, tileSize);
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
					//call provided function for each tile
					for (int tileRow = 0; tileRow < tileRows; tileRow++)
						func(tile, i, j, tileRow, threadId);
				}
			}
		}
	}

	//compute Prediction error data (coefficients, error sequence), and if needed, prediction error mask
	template<bool maskNeeded>
	void computePredictionErrorData()
	{
		meMatrixData.setZero();
		applyToTilesParallel([&](const TileMatrix& tile, const int i, const int j, const int tileRow, const int threadId) {
			meMatrixData.computePredictionErrorMatrices(tile.col(tileRow), padded(i + tileRow, j), threadId);
		});
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
		applyToTilesParallel([&](const TileMatrix& tile, const int i, const int j, const int tileRow, const int) {
			outputErrorSequence(i - pad + tileRow, j - pad) = padded(i + tileRow, j) - tile.col(tileRow).dot(coefficients);
		});
	}
};