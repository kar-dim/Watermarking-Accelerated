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
	static constexpr int tileSize = 64;
	using LocalVector = Eigen::Matrix<float, localSize, 1>;
	using TileMatrix = Eigen::Matrix<float, localSize, tileSize>;
	using ArrayXXf = Eigen::ArrayXXf;

public:
	WatermarkEigen<p>(const unsigned int rows, const unsigned int cols, const std::string& randomMatrixPath, const float psnr) :
		WatermarkBase(rows, cols, randomMatrixPath, psnr), mask(rows, cols), errorSequence(rows, cols), 
		filteredEstimation(rows, cols), u(rows, cols), uStrengthened(rows, cols), meMatrixData(omp_get_max_threads())
	{ }

	//main watermark embedding method
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

	//main watermark detection method
	float detectWatermark(const BufferType& inputImage, MASK_TYPE maskType) override
	{
		const auto& watermarkedBuffer = inputImage.getGray();
		if (maskType == NVF)
		{
			computePredictionErrorData<maskCalcNotRequired>(watermarkedBuffer);
			computeCustomMask(watermarkedBuffer);
		}
		else
			computePredictionErrorData<maskCalcRequired>(watermarkedBuffer);

		u = mask * randomMatrix.getGray();
		computeErrorSequence(u, filteredEstimation);
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
		float correlation = dot_ez_eu / (d_ez * d_eu);
		return std::isfinite(correlation) ? correlation : 0.0f;
	}

private:
	ArrayXXf mask, errorSequence, filteredEstimation, u, uStrengthened;
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

	//helper method for calling lambda border handlers for both custom and prediction error masks
	template <typename ProcessBorderFunc>
	inline void computeMaskBorders(const int startRow, const int endRow, const int startCol, const int endCol, bool hasCenterRegion, const ProcessBorderFunc& processBorder)
	{
		if (startRow > 0)
			processBorder(0, startRow, 0, baseCols);
		if (endRow < baseRows)
			processBorder(endRow, baseRows, 0, baseCols);
		if (startCol > 0 && hasCenterRegion)
			processBorder(startRow, endRow, 0, startCol);
		if (endCol < baseCols && hasCenterRegion)
			processBorder(startRow, endRow, endCol, baseCols);
	}

	//main method to compute the custom mask
	void computeCustomMask(const ArrayXXf& image)
	{
		const int rows = static_cast<int>(baseRows);
		const int cols = static_cast<int>(baseCols);
		const int startRow = pad;
		const int endRow = rows - pad;
		const int startCol = pad;
		const int endCol = cols - pad;
		const bool hasCenterRegion = (endRow > startRow) && (endCol > startCol);
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

		//process CENTER region
		if (hasCenterRegion)
		{
#pragma omp parallel for
			for (int j = startCol; j < endCol; j++)
			{
				float sum = 0.0f, sumSq = 0.0f;
				for (int jj = -pad; jj <= pad; jj++)
					for (int ii = -pad; ii <= pad; ii++)
						computeCustomMaskSums<Op::ADD>(image(pad + ii, j + jj), sum, sumSq);
				computeCustomMaskPixel(sum, sumSq, pad, j);
				//slide window down for remaining center rows in this column
				for (int i = startRow + 1; i < endRow; i++)
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
		//process BORDER region
		computeMaskBorders(startRow, endRow, startCol, endCol, hasCenterRegion, processBorder);
	}

	//compute the strengthened watermark, calculated by multiplying the mask with the strengthened watermark (random matrix)
	void computeStrengthenedWatermark(const ArrayXXf& inputImage, float& watermarkStrength, MASK_TYPE maskType)
	{
		if (maskType == NVF)
			computeCustomMask(inputImage);
		else 
			computePredictionErrorData<maskCalcRequired>(inputImage);
		const auto& gray = randomMatrix.getGray();

		//optimized calculation of the strengthened watermark
		float sumSq = 0.0f;
#pragma omp parallel for reduction(+:sumSq)
		for (auto i = 0; i < mask.size(); i++) 
		{
			float u = mask(i) * gray(i);
			uStrengthened(i) = u;
			sumSq += u * u;
		}
		watermarkStrength = strengthFactor / std::sqrt(sumSq / (baseRows * baseCols));
		uStrengthened *= watermarkStrength;
	}

	//helper method to load a tile block from the image into the tile matrix (either direct or clamped access for border pixels)
	template<typename PixelAccessor>
	void loadTileBlock(TileMatrix& tile, const ArrayXXf& image, const int i, const int j, const int tileRows, const PixelAccessor& pixelAccessor)
	{
		constexpr int center = pad;
		int k;
		for (int a = 0; a < tileRows; a++)
		{
			k = 0;
			for (int dj = 0; dj < p; dj++)
			{
				for (int di = 0; di < p; di++)
				{
					if (di == center && dj == center)
						continue;
					tile(k++, a) = pixelAccessor(i + a + di - center, j + dj - center);
				}
			}
		}
	}

	//helper method to calculate a tile and apply a function to each
	//used in prediction error calculations
	template <typename Func>
	void applyToTilesParallel(const ArrayXXf& image, Func&& func)
	{
		const int rows = static_cast<int>(baseRows);
		const int cols = static_cast<int>(baseCols);
		const int startRow = pad;
		const int endRow = rows - pad;
		const int startCol = pad;
		const int endCol = cols - pad;
		const bool hasCenterRegion = (endRow > startRow) && (endCol > startCol);
		auto directAccessor = [&](int x, int y) { return image(x, y); };
		auto clampedAccessor = [&](int x, int y) { return clampedValue(image, x, y, rows, cols); };
#pragma omp parallel
		{
			TileMatrix tile;
			const int threadId = omp_get_thread_num();
			//process CENTER region
			if (hasCenterRegion)
			{
#pragma omp for
				for (int j = startCol; j < endCol; j++)
				{
					for (int i = startRow; i < endRow; i += tileSize)
					{
						const int tileRows = std::min(endRow - i, tileSize);
						loadTileBlock(tile, image, i, j, tileRows, directAccessor);
						for (int tileRow = 0; tileRow < tileRows; tileRow++)
							func(tile, i, j, tileRow, threadId);
					}
				}
			}

			//helper lambda to process BORDER regions
			auto processBorder = [&](const int rowStart, const int rowEnd, const int colStart, const int colEnd)
			{
#pragma omp for collapse(2) schedule(dynamic, 8)
				for (int j = colStart; j < colEnd; j++)
				{
					for (int i = rowStart; i < rowEnd; i += tileSize)
					{
						const int tileRows = std::min(rowEnd - i, tileSize);
						loadTileBlock(tile, image, i, j, tileRows, clampedAccessor);
						for (int tileRow = 0; tileRow < tileRows; tileRow++)
							func(tile, i, j, tileRow, threadId);
					}
				}
			};

			//process BORDER regions
			computeMaskBorders(startRow, endRow, startCol, endCol, hasCenterRegion, processBorder);
		}
	}

	//compute Prediction error data (coefficients, error sequence), and if needed, prediction error mask
	template<bool maskNeeded>
	void computePredictionErrorData(const ArrayXXf& image)
	{
		meMatrixData.setZero();
		applyToTilesParallel(image, [&](const TileMatrix& tile, const int i, const int j, const int tileRow, const int threadId) {
			meMatrixData.computePredictionErrorMatrices(tile.col(tileRow), image(i + tileRow, j), threadId);
		});
		meMatrixData.computeCoefficients();
		//calculate ex(i,j)
		computeErrorSequence(image, errorSequence);
		if constexpr (maskNeeded)
		{
			auto errorSequenceAbs = errorSequence.abs();
			mask = errorSequenceAbs / errorSequenceAbs.maxCoeff();
		}
	}

	//computes the prediction error sequence of the input image
	void computeErrorSequence(const ArrayXXf& image, ArrayXXf& outputErrorSequence)
	{
		const auto& coefficients = meMatrixData.getCoefficients();
		applyToTilesParallel(image, [&](const TileMatrix& tile, const int i, const int j, const int tileRow, const int) {
			outputErrorSequence(i + tileRow, j) = image(i + tileRow, j) - tile.col(tileRow).dot(coefficients);
		});
	}
};