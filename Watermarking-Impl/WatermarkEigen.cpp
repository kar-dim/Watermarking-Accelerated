#include "buffer.hpp"
#include "eigen_rgb_array.hpp"
#include "EigenImage.hpp"
#include "WatermarkBase.hpp"
#include "WatermarkEigen.hpp"
#include <cmath>
#include <Eigen/Dense>
#include <omp.h>
#include <string>
#include <utility>
#include <vector>

using namespace Eigen;
using std::string;

//constructor to initialize all the necessary data
WatermarkEigen::WatermarkEigen(const unsigned int rows, const unsigned int cols, const string& randomMatrixPath, const int p, const float psnr) :
	WatermarkBase(rows, cols, randomMatrixPath, p, (255.0f / sqrt(pow(10.0f, psnr / 10.0f)))), pad(p / 2), paddedRows(rows + 2 * pad), 
	paddedCols(cols + 2 * pad), pSquared(p * p), halfNeighborsSize((pSquared - 1) / 2), padded(ArrayXXf::Zero(paddedRows, paddedCols))
{ }

//generate p x p neighbors
void WatermarkEigen::createNeighbors(const ArrayXXf& array, VectorXf& x_, const int neighborSize, const int i, const int j) const
{
	const auto x_temp = array.block(i - neighborSize, j - neighborSize, p, p).reshaped();
	//ignore the central pixel value
	x_.head(halfNeighborsSize) = x_temp.head(halfNeighborsSize);
	x_.tail(pSquared - halfNeighborsSize - 1) = x_temp.tail(halfNeighborsSize);
}

//computes the custom mask, in this case "NVF" mask
ArrayXXf WatermarkEigen::computeCustomMask(const ArrayXXf& image, const ArrayXXf& padded) const
{
	ArrayXXf nvf(baseRows, baseCols);
	const int neighborsSize = (p - 1) / 2;
#pragma omp parallel for
	for (int j = pad; j < baseCols + pad; j++)
	{
		for (int i = pad; i < baseRows + pad; i++)
		{
			const auto neighb = padded.block(i - neighborsSize, j - neighborsSize, p, p);
			const float mean = neighb.mean();
			const float variance = (neighb - mean).square().sum() / pSquared;
			nvf(i - pad, j - pad) = variance / (1.0f + variance);
		}
	}
	return nvf;
}

//Main watermark embedding method
//it embeds the watermark computed fom "inputImage" (always grayscale)
//into a new array based on "outputImage" (RGB or grayscale)
BufferType WatermarkEigen::makeWatermark(const BufferType& inputImage, const BufferType& outputImage, float& watermarkStrength, MASK_TYPE type)
{
	const ArrayXXf uStrength = computeStrengthenedWatermark(inputImage.getGray(), watermarkStrength, type);
	if (outputImage.isRGB())
	{
		EigenArrayRGB watermarkedImage;
#pragma omp parallel for
		for (int channel = 0; channel < 3; channel++)
			watermarkedImage[channel] = (outputImage.getRGB()[channel] + uStrength).cwiseMax(0).cwiseMin(255);
		return watermarkedImage;
	}
	ArrayXXf watermarkedImage = (outputImage.getGray() + uStrength).cwiseMax(0).cwiseMin(255);
	return BufferType(std::move(watermarkedImage));
}

ArrayXXf WatermarkEigen::computeStrengthenedWatermark(const ArrayXXf& inputImage, float& watermarkStrength, MASK_TYPE maskType)
{
	padded.block(pad, pad, inputImage.rows(), inputImage.cols()) = inputImage;
	ArrayXXf mask;
	if (maskType == MASK_TYPE::NVF)
		mask = computeCustomMask(inputImage, padded);
	else
	{
		ArrayXXf errorSequence;
		VectorXf coefficients;
		mask = computePredictionErrorMask(padded, errorSequence, coefficients, ME_MASK_CALCULATION_REQUIRED_YES);
	}
	const ArrayXXf u = mask * randomMatrix.getGray();
	watermarkStrength = strengthFactor / sqrt(u.square().sum() / (baseRows * baseCols));
	return u * watermarkStrength;
}

//compute Prediction error mask
ArrayXXf WatermarkEigen::computePredictionErrorMask(const ArrayXXf& paddedImage, ArrayXXf& errorSequence, VectorXf& coefficients, const bool maskNeeded) const
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
			createNeighbors(paddedImage, x_, neighborsSize, i, j);
			//calculate Rx and rx
			Rx_all[omp_get_thread_num()].noalias() += x_ * x_.transpose();
			rx_all[omp_get_thread_num()].noalias() += x_ * paddedImage(i, j);
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
	errorSequence = computeErrorSequence(paddedImage, coefficients);
	if (maskNeeded) 
	{
		auto errorSequenceAbs = errorSequence.abs();
		return errorSequenceAbs / errorSequenceAbs.maxCoeff();
	}
	return ArrayXXf();
}

//computes the prediction error sequence 
ArrayXXf WatermarkEigen::computeErrorSequence(const ArrayXXf& padded, const VectorXf& coefficients) const
{
	ArrayXXf errorSequence(baseRows, baseCols);
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
			errorSequence(i, j) = padded(iPad, jPad) - x_.dot(coefficients);
		}
	}
	return errorSequence;
}

//main mask detector for Me and NVF masks
float WatermarkEigen::detectWatermark(const BufferType& watermarkedImage, MASK_TYPE maskType)
{
	VectorXf a_z;
	ArrayXXf mask, e_z;
	const auto &watermarkedBuffer = watermarkedImage.getGray();
	//pad by using the preallocated block
	padded.block(pad, pad, watermarkedBuffer.rows(), watermarkedBuffer.cols()) = watermarkedBuffer;
	if (maskType == MASK_TYPE::NVF) 
	{
		computePredictionErrorMask(padded, e_z, a_z, ME_MASK_CALCULATION_REQUIRED_NO);
		mask = computeCustomMask(watermarkedBuffer, padded);
	}
	else
		mask = computePredictionErrorMask(padded, e_z, a_z, ME_MASK_CALCULATION_REQUIRED_YES);
	
	padded.block(pad, pad, watermarkedBuffer.rows(), watermarkedBuffer.cols()) = (mask * randomMatrix.getGray());
	const ArrayXXf e_u = computeErrorSequence(padded, a_z);
	float dot_ez_eu, d_ez, d_eu;
	
#pragma omp parallel sections
	{
#pragma omp section
		dot_ez_eu = e_z.cwiseProduct(e_u).sum();
#pragma omp section
		d_ez = std::sqrt(e_z.matrix().squaredNorm());
#pragma omp section
		d_eu = std::sqrt(e_u.matrix().squaredNorm());
	}
	return dot_ez_eu / (d_ez * d_eu);
}
