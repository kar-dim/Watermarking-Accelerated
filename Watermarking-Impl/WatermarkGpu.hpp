#pragma once
#include "buffer.hpp"
#include "WatermarkBase.hpp"
#include <arrayfire.h>
#include <concepts>
#include <string>
#include <utility>

/*!
 *  \brief  Functions for watermark computation and detection, Base GPU class.
 *			GPU implementations must inherit from this class.
 *  \author Dimitris Karatzas
 */
template<int p>
class WatermarkGPU : public WatermarkBase
{
public:
	WatermarkGPU<p>(const unsigned int rows, const unsigned int cols, const std::string& randomMatrixPath, const float psnr)
		: WatermarkBase(rows, cols, randomMatrixPath, psnr)
	{ }

	WatermarkGPU<p>(const unsigned int rows, const unsigned int cols, const ImageBuffer& randomMatrix, const float strengthFactor)
		: WatermarkBase(rows, cols, randomMatrix, strengthFactor)
	{ }

	~WatermarkGPU<p>() override = default;

	void makeWatermark(const ImageBuffer& inputGrayImage, const ImageBuffer& inputImage, ImageBuffer& output, float& watermarkStrength, const MASK_TYPE maskType)
	{
		af::array mask, inputErrorSequence, inputCoefficients;
		if (maskType == ME)
		{
			computePredictionErrorData(inputGrayImage, inputErrorSequence, inputCoefficients, true);
			//if the system is not solvable, don't waste time embeding the watermark
			//set the output to the input image and exit
			if (inputCoefficients.elements() == 0)
			{
				output = inputImage;
				return;
			}
			mask = computePredictionErrorMask<false>(inputErrorSequence);
		}
		else
			mask = computeCustomMask(inputGrayImage);
		const af::array u = mask * randomMatrix;
		watermarkStrength = strengthFactor / static_cast<float>(af::norm(u) / std::sqrt(u.elements()));
		output = af::clamp(inputImage + (u * watermarkStrength), 0, 255);
	}

	float detectWatermark(const ImageBuffer& inputImage, const MASK_TYPE maskType)
	{
		af::array mask, errorSequenceW, coefficients;
		computePredictionErrorData(inputImage, errorSequenceW, coefficients, false);
		//if the system is not solvable, don't waste time computing the correlation, there is no watermark
		if (coefficients.elements() == 0)
			return 0.0f;
		mask = maskType == NVF ? computeCustomMask(inputImage) : computePredictionErrorMask<true>(errorSequenceW);
		const af::array u = mask * randomMatrix;
		return computeCorrelation(computeErrorSequence(u, coefficients, false), errorSequenceW);
	}

	//helper method to unlock multiple af::arrays (return memory to ArrayFire)
	template<std::same_as<af::array>... Args>
	static void unlockArrays(const Args&... arrays) { (arrays.unlock(), ...); }

	//helper method to display an af::array in a window
	static void displayArray(const af::array& array, const int width = 1600, const int height = 900)
	{
		af::Window window(width, height);
		while (!window.close())
			window.image(array);
	}

protected:
	static constexpr int pSquared = p * p;
	static constexpr int pad = p / 2;
	static constexpr int localSize = pSquared - 1;
	static constexpr int localSizeSq = localSize * localSize;

	//computes custom Mask
	virtual af::array computeCustomMask(const af::array& image) const = 0;
	
	//computes error sequence, used in prediction error mask
	virtual af::array computeErrorSequence(const af::array& image, const af::array& coefficients, const bool calculateAbs) const = 0;
	
	//Used in both creation and detection of the watermark.
	//Calculates error sequence and prediction error filter (coefficients)
	virtual void computePredictionErrorData(const af::array& image, af::array& errorSequence, af::array& coefficients, const bool calculateAbs) const = 0;
	
	//helper method used in detectors
	virtual float computeCorrelation(const af::array& e_u, const af::array& e_z) const = 0;

	//compute prediction error mask
	template<bool CALC_ABS>
	af::array computePredictionErrorMask(const af::array& errorSequence) const
	{
		const af::array& input = CALC_ABS ? af::abs(errorSequence) : errorSequence;
		return input / af::max<float>(input);
	}

	//helper method to sum the incomplete Rx_partial and rxPartial arrays which were produced from the custom kernel
	//and to transform them to the correct size, so that they can be used by the system solver
	std::pair<af::array, af::array> transformCorrelationArrays(const af::array& RxPartial, const af::array& rxPartial) const
	{
		const auto paddedElems = RxPartial.dims(0) * RxPartial.dims(1);
		//reduction sum of blocks
		//all [p^2-1,1] blocks will be summed in rx
		//all [p^2-1, p^2-1] blocks will be summed in Rx
		const af::array Rx = af::moddims(af::sum(af::moddims(RxPartial, localSizeSq, paddedElems / localSizeSq), 1), localSize, localSize);
		const af::array rx = af::sum(af::moddims(rxPartial, localSize, paddedElems / localSizeSq), 1);
		return std::make_pair(Rx, rx);
	}
};