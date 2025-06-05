#pragma once

#include "WatermarkBase.hpp"
#include <arrayfire.h>
#include <concepts>
#include <utility>

class WatermarkGPU {
public:
	virtual ~WatermarkGPU() = default;

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
	virtual void initializeMemory() = 0;
	virtual af::array computeCustomMask() const = 0;
	virtual af::array computeScaledNeighbors(const af::array& coefficients) const = 0;
	virtual af::array computePredictionErrorMask(const af::array& image, af::array& errorSequence, af::array& coefficients, const bool maskNeeded) const = 0;
	virtual void copyDataToTexture(const af::array& image) const = 0;

	//Main watermark embedding method for GPU-based implementations
	af::array makeWatermarkGpu(const af::array& inputImage, const af::array& outputImage, const af::array& randomMatrix, const float strengthFactor, float& watermarkStrength, MASK_TYPE maskType)
	{
		af::array mask, errorSequence, coefficients;
		copyDataToTexture(inputImage);
		if (maskType == MASK_TYPE::ME)
		{
			mask = computePredictionErrorMask(inputImage, errorSequence, coefficients, ME_MASK_CALCULATION_REQUIRED_YES);
			//if the system is not solvable, don't waste time embeding the watermark, return output image without modification
			if (coefficients.elements() == 0)
				return outputImage;
		}
		else
			mask = computeCustomMask();
		const af::array u = mask * randomMatrix;
		watermarkStrength = strengthFactor / static_cast<float>(af::norm(u) / sqrt(inputImage.elements()));
		return af::clamp(outputImage + (u * watermarkStrength), 0, 255);
	}

	//main detector method for GPU-based implementations
	float detectWatermarkGpu(const af::array& watermarkedImage, const af::array& randomMatrix, MASK_TYPE maskType)
	{
		af::array mask, errorSequenceW, coefficients;
		copyDataToTexture(watermarkedImage);
		if (maskType == MASK_TYPE::NVF)
		{
			computePredictionErrorMask(watermarkedImage, errorSequenceW, coefficients, ME_MASK_CALCULATION_REQUIRED_NO);
			mask = computeCustomMask();
		}
		else
			mask = computePredictionErrorMask(watermarkedImage, errorSequenceW, coefficients, ME_MASK_CALCULATION_REQUIRED_YES);
		//if the system is not solvable, don't waste time computing the correlation, there is no watermark
		if (coefficients.elements() == 0)
			return 0.0f;
		const af::array u = mask * randomMatrix;
		return computeCorrelation(computeErrorSequence(u, coefficients), errorSequenceW);
	}

	//helper method used in detectors
	float computeCorrelation(const af::array& e_u, const af::array& e_z) const
	{
		return af::dot<float>(af::flat(e_u), af::flat(e_z)) / static_cast<float>(af::norm(e_z) * af::norm(e_u));
	}

	//helper method that calculates the error sequence by using a supplied prediction filter coefficients
	af::array computeErrorSequence(const af::array& u, const af::array& coefficients) const
	{
		copyDataToTexture(u);
		return u - computeScaledNeighbors(coefficients);
	}

	//helper method to sum the incomplete Rx_partial and rxPartial arrays which were produced from the custom kernel
	//and to transform them to the correct size, so that they can be used by the system solver
	std::pair<af::array, af::array> transformCorrelationArrays(const af::array& RxPartial, const af::array& rxPartial, const int p) const
	{
		const int neighborsSize = (p * p) - 1;
		const int neighborsSizeSq = neighborsSize * neighborsSize;
		const auto paddedElems = RxPartial.dims(0) * RxPartial.dims(1);
		//reduction sum of blocks
		//all [p^2-1,1] blocks will be summed in rx
		//all [p^2-1, p^2-1] blocks will be summed in Rx
		const af::array Rx = af::moddims(af::sum(af::moddims(RxPartial, neighborsSizeSq, paddedElems / neighborsSizeSq), 1), neighborsSize, neighborsSize);
		const af::array rx = af::sum(af::moddims(rxPartial, neighborsSize, paddedElems / (8 * neighborsSize)), 1);
		return std::make_pair(Rx, rx);
	}
};