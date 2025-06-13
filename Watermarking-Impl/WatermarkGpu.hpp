#pragma once

#include "WatermarkBase.hpp"
#include <arrayfire.h>
#include <concepts>
#include <utility>

class WatermarkGPU 
{
public:
	virtual ~WatermarkGPU() = default;

	//helper method to unlock multiple af::arrays (return memory to ArrayFire)
	template<std::same_as<af::array>... Args>
	static void unlockArrays(const Args&... arrays) { (arrays.unlock(), ...); }

	//helper method to display an af::array in a window
	static void displayArray(const af::array& array, const int width = 1600, const int height = 900);

protected:
	//initialize internal GPU buffers
	virtual void initializeGpuMemory() = 0;

	//computes custom Mask
	virtual af::array computeCustomMask() const = 0;
	
	//computes scaled neighbors array used in prediction error mask
	virtual af::array computeScaledNeighbors(const af::array& coefficients) const = 0;
	
	//Compute prediction error mask. Used in both creation and detection of the watermark.
	//can also calculate error sequence and prediction error filter
	virtual af::array computePredictionErrorMask(const af::array& image, af::array& errorSequence, af::array& coefficients, const bool maskNeeded) const = 0;
	
	//copy data to texture and transfer ownership back to arrayfire
	virtual void copyDataToTexture(const af::array& image) const = 0;

	//main watermark embedding method for GPU-based implementations
	af::array makeWatermarkGpu(const af::array& inputImage, const af::array& outputImage, const af::array& randomMatrix, const float strengthFactor, float& watermarkStrength, MASK_TYPE maskType);

	//main detector method for GPU-based implementations
	float detectWatermarkGpu(const af::array& inputImage, const af::array& randomMatrix, MASK_TYPE maskType);

	//helper method used in detectors
	float computeCorrelation(const af::array& e_u, const af::array& e_z) const;

	//helper method that calculates the error sequence by using a supplied prediction filter coefficients
	af::array computeErrorSequence(const af::array& u, const af::array& coefficients) const;

	//helper method to sum the incomplete Rx_partial and rxPartial arrays which were produced from the custom kernel
	//and to transform them to the correct size, so that they can be used by the system solver
	std::pair<af::array, af::array> transformCorrelationArrays(const af::array& RxPartial, const af::array& rxPartial, const int p) const;
};