#pragma once
#include "buffer.hpp"
#include "WatermarkBase.hpp"
#include <arrayfire.h>
#include <concepts>
#include <stdexcept>
#include <string>
#include <utility>

/*!
 *  \brief  Functions for watermark computation and detection, Base GPU class.
 *			GPU implementations must inherit from this class.
 *  \author Dimitris Karatzas
 */
class WatermarkGPU : public WatermarkBase
{
public:
	WatermarkGPU(const unsigned int rows, const unsigned int cols, const std::string& randomMatrixPath, const float psnr, const int p) 
		: WatermarkBase(rows, cols, randomMatrixPath, psnr), p(p)
	{ }

	WatermarkGPU(const unsigned int rows, const unsigned int cols, const BufferType& randomMatrix, const float strengthFactor, const int p)
		: WatermarkBase(rows, cols, randomMatrix, strengthFactor), p(p)
	{
		if (p != 3 && p != 5 && p != 7 && p != 9)
			throw std::invalid_argument("Unsupported value for p. Allowed values: 3, 5, 7, 9.");
	}

	~WatermarkGPU() override = default;

	BufferType makeWatermark(const BufferType& inputImage, const BufferType& outputImage, float& watermarkStrength, const MASK_TYPE maskType) override;

	float detectWatermark(const BufferType& inputImage, const MASK_TYPE maskType) override;

	//helper method to unlock multiple af::arrays (return memory to ArrayFire)
	template<std::same_as<af::array>... Args>
	static void unlockArrays(const Args&... arrays) { (arrays.unlock(), ...); }

	//helper method to display an af::array in a window
	static void displayArray(const af::array& array, const int width = 1600, const int height = 900);

protected:
	int p;

	//computes custom Mask
	virtual af::array computeCustomMask(const af::array& image) const = 0;
	
	//computes scaled neighbors array used in prediction error mask
	virtual af::array computeScaledNeighbors(const af::array& image, const af::array& coefficients) const = 0;
	
	//Used in both creation and detection of the watermark.
	//Calculates error sequence and prediction error filter (coefficients)
	virtual void computePredictionErrorData(const af::array& image, af::array& errorSequence, af::array& coefficients) const = 0;
	
	//compute prediction error mask
	af::array computePredictionErrorMask(const af::array& errorSequence) const;

	//helper method used in detectors
	float computeCorrelation(const af::array& e_u, const af::array& e_z) const;

	//helper method that calculates the error sequence by using a supplied prediction filter coefficients
	af::array computeErrorSequence(const af::array& u, const af::array& coefficients) const;

	//helper method to sum the incomplete Rx_partial and rxPartial arrays which were produced from the custom kernel
	//and to transform them to the correct size, so that they can be used by the system solver
	std::pair<af::array, af::array> transformCorrelationArrays(const af::array& RxPartial, const af::array& rxPartial) const;
};