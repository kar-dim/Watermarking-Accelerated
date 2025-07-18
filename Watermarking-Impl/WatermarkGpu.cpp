#include "buffer.hpp"
#include "WatermarkBase.hpp"
#include "WatermarkGpu.hpp"
#include <arrayfire.h>
#include <cmath>
#include <utility>

void WatermarkGPU::displayArray(const af::array& array, const int width, const int height)
{
	af::Window window(width, height);
	while (!window.close())
		window.image(array);
}

BufferType WatermarkGPU::makeWatermark(const BufferType& inputImage, const BufferType& outputImage, float& watermarkStrength, const MASK_TYPE maskType)
{
	af::array mask, inputErrorSequence, inputCoefficients;
	if (maskType == ME)
	{
		computePredictionErrorData(inputImage, inputErrorSequence, inputCoefficients, true);
		//if the system is not solvable, don't waste time embeding the watermark, return output image without modification
		if (inputCoefficients.elements() == 0)
			return outputImage;
		mask = computePredictionErrorMask<false>(inputErrorSequence);
	}
	else
		mask = computeCustomMask(inputImage);
	const af::array u = mask * randomMatrix;
	watermarkStrength = strengthFactor / static_cast<float>(af::norm(u) / std::sqrt(u.elements()));
	return af::clamp(outputImage + (u * watermarkStrength), 0, 255);
}

float WatermarkGPU::detectWatermark(const BufferType& inputImage, const MASK_TYPE maskType)
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

std::pair<af::array, af::array> WatermarkGPU::transformCorrelationArrays(const af::array& RxPartial, const af::array& rxPartial) const
{
	const int localSize = (p * p) - 1;
	const int localSizeSq = localSize * localSize;
	const auto paddedElems = RxPartial.dims(0) * RxPartial.dims(1);
	//reduction sum of blocks
	//all [p^2-1,1] blocks will be summed in rx
	//all [p^2-1, p^2-1] blocks will be summed in Rx
	const af::array Rx = af::moddims(af::sum(af::moddims(RxPartial, localSizeSq, paddedElems / localSizeSq), 1), localSize, localSize);
	const af::array rx = af::sum(af::moddims(rxPartial, localSize, paddedElems / localSizeSq), 1);
	return std::make_pair(Rx, rx);
}