#include "cuda_utils.hpp"
#include "kernels/kernels.cuh"
#include "WatermarkCuda.cuh"
#include "WatermarkGpu.hpp"
#include <af/cuda.h>
#include <arrayfire.h>
#include <cuda_runtime.h>
#include <string>
#include <utility>

using std::string;

cudaStream_t WatermarkCuda::afStream = afcu::getStream(afcu::getNativeId(af::getDevice()));

//initialize data and memory
WatermarkCuda::WatermarkCuda(const unsigned int rows, const unsigned int cols, const string& randomMatrixPath, const int p, const float psnr)
	: WatermarkGPU(rows, cols, randomMatrixPath, psnr, p), meKernelDims(align<64>(cols), rows)
{ }

//copy constructor
WatermarkCuda::WatermarkCuda(const WatermarkCuda& other) : WatermarkGPU(other.baseRows, other.baseCols, other.randomMatrix, other.strengthFactor, other.p),
	meKernelDims(other.meKernelDims)
{ }

//move constructor
WatermarkCuda::WatermarkCuda(WatermarkCuda&& other) noexcept : 
	WatermarkGPU(other.baseRows, other.baseCols, std::move(other.randomMatrix), other.strengthFactor, other.p), meKernelDims(other.meKernelDims)
{ }

//helper method to copy the parameters of another watermark object (for move/copy operators)
void WatermarkCuda::copyParams(const WatermarkCuda& other) noexcept
{
	baseRows = other.baseRows;
	baseCols = other.baseCols;
	meKernelDims = other.meKernelDims;
	p = other.p;
	strengthFactor = other.strengthFactor;
}

//move assignment operator
WatermarkCuda& WatermarkCuda::operator=(WatermarkCuda&& other) noexcept
{
	if (this != &other) 
	{
		copyParams(other);
		randomMatrix = std::move(other.randomMatrix);
	}
	return *this;
}

//copy assignment operator
WatermarkCuda& WatermarkCuda::operator=(const WatermarkCuda& other)
{
	if (this != &other) 
	{
		copyParams(other);
		randomMatrix = other.randomMatrix;
	}
	return *this;
}

af::array WatermarkCuda::computeCustomMask(const af::array& inputImage) const
{
	//transposed grid dimensions because of column-major order in arrayfire
	const dim3 gridSize = cuda_utils::gridSizeCalculate(windowBlockSize, baseCols, baseRows);
	const af::array customMask(baseRows, baseCols);
	//call NVF kernel
	nvf<3> <<<gridSize, windowBlockSize, 0, afStream>>> (inputImage.device<float>(), customMask.device<float>(), baseCols, baseRows);
	//transfer ownership to arrayfire and return output array
	unlockArrays(inputImage, customMask);
	return customMask;
}

af::array WatermarkCuda::computeErrorSequence(const af::array& image, const af::array& coefficients) const
{
	//transposed grid dimensions because of column-major order in arrayfire
	const dim3 gridSize = cuda_utils::gridSizeCalculate(windowBlockSize, baseCols, baseRows);
	const af::array neighbors(baseRows, baseCols);
	//populate constant memory and call scaled neighbors kernel
	setCoeffs(coefficients.device<float>());
	calculate_error_sequence_p3 << <gridSize, windowBlockSize, 0, afStream >> > (image.device<float>(), neighbors.device<float>(), baseCols, baseRows);
	//transfer ownership to arrayfire and return output array
	unlockArrays(image, neighbors, coefficients);
	return neighbors;
}

void WatermarkCuda::computePredictionErrorData(const af::array& image, af::array& errorSequence, af::array& coefficients) const
{
	const dim3 gridSize = cuda_utils::gridSizeCalculate(meBlockSize, meKernelDims.y, meKernelDims.x);
	//call prediction error mask kernel
	const af::array RxPartial(baseRows, meKernelDims.x);
	const af::array rxPartial(baseRows, meKernelDims.x / 8);
	me_p3 <<<gridSize, meBlockSize, 0, afStream>>> (image.device<float>(), RxPartial.device<float>(), rxPartial.device<float>(), baseCols, meKernelDims.x, baseRows);
	unlockArrays(image, RxPartial, rxPartial);
	//calculation of coefficients and error sequence
	const auto correlationArrays = transformCorrelationArrays(RxPartial, rxPartial);
	coefficients = af::solve(correlationArrays.first, correlationArrays.second);
	//if system is not solvable, don't waste computing the error sequence, there is no watermark to embed
	if (af::anyTrue<bool>(af::isNaN(coefficients))) 
	{
		coefficients = af::array(0, f32);
		return;
	}
	//call scaled neighbors kernel and compute error sequence
	errorSequence = computeErrorSequence(image, coefficients);
}

float WatermarkCuda::computeCorrelation(const af::array& e_u, const af::array& e_z) const
{
	const int N = static_cast<int>(e_u.elements());
	const int blocks = (N + corrPartialBlockSize - 1) / corrPartialBlockSize;
	const af::array dotPartial(blocks);
	const af::array uNormPartial(blocks);
	const af::array zNormPartial(blocks);
	const af::array correlationResult(1);
	float* dotPartialPtr = dotPartial.device<float>();
	float* uNormPartialPtr = uNormPartial.device<float>();
	float* zNormPartialPtr = zNormPartial.device<float>();

	//calculate partial dot products and norms
	calculate_partial_correlation <<<blocks, corrPartialBlockSize, 0, afStream >>> (e_u.device<float>(), e_z.device<float>(), dotPartialPtr, uNormPartialPtr, zNormPartialPtr, N);
	//reduce partials and compute correlation
	calculate_final_correlation << <1, corrFinalBlockSize, 0, afStream >> > (dotPartialPtr, uNormPartialPtr, zNormPartialPtr, correlationResult.device<float>(), blocks);
	
	unlockArrays(e_u, e_z, dotPartial, uNormPartial, zNormPartial, correlationResult);
	float correlation = correlationResult.scalar<float>();
	return correlation;

}