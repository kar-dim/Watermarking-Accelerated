#pragma once
#include "cuda_utils.hpp"
#include "CudaStreamManager.hpp"
#include "kernels/kernels.cuh"
#include "WatermarkBase.hpp"
#include "WatermarkGpu.hpp"
#include <arrayfire.h>
#include <cuda_runtime.h>
#include <string>

/*!
 *  \brief  Functions for watermark computation and detection, CUDA implementation.
 *  \author Dimitris Karatzas
 */
template<int p>
class WatermarkCuda final : public WatermarkGPU<p>
{
public:
	WatermarkCuda<p>(const unsigned int rows, const unsigned int cols, const std::string& randomMatrixPath, const float psnr)
		: WatermarkGPU<p>(rows, cols, randomMatrixPath, psnr), meKernelDims{ WatermarkBase::align<meBlockSize.x>(cols), rows }, afStream(CudaStreamManager::getInstance().getAfStream())
	{ }

private:
	static constexpr dim3 windowBlockSize{ 16, 16 }, meBlockSize{ 256, 1 };
	static constexpr unsigned int corrPartialBlockSize = 256, corrFinalBlockSize = 1024;
	dim3 meKernelDims;
	cudaStream_t afStream;

	af::array computeCustomMask(const af::array& inputImage) const
	{
		//transposed grid dimensions because of column-major order in arrayfire
		const dim3 gridSize = cuda_utils::gridSizeCalculate(windowBlockSize, this->baseCols, this->baseRows);
		const af::array customMask(this->baseRows, this->baseCols);
		//call NVF kernel
		nvf<3> << <gridSize, windowBlockSize, 0, afStream >> > (inputImage.device<float>(), customMask.device<float>(), this->baseCols, this->baseRows);
		//transfer ownership to arrayfire and return output array
		this->unlockArrays(inputImage, customMask);
		return customMask;
	}

	af::array computeErrorSequence(const af::array& image, const bool calculateAbs) const
	{
		//transposed grid dimensions because of column-major order in arrayfire
		const dim3 gridSize = cuda_utils::gridSizeCalculate(windowBlockSize, this->baseCols, this->baseRows);
		const af::array errorSequence(this->baseRows, this->baseCols);
		//call error sequence kernel
		calculate_error_sequence_p3 << <gridSize, windowBlockSize, 0, afStream >> > (image.device<float>(), errorSequence.device<float>(), this->coefficients.template device<float>(), this->baseCols, this->baseRows, calculateAbs, this->stopFlag.template device<int>());
		//transfer ownership to arrayfire and return output array
		this->unlockArrays(image, errorSequence, this->coefficients, this->stopFlag);
		return errorSequence;
	}

	af::array computePredictionErrorData(const af::array& image, const bool calculateAbs) const
	{
		const dim3 gridSize = cuda_utils::gridSizeCalculate(meBlockSize, meKernelDims.y, meKernelDims.x);
		//call prediction error mask kernel
		const af::array RxPartial(this->baseRows, meKernelDims.x / 4);
		const af::array rxPartial(this->baseRows, meKernelDims.x / 32);
		me_p3 << <gridSize, meBlockSize, 0, afStream >> > (image.device<float>(), RxPartial.device<float>(), rxPartial.device<float>(), this->baseCols, this->baseRows);
		this->unlockArrays(image, RxPartial, rxPartial);
		//calculation of coefficients and error sequence
		const auto correlationArrays = this->transformCorrelationArrays(RxPartial, rxPartial);
		const af::array& Rx = correlationArrays.first;
		const af::array& rx = correlationArrays.second;
		cholesky_solver_p3 << <1, 1, 0, afStream >> > (Rx.device<float>(), rx.device<float>(), this->coefficients.template device<float>(), this->stopFlag.template device<int>());
		this->unlockArrays(Rx, rx, this->coefficients, this->stopFlag);
		return computeErrorSequence(image, calculateAbs);
	}

	float computeCorrelation(const af::array& e_u, const af::array& e_z) const
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
		calculate_partial_correlation << <blocks, corrPartialBlockSize, 0, afStream >> > (e_u.device<float>(), e_z.device<float>(), dotPartialPtr, uNormPartialPtr, zNormPartialPtr, N);
		//reduce partials and compute correlation
		calculate_final_correlation << <1, corrFinalBlockSize, 0, afStream >> > (dotPartialPtr, uNormPartialPtr, zNormPartialPtr, correlationResult.device<float>(), blocks);
		//transfer ownership to arrayfire and return output correlation scalar to host
		this->unlockArrays(e_u, e_z, dotPartial, uNormPartial, zNormPartial, correlationResult);
		return correlationResult.scalar<float>();
	}
};