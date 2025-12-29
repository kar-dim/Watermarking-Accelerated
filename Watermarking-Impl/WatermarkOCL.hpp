#pragma once
#include "opencl_init.h"
#include "opencl_utils.hpp"
#include "WatermarkBase.hpp"
#include "WatermarkGpu.hpp"
#include <af/opencl.h>
#include <arrayfire.h>
#include <memory>
#include <string>

struct dim2
{
	dim_t rows;
	dim_t cols;
};

using namespace cl_utils;

/*!
 *  \brief  Functions for watermark computation and detection, OpenCL implementation.
 *  \author Dimitris Karatzas
 */
template<int p>
class WatermarkOCL final : public WatermarkGPU<p>
{
public:
	WatermarkOCL<p>(const unsigned int rows, const unsigned int cols, const std::string& randomMatrixPath, const float psnr)
		: WatermarkGPU<p>(rows, cols, randomMatrixPath, psnr),
		texKernelDims{ align<windowBlockSize>(rows), align<windowBlockSize>(cols) }, meKernelDims{ rows, align<meBlockSize>(cols) },
		programs(buildKernels(p))
	{ }

private:
	using WatermarkBase::align;
	using clMemPtr = std::unique_ptr<cl_mem>;

	static constexpr unsigned int corrPartialBlockSize = 256;
	static constexpr unsigned int windowBlockSize = 16;
	static constexpr unsigned int meBlockSize = 64;

	cl::Context context{ afcl::getContext(true) };
	cl::CommandQueue queue{ afcl::getQueue(true) };
	cl::Device device{ afcl::getDeviceId(), true };
	dim2 texKernelDims, meKernelDims;
	unsigned int corrFinalLocalSize = maxPow2WorkGroupSize(device);
	cl::Program programs;

	af::array computeCustomMask(const af::array& image) const
	{
		const af::array customMask(this->baseRows, this->baseCols);
		const clMemPtr imageMem(image.device<cl_mem>());
		const clMemPtr outputMem(customMask.device<cl_mem>());
		//transposed global dimensions because of column-major order in arrayfire
		executeKernel([&]() {
			queue.enqueueNDRangeKernel(
				KernelBuilder(programs, "nvf").args(wrap(imageMem.get()), wrap(outputMem.get()), this->baseCols, this->baseRows).build(),
				cl::NDRange(), cl::NDRange(texKernelDims.rows, texKernelDims.cols), cl::NDRange(windowBlockSize, windowBlockSize));
			this->unlockArrays(image, customMask);
		}, "nvf");
		return customMask;
	}

	af::array computeErrorSequence(const af::array& image, const bool calculateAbs) const
	{
		const af::array errorSequence(this->baseRows, this->baseCols);
		const clMemPtr imageMem(image.device<cl_mem>());
		const clMemPtr coeffsMem(this->coefficients.template device<cl_mem>());
		const clMemPtr errorSequenceMem(errorSequence.device<cl_mem>());
		const clMemPtr stopFlagMem(this->stopFlag.template device<cl_mem>());
		//transposed global dimensions because of column-major order in arrayfire
		executeKernel([&]() {
			queue.enqueueNDRangeKernel(
				KernelBuilder(programs, "error_sequence_p3").args(wrap(imageMem.get()), wrap(errorSequenceMem.get()), wrap(coeffsMem.get()), this->baseCols, this->baseRows, (int)calculateAbs, wrap(stopFlagMem.get())).build(),
				cl::NDRange(), cl::NDRange(texKernelDims.rows, texKernelDims.cols), cl::NDRange(windowBlockSize, windowBlockSize));
			this->unlockArrays(image, errorSequence, this->coefficients, this->stopFlag);
		}, "error_sequence_p3");
		return errorSequence;
	}

	af::array computePredictionErrorData(const af::array& image, const bool calculateAbs) const
	{
		const af::array RxPartial(this->baseRows, (meKernelDims.cols / 64) * 36);
		const af::array rxPartial(this->baseRows, meKernelDims.cols / 8);
		const clMemPtr imageMem(image.device<cl_mem>());
		const clMemPtr RxPartialMem(RxPartial.device<cl_mem>());
		const clMemPtr rxPartialMem(rxPartial.device<cl_mem>());
		return executeKernel([&]() -> af::array {
			queue.enqueueNDRangeKernel(
				KernelBuilder(programs, "me").args(wrap(imageMem.get()), wrap(RxPartialMem.get()), wrap(rxPartialMem.get()), this->baseCols, this->baseRows).build(),
				cl::NDRange(), cl::NDRange(meKernelDims.cols, meKernelDims.rows), cl::NDRange(meBlockSize, 1));
			//return memory to arrayfire
			this->unlockArrays(image, RxPartial, rxPartial);
			
			//calculation of coefficients and error sequence
			const auto correlationArrays = this->transformCorrelationArrays(RxPartial, rxPartial);
			const af::array& Rx = correlationArrays.first;
			const af::array& rx = correlationArrays.second;
			const clMemPtr RxMemPtr(Rx.device<cl_mem>());
			const clMemPtr rxMemPtr(rx.device<cl_mem>());
			const clMemPtr coeffsMem(this->coefficients.template device<cl_mem>());
			const clMemPtr stopFlagMem(this->stopFlag.template device<cl_mem>());
			queue.enqueueNDRangeKernel(
				KernelBuilder(programs, "cholesky_solver_p3").args(wrap(RxMemPtr.get()), wrap(rxMemPtr.get()), wrap(coeffsMem.get()), wrap(stopFlagMem.get())).build(),
				cl::NDRange(), cl::NDRange(1), cl::NDRange(1));
			//return memory to arrayfire
			this->unlockArrays(Rx, rx, this->coefficients, this->stopFlag);
			return computeErrorSequence(image, calculateAbs);
		}, "me");
	}

	float computeCorrelation(const af::array& e_u, const af::array& e_z) const
	{
		const int N = static_cast<int>(e_u.elements());
		const int globalSizePartials = align<corrPartialBlockSize>(N);
		const int blocks = globalSizePartials / corrPartialBlockSize;
		const af::array dotPartial(blocks);
		const af::array uNormPartial(blocks);
		const af::array zNormPartial(blocks);
		const af::array correlationResult(1);
		const clMemPtr euMem(e_u.device<cl_mem>());
		const clMemPtr ezMem(e_z.device<cl_mem>());
		const clMemPtr dotPartialMem(dotPartial.device<cl_mem>());
		const clMemPtr uNormPartialMem(uNormPartial.device<cl_mem>());
		const clMemPtr zNormPartialMem(zNormPartial.device<cl_mem>());
		const clMemPtr correlationResultMem(correlationResult.device<cl_mem>());
		float correlation = 0.0f;
		executeKernel([&]() {
			//calculate partial dot products and norms
			queue.enqueueNDRangeKernel(
				KernelBuilder(programs, "calculate_partial_correlation").args(
					wrap(euMem.get()), wrap(ezMem.get()), wrap(dotPartialMem.get()), wrap(uNormPartialMem.get()), wrap(zNormPartialMem.get()), N).build(),
				cl::NDRange(), cl::NDRange(globalSizePartials), cl::NDRange(corrPartialBlockSize));
			//reduce partials and compute correlation
			queue.enqueueNDRangeKernel(
				KernelBuilder(programs, "calculate_final_correlation").args(
					wrap(dotPartialMem.get()), wrap(uNormPartialMem.get()), wrap(zNormPartialMem.get()), wrap(correlationResultMem.get()), blocks).build(),
				cl::NDRange(), cl::NDRange(corrFinalLocalSize), cl::NDRange(corrFinalLocalSize));
			//retrieve the correlation result
			this->unlockArrays(e_u, e_z, dotPartial, uNormPartial, zNormPartial, correlationResult);
			correlation = correlationResult.scalar<float>();
		}, "compute correlation kernels");
		return correlation;
	}
};