#pragma once
#include "opencl_init.h"
#include "opencl_utils.hpp"
#include "WatermarkBase.hpp"
#include "WatermarkGpu.hpp"
#include <af/opencl.h>
#include <algorithm>
#include <arrayfire.h>
#include <memory>
#include <stdexcept>
#include <string>

struct dim2
{
	dim_t rows;
	dim_t cols;
};

/*!
 *  \brief  Functions for watermark computation and detection, OpenCL implementation.
 *  \author Dimitris Karatzas
 */
template<int p>
class WatermarkOCL final : public WatermarkGPU<p>
{
private:
	using WatermarkBase::align;
	using clMemPtr = std::unique_ptr<cl_mem>;

public:
	WatermarkOCL<p>(const unsigned int rows, const unsigned int cols, const std::string& randomMatrixPath, const float psnr)
		: WatermarkGPU<p>(rows, cols, randomMatrixPath, psnr), texKernelDims({ WatermarkBase::align<16>(rows), WatermarkBase::align<16>(cols) }), meKernelDims({ rows, WatermarkBase::align<64>(cols) }),
		programs(cl_utils::buildKernels(p))
	{ }

private:
	static constexpr int RxMappings[64]
	{
		0,  1,  2,  3,  4,  5,  6,  7,
		1,  8,  9,  10, 11, 12, 13, 14,
		2,  9,  15, 16, 17, 18, 19, 20,
		3,  10, 16, 21, 22, 23, 24, 25,
		4,  11, 17, 22, 26, 27, 28, 29,
		5,  12, 18, 23, 27, 30, 31, 32,
		6,  13, 19, 24, 28, 31, 33, 34,
		7,  14, 20, 25, 29, 32, 34, 35
	};
	static constexpr unsigned int corrPartialBlockSize = 256;

	cl::Context context{ afcl::getContext(true) };
	cl::CommandQueue queue{ afcl::getQueue(true) };
	cl::Device device{ afcl::getDeviceId(), true };
	cl::Buffer RxMappingsBuff{ context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * 64, (void*)RxMappings, NULL };
	dim2 texKernelDims, meKernelDims;
	unsigned int corrFinalLocalSize = std::min(1024, static_cast<int>(device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>()));
	cl::Program programs;

	inline cl::Buffer wrap(const cl_mem* mem) const { return cl::Buffer(*mem, true); }
	af::array computeCustomMask(const af::array& image) const
	{
		const af::array customMask(this->baseRows, this->baseCols);
		const clMemPtr imageMem(image.device<cl_mem>());
		const clMemPtr outputMem(customMask.device<cl_mem>());
		//transposed global dimensions because of column-major order in arrayfire
		executeKernel([&]() {
			queue.enqueueNDRangeKernel(
				cl_utils::KernelBuilder(programs, "nvf").args(wrap(imageMem.get()), wrap(outputMem.get()), this->baseCols, this->baseRows).build(),
				cl::NDRange(), cl::NDRange(texKernelDims.rows, texKernelDims.cols), cl::NDRange(16, 16));
			queue.finish();
			this->unlockArrays(image, customMask);
		}, "nvf");
		return customMask;
	}
	af::array computeErrorSequence(const af::array& image, const af::array& coefficients, const bool calculateAbs) const
	{
		const af::array errorSequence(this->baseRows, this->baseCols);
		const clMemPtr imageMem(image.device<cl_mem>());
		const clMemPtr coeffsMem(coefficients.device<cl_mem>());
		const clMemPtr errorSequenceMem(errorSequence.device<cl_mem>());
		//transposed global dimensions because of column-major order in arrayfire
		executeKernel([&]() {
			queue.enqueueNDRangeKernel(
				cl_utils::KernelBuilder(programs, "error_sequence_p3").args(wrap(imageMem.get()), wrap(errorSequenceMem.get()), wrap(coeffsMem.get()), this->baseCols, this->baseRows, (int)calculateAbs).build(),
				cl::NDRange(), cl::NDRange(texKernelDims.rows, texKernelDims.cols), cl::NDRange(16, 16));
			queue.finish();
			this->unlockArrays(image, coefficients, errorSequence);
		}, "error_sequence_p3");
		return errorSequence;
	}
	void computePredictionErrorData(const af::array& image, af::array& errorSequence, af::array& coefficients, const bool calculateAbs) const
	{
		const af::array RxPartial(this->baseRows, meKernelDims.cols);
		const af::array rxPartial(this->baseRows, meKernelDims.cols / 8);
		const clMemPtr imageMem(image.device<cl_mem>());
		const clMemPtr RxPartialMem(RxPartial.device<cl_mem>());
		const clMemPtr rxPartialMem(rxPartial.device<cl_mem>());
		executeKernel([&]() {
			queue.enqueueNDRangeKernel(
				cl_utils::KernelBuilder(programs, "me").args(wrap(imageMem.get()), wrap(RxPartialMem.get()), wrap(rxPartialMem.get()), RxMappingsBuff, this->baseCols, this->baseRows).build(),
				cl::NDRange(), cl::NDRange(meKernelDims.cols, meKernelDims.rows), cl::NDRange(64, 1));
			//finish and return memory to arrayfire
			queue.finish();
			this->unlockArrays(image, RxPartial, rxPartial);
			//calculation of coefficients and error sequence
			const auto correlationArrays = this->transformCorrelationArrays(RxPartial, rxPartial);
			//solve() may crash in OpenCL ArrayFire implementation if the system is not solvable.
			try {
				coefficients = af::solve(correlationArrays.first, correlationArrays.second);
			}
			catch (const af::exception&) {
				coefficients = af::array(0, f32);
				return;
			}
			errorSequence = computeErrorSequence(image, coefficients, calculateAbs);
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
				cl_utils::KernelBuilder(programs, "calculate_partial_correlation").args(
					wrap(euMem.get()), wrap(ezMem.get()), wrap(dotPartialMem.get()), wrap(uNormPartialMem.get()), wrap(zNormPartialMem.get()), N).build(),
				cl::NDRange(), cl::NDRange(globalSizePartials), cl::NDRange(corrPartialBlockSize));
			queue.finish();
			//reduce partials and compute correlation
			queue.enqueueNDRangeKernel(
				cl_utils::KernelBuilder(programs, "calculate_final_correlation").args(
					wrap(dotPartialMem.get()), wrap(uNormPartialMem.get()), wrap(zNormPartialMem.get()), wrap(correlationResultMem.get()), blocks).build(),
				cl::NDRange(), cl::NDRange(corrFinalLocalSize), cl::NDRange(corrFinalLocalSize));
			queue.finish();
			//retrieve the correlation result
			this->unlockArrays(e_u, e_z, dotPartial, uNormPartial, zNormPartial, correlationResult);
			correlation = correlationResult.scalar<float>();
		}, "compute correlation kernels");
		return correlation;
	}

	template<typename Func>
	void executeKernel(const Func& kernelFunc, const std::string& context) const
	{
		try {
			kernelFunc();
		}
		catch (const cl::Error& ex) {
			throw std::runtime_error("OpenCL Error in " + context + ": " + std::string(ex.what()) + " Error code: " + std::to_string(ex.err()) + "\n");
		}
	}
};