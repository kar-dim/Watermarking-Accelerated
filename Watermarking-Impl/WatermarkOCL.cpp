#include "opencl_init.h"
#include "opencl_utils.hpp"
#include "WatermarkBase.hpp"
#include "WatermarkGpu.hpp"
#include "WatermarkOCL.hpp"
#include <arrayfire.h>
#include <memory>
#include <string>
#include <utility>

using std::string;

//initialize data and memory
WatermarkOCL::WatermarkOCL(const unsigned int rows, const unsigned int cols, const string& randomMatrixPath, const int p, const float psnr)
	: WatermarkGPU(rows, cols, randomMatrixPath, psnr, p), texKernelDims({ align<16>(rows), align<16>(cols) }), meKernelDims({ rows, align<64>(cols) }),
	  programs(cl_utils::buildKernels(p))
{ }

//copy constructor
WatermarkOCL::WatermarkOCL(const WatermarkOCL& other) : WatermarkGPU(other.baseRows, other.baseCols, other.randomMatrix, other.strengthFactor, other.p),
	texKernelDims(other.texKernelDims), meKernelDims(other.meKernelDims), programs(other.programs)
{ }

//copy assignment operator
WatermarkOCL& WatermarkOCL::operator=(const WatermarkOCL& other)
{
	if (this != &other) 
	{
		baseRows = other.baseRows;
		baseCols = other.baseCols;
		randomMatrix = other.randomMatrix;
		texKernelDims = other.texKernelDims;
		meKernelDims = other.meKernelDims;
		programs = other.programs;
		p = other.p;
		strengthFactor = other.strengthFactor;
	}
	return *this;
}

af::array WatermarkOCL::computeCustomMask(const af::array& image) const
{
	const af::array customMask(baseRows, baseCols);
	const std::unique_ptr<cl_mem> imageMem(image.device<cl_mem>());
	const std::unique_ptr<cl_mem> outputMem(customMask.device<cl_mem>());
	//transposed global dimensions because of column-major order in arrayfire
	executeKernel([&]() {
		cl::Buffer imageBuff(*imageMem.get(), true);
		cl::Buffer outputBuff(*outputMem.get(), true);
		queue.enqueueNDRangeKernel(
			cl_utils::KernelBuilder(programs, "nvf").args(imageBuff, outputBuff, baseCols, baseRows).build(),
			cl::NDRange(), cl::NDRange(texKernelDims.rows, texKernelDims.cols), cl::NDRange(16, 16));
		queue.finish();
		unlockArrays(image, customMask);
	}, "nvf");
	return customMask;
}

af::array WatermarkOCL::computeErrorSequence(const af::array& image, const af::array& coefficients) const
{
	const af::array neighbors(baseRows, baseCols);
	const std::unique_ptr<cl_mem> imageMem(image.device<cl_mem>());
	const std::unique_ptr<cl_mem> coeffsMem(coefficients.device<cl_mem>());
	const std::unique_ptr<cl_mem> neighborsMem(neighbors.device<cl_mem>());
	//transposed global dimensions because of column-major order in arrayfire
	executeKernel([&]() {
		cl::Buffer imageBuff(*imageMem.get(), true);
		cl::Buffer neighborsBuff(*neighborsMem.get(), true);
		cl::Buffer coeffsBuff(*coeffsMem.get(), true);
		queue.enqueueNDRangeKernel(
			cl_utils::KernelBuilder(programs, "error_sequence_p3").args(imageBuff, neighborsBuff, coeffsBuff, baseCols, baseRows).build(),
			cl::NDRange(), cl::NDRange(texKernelDims.rows, texKernelDims.cols), cl::NDRange(16, 16));
		queue.finish();
		unlockArrays(image, coefficients, neighbors);
	}, "error_sequence_p3");
	return neighbors;
}

void WatermarkOCL::computePredictionErrorData(const af::array& image, af::array& errorSequence, af::array& coefficients) const
{
	const af::array RxPartial(baseRows, meKernelDims.cols);
	const af::array rxPartial(baseRows, meKernelDims.cols / 8);
	const std::unique_ptr<cl_mem> imageMem(image.device<cl_mem>());
	const std::unique_ptr<cl_mem> RxPartialMem(RxPartial.device<cl_mem>());
	const std::unique_ptr<cl_mem> rxPartialMem(rxPartial.device<cl_mem>());
	executeKernel([&]() {
		cl::Buffer imageBuff(*imageMem.get(), true);
		cl::Buffer Rx_buff(*RxPartialMem.get(), true);
		cl::Buffer rx_buff(*rxPartialMem.get(), true);
		queue.enqueueNDRangeKernel(
			cl_utils::KernelBuilder(programs, "me").args(imageBuff, Rx_buff, rx_buff, RxMappingsBuff, baseCols, static_cast<unsigned int>(meKernelDims.cols), baseRows).build(),
			cl::NDRange(), cl::NDRange(meKernelDims.cols, meKernelDims.rows), cl::NDRange(64, 1));
		//finish and return memory to arrayfire
		queue.finish();
		unlockArrays(image, RxPartial, rxPartial);
		//calculation of coefficients and error sequence
		const auto correlationArrays = transformCorrelationArrays(RxPartial, rxPartial);
		//solve() may crash in OpenCL ArrayFire implementation if the system is not solvable.
		try {
			coefficients = af::solve(correlationArrays.first, correlationArrays.second);
		}
		catch (const af::exception&) {
			coefficients = af::array(0, f32);
			return;
		}
		errorSequence = computeErrorSequence(image, coefficients);
	}, "me");
}

float WatermarkOCL::computeCorrelation(const af::array& e_u, const af::array& e_z) const
{
	const int N = static_cast<int>(e_u.elements());
	const int globalSize = align<256>(N);
	const int blocks = globalSize / 256;
	const af::array dotPartial(blocks);
	const af::array uNormPartial(blocks);
	const af::array zNormPartial(blocks);
	const af::array correlationResult(1);
	const std::unique_ptr<cl_mem> euMem(e_u.device<cl_mem>());
	const std::unique_ptr<cl_mem> ezMem(e_z.device<cl_mem>());
	const std::unique_ptr<cl_mem> dotPartialMem(dotPartial.device<cl_mem>());
	const std::unique_ptr<cl_mem> uNormPartialMem(uNormPartial.device<cl_mem>());
	const std::unique_ptr<cl_mem> zNormPartialMem(zNormPartial.device<cl_mem>());
	const std::unique_ptr<cl_mem> correlationResultMem(correlationResult.device<cl_mem>());
	float correlation = 0.0f;
	executeKernel([&]() {
		cl::Buffer euBuff(*euMem.get(), true);
		cl::Buffer ezBuff(*ezMem.get(), true);
		cl::Buffer dotBuff(*dotPartialMem.get(), true);
		cl::Buffer uNormBuff(*uNormPartialMem.get(), true);
		cl::Buffer zNormBuff(*zNormPartialMem.get(), true);
		cl::Buffer corrBuff(*correlationResultMem.get(), true);

		//calculate partial dot products and norms
		queue.enqueueNDRangeKernel(
			cl_utils::KernelBuilder(programs, "calculate_partial_correlation").args(euBuff, ezBuff, dotBuff, uNormBuff, zNormBuff, N).build(),
			cl::NDRange(), cl::NDRange(globalSize), cl::NDRange(256));
		queue.finish();

		//reduce partials and compute correlation
		queue.enqueueNDRangeKernel(
			cl_utils::KernelBuilder(programs, "calculate_final_correlation").args(dotBuff, uNormBuff, zNormBuff, corrBuff, blocks).build(),
			cl::NDRange(), cl::NDRange(1024), cl::NDRange(1024));
		queue.finish();

		unlockArrays(e_u, e_z, dotPartial, uNormPartial, zNormPartial, correlationResult);
		correlation = correlationResult.scalar<float>();
	}, "compute correlation kernels");
	return correlation;
}