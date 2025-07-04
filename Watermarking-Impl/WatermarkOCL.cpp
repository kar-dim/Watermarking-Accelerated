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
	const int localMemBytes = sizeof(float) * (16 + p) * (16 + p);
	//transposed global dimensions because of column-major order in arrayfire
	executeKernel([&]() {
		cl::Buffer imageBuff(*imageMem.get(), true);
		cl::Buffer outputBuff(*outputMem.get(), true);
		queue.enqueueNDRangeKernel(
			cl_utils::KernelBuilder(programs, "nvf").args(imageBuff, outputBuff, baseCols, baseRows, cl::Local(localMemBytes)).build(),
			cl::NDRange(), cl::NDRange(texKernelDims.rows, texKernelDims.cols), cl::NDRange(16, 16));
		queue.finish();
		unlockArrays(image, customMask);
	}, "nvf");
	return customMask;
}

af::array WatermarkOCL::computeScaledNeighbors(const af::array& image, const af::array& coefficients) const
{
	const af::array neighbors(baseRows, baseCols);
	const std::unique_ptr<cl_mem> imageMem(image.device<cl_mem>());
	const std::unique_ptr<cl_mem> coeffsMem(coefficients.device<cl_mem>());
	const std::unique_ptr<cl_mem> neighborsMem(neighbors.device<cl_mem>());
	const int localMemBytes = sizeof(float) * (16 + p) * (16 + p);
	//transposed global dimensions because of column-major order in arrayfire
	executeKernel([&]() {
		cl::Buffer imageBuff(*imageMem.get(), true);
		cl::Buffer neighborsBuff(*neighborsMem.get(), true);
		cl::Buffer coeffsBuff(*coeffsMem.get(), true);
		queue.enqueueNDRangeKernel(
			cl_utils::KernelBuilder(programs, "scaled_neighbors_p3").args(imageBuff, neighborsBuff, coeffsBuff, baseCols, baseRows, cl::Local(localMemBytes)).build(),
			cl::NDRange(), cl::NDRange(texKernelDims.rows, texKernelDims.cols), cl::NDRange(16, 16));
		queue.finish();
		unlockArrays(image, coefficients, neighbors);
	}, "scaled_neighbors_p3");
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
			cl_utils::KernelBuilder(programs, "me").args(imageBuff, Rx_buff, rx_buff, RxMappingsBuff, baseCols, static_cast<unsigned int>(meKernelDims.cols), baseRows,
			cl::Local(sizeof(cl_half) * 2304), cl::Local(sizeof(cl_half) * 198)).build(),
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
		//call scaled neighbors kernel and compute error sequence
		errorSequence = image - computeScaledNeighbors(image, coefficients);
	}, "me");
}