#include "buffer.hpp"
#include "opencl_init.h"
#include "opencl_utils.hpp"
#include "WatermarkBase.hpp"
#include "WatermarkOCL.hpp"
#include <arrayfire.h>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using std::string;

//initialize data and memory
WatermarkOCL::WatermarkOCL(const unsigned int rows, const unsigned int cols, const string& randomMatrixPath, const int p, const float psnr)
	: WatermarkBase(rows, cols, randomMatrixPath, p, (255.0f / sqrt(pow(10.0f, psnr / 10.0f)))), texKernelDims({ ALIGN(rows, 16), ALIGN(cols, 16) }), meKernelDims({ rows, ALIGN(cols, 64) })
{
	if (p != 3 && p != 5 && p != 7 && p != 9)
		throw std::runtime_error(string("Wrong p parameter: ") + std::to_string(p) + "!\n");
	//compile opencl kernels and initialize memory
	cl_utils::buildKernels(programs, p);
	initializeGpuMemory();
}

//copy constructor
WatermarkOCL::WatermarkOCL(const WatermarkOCL& other) : WatermarkBase(other.baseRows, other.baseCols, other.randomMatrix, other.p, other.strengthFactor),
	texKernelDims(other.texKernelDims), meKernelDims(other.meKernelDims), programs(other.programs)
{
	initializeGpuMemory();
}

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
		initializeGpuMemory();
	}
	return *this;
}

void WatermarkOCL::initializeGpuMemory()
{
	//initialize texture (transposed dimensions, arrayfire is column wise, we skip an extra transpose)
	image2d = cl::Image2D(context, CL_MEM_READ_ONLY, cl::ImageFormat(CL_LUMINANCE, CL_FLOAT), baseRows, baseCols, 0, NULL);
}

void WatermarkOCL::copyDataToTexture(const af::array& image) const
{
	const std::unique_ptr<cl_mem> imageMem(image.device<cl_mem>());
	cl_utils::copyBufferToImage(queue, image2d, imageMem.get(), baseCols, baseRows);
	unlockArrays(image);
}

void WatermarkOCL::onReinitialize()
{
	texKernelDims = { ALIGN(baseRows, 16), ALIGN(baseCols, 16) };
	meKernelDims = { baseRows, ALIGN(baseCols, 64) };
	initializeGpuMemory();
}

af::array WatermarkOCL::computeCustomMask() const
{
	const af::array customMask(baseRows, baseCols);
	const std::unique_ptr<cl_mem> outputMem(customMask.device<cl_mem>());
	const int localMemElements = (16 + p) * (16 + p);
	//execute kernel
	try {
		cl::Buffer buff(*outputMem.get(), true);
		queue.enqueueNDRangeKernel(
			cl_utils::KernelBuilder(programs[0],"nvf").args(image2d, buff, cl::Local(sizeof(float) * localMemElements)).build(),
			cl::NDRange(), cl::NDRange(texKernelDims.rows, texKernelDims.cols), cl::NDRange(16, 16));
		queue.finish();
		unlockArrays(customMask);
		return customMask;
	}
	catch (const cl::Error& ex) {
		throw std::runtime_error("ERROR in nvf: " + string(ex.what()) + " Error code: " + std::to_string(ex.err()) + "\n");
	}
}

af::array WatermarkOCL::computeScaledNeighbors(const af::array& coefficients) const
{
	const af::array neighbors(baseRows, baseCols);
	const std::unique_ptr<cl_mem> coeffsMem(coefficients.device<cl_mem>());
	const std::unique_ptr<cl_mem> neighborsMem(neighbors.device<cl_mem>());
	//execute kernel
	try {
		cl::Buffer neighborsBuff(*neighborsMem.get(), true);
		cl::Buffer coeffsBuff(*coeffsMem.get(), true);
		queue.enqueueNDRangeKernel(
			cl_utils::KernelBuilder(programs[2], "scaled_neighbors_p3").args(image2d, neighborsBuff, coeffsBuff, cl::Local(sizeof(float) * 324)).build(),
			cl::NDRange(), cl::NDRange(texKernelDims.rows, texKernelDims.cols), cl::NDRange(16, 16));
		queue.finish();
		unlockArrays(coefficients, neighbors);
		return neighbors;
	}
	catch (const cl::Error& ex) {
		throw std::runtime_error("ERROR in scaled_neighbors_p3: " + string(ex.what()) + " Error code: " + std::to_string(ex.err()) + "\n");
	}
}

BufferType WatermarkOCL::makeWatermark(const BufferType& inputImage, const BufferType& outputImage, float& watermarkStrength, MASK_TYPE maskType)
{
	return makeWatermarkGpu(inputImage, outputImage, randomMatrix, strengthFactor, watermarkStrength, maskType);
}

af::array WatermarkOCL::computePredictionErrorMask(const af::array& image, af::array& errorSequence, af::array& coefficients, const bool maskNeeded) const
{
	const af::array RxPartial(baseRows, meKernelDims.cols);
	const af::array rxPartial(baseRows, meKernelDims.cols / 8);
	const std::unique_ptr<cl_mem> RxPartialMem(RxPartial.device<cl_mem>());
	const std::unique_ptr<cl_mem> rxPartialMem(rxPartial.device<cl_mem>());
	try {
		//initialize custom kernel memory
		cl::Buffer Rx_buff(*RxPartialMem.get(), true);
		cl::Buffer rx_buff(*rxPartialMem.get(), true);
		//call prediction error mask kernel
		queue.enqueueNDRangeKernel(
			cl_utils::KernelBuilder(programs[1], "me").args(image2d, Rx_buff, rx_buff, RxMappingsBuff,
			cl::Local(sizeof(cl_half) * 2304)).build(),
			cl::NDRange(), cl::NDRange(meKernelDims.cols, meKernelDims.rows), cl::NDRange(64, 1));
		//finish and return memory to arrayfire
		queue.finish();
	}
	catch (const cl::Error& ex) {
		throw std::runtime_error(string("ERROR in compute_me_mask(): " + string(ex.what()) + " Error code: " + std::to_string(ex.err()) + "\n"));
	}

	unlockArrays(RxPartial, rxPartial);
	//calculation of coefficients, error sequence and mask
	const auto correlationArrays = transformCorrelationArrays(RxPartial, rxPartial, p);
	//solve() may crash in OpenCL ArrayFire implementation if the system is not solvable.
	try {
		coefficients = af::solve(correlationArrays.first, correlationArrays.second);
	}
	catch (const af::exception&) {
		coefficients = af::array(0, f32);
		return af::array();
	}
	//call scaled neighbors kernel and compute error sequence
	errorSequence = image - computeScaledNeighbors(coefficients);
	if (maskNeeded) 
	{
		const af::array errorSequenceAbs = af::abs(errorSequence);
		return errorSequenceAbs / af::max<float>(errorSequenceAbs);
	}
	return af::array();
}

float WatermarkOCL::detectWatermark(const BufferType& inputImage, MASK_TYPE maskType)
{
	return detectWatermarkGpu(inputImage, randomMatrix, maskType);
}