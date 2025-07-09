#pragma once
#include "opencl_init.h"
#include "WatermarkGpu.hpp"
#include <af/opencl.h>
#include <arrayfire.h>
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
class WatermarkOCL final : public WatermarkGPU 
{
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
	const cl::Context context{ afcl::getContext(false) };
	const cl::CommandQueue queue{ afcl::getQueue(false) };
	const cl::Buffer RxMappingsBuff{ context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * 64, (void*)RxMappings, NULL };
	dim2 texKernelDims, meKernelDims;
	cl::Program programs;

	af::array computeCustomMask(const af::array& image) const override;
	af::array computeScaledNeighbors(const af::array& image, const af::array& coefficients) const override;
	void computePredictionErrorData(const af::array& image, af::array& errorSequence, af::array& coefficients) const override;
	float computeCorrelation(const af::array& e_u, const af::array& e_z) const override;

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

public:
	WatermarkOCL(const unsigned int rows, const unsigned int cols, const std::string& randomMatrixPath, const int p, const float psnr);
	WatermarkOCL(const WatermarkOCL& other);
	WatermarkOCL(WatermarkOCL&& other) noexcept = delete;
	WatermarkOCL& operator=(WatermarkOCL&& other) noexcept = delete;
	WatermarkOCL& operator=(const WatermarkOCL& other);
};