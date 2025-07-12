#pragma once
#include "WatermarkGpu.hpp"
#include <arrayfire.h>
#include <cuda_runtime.h>
#include <string>

/*!
 *  \brief  Functions for watermark computation and detection, CUDA implementation.
 *  \author Dimitris Karatzas
 */
class WatermarkCuda final : public WatermarkGPU
{
private:
	static constexpr dim3 windowBlockSize{ 16, 16 }, meBlockSize{ 64, 1 }, corrBlockSize{ 768, 1 };
	static constexpr unsigned int corrPartialBlockSize = 256, corrFinalBlockSize = 1024;
	dim3 meKernelDims;
	static cudaStream_t afStream;

	af::array computeCustomMask(const af::array& image) const override;
	af::array computeErrorSequence(const af::array& image, const af::array& coefficients) const override;
	void computePredictionErrorData(const af::array& image, af::array& errorSequence, af::array& coefficients) const override;
	float computeCorrelation(const af::array& e_u, const af::array& e_z) const override;
	void copyParams(const WatermarkCuda& other) noexcept;

public:
	WatermarkCuda(const unsigned int rows, const unsigned int cols, const std::string& randomMatrixPath, const int p, const float psnr);
	WatermarkCuda(const WatermarkCuda& other);
	WatermarkCuda(WatermarkCuda&& other) noexcept;
	WatermarkCuda& operator=(WatermarkCuda&& other) noexcept;
	WatermarkCuda& operator=(const WatermarkCuda& other);
	~WatermarkCuda() override = default;
};