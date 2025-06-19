#pragma once
#include "WatermarkGpu.hpp"
#include <arrayfire.h>
#include <cuda_runtime.h>
#include <string>

class WatermarkCuda : public WatermarkGPU
{
private:
	static constexpr dim3 texKernelBlockSize{ 16, 16 }, meKernelBlockSize{ 64, 1 };
	dim3 meKernelDims;
	cudaTextureObject_t texObj;
	cudaArray* texArray;
	static cudaStream_t afStream;

	void initializeGpuMemory() override;
	af::array computeCustomMask() const override;
	af::array computeScaledNeighbors(const af::array& coefficients) const override;
	void computePredictionErrorData(const af::array& image, af::array& errorSequence, af::array& coefficients) const override;
	void copyDataToTexture(const af::array& image) const override;
	void copyParams(const WatermarkCuda& other) noexcept;

public:
	WatermarkCuda(const unsigned int rows, const unsigned int cols, const std::string& randomMatrixPath, const int p, const float psnr);
	WatermarkCuda(const WatermarkCuda& other);
	WatermarkCuda(WatermarkCuda&& other) noexcept;
	WatermarkCuda& operator=(WatermarkCuda&& other) noexcept;
	WatermarkCuda& operator=(const WatermarkCuda& other);
	~WatermarkCuda();
};