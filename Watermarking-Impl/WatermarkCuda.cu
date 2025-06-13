#include "buffer.hpp"
#include "cuda_utils.hpp"
#include "kernels.cuh"
#include "WatermarkBase.hpp"
#include "WatermarkCuda.cuh"
#include <af/cuda.h>
#include <arrayfire.h>
#include <cmath>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <utility>

using std::string;

cudaStream_t WatermarkCuda::afStream = afcu::getStream(afcu::getNativeId(af::getDevice()));

//initialize data and memory
WatermarkCuda::WatermarkCuda(const unsigned int rows, const unsigned int cols, const string& randomMatrixPath, const int p, const float psnr)
	: WatermarkBase(rows, cols, randomMatrixPath, p, (255.0f / sqrt(pow(10.0f, psnr / 10.0f)))), meKernelDims(ALIGN(cols, 64), rows)
{
	if (p != 3 && p != 5 && p != 7 && p != 9)
		throw std::runtime_error(string("Wrong p parameter: ") + std::to_string(p) + "!\n");
	initializeGpuMemory();
}

//copy constructor
WatermarkCuda::WatermarkCuda(const WatermarkCuda& other) : WatermarkBase(other.baseRows, other.baseCols, other.randomMatrix, other.p, other.strengthFactor),
	meKernelDims(other.meKernelDims)
{
	//we don't need to copy the internal buffers data, only to allocate the correct size based on other
	initializeGpuMemory();
}

//move constructor
WatermarkCuda::WatermarkCuda(WatermarkCuda&& other) noexcept : WatermarkBase(other.baseRows, other.baseCols, std::move(other.randomMatrix), other.p, other.strengthFactor),
	meKernelDims(other.meKernelDims)
{
	static constexpr auto moveMember = [](auto& thisData, auto& otherData, auto value) { thisData = otherData; otherData = value; };
	//move texture data and nullify other
	moveMember(texObj, other.texObj, 0);
	moveMember(texArray, other.texArray, nullptr);
}

//helper method to copy the parameters of another watermark object (for move/copy operators)
void WatermarkCuda::copyParams(const WatermarkCuda& other) noexcept
{
	baseRows = other.baseRows;
	baseCols = other.baseCols;
	meKernelDims = other.meKernelDims;
	p = other.p;
	strengthFactor = other.strengthFactor;
}

void WatermarkCuda::copyDataToTexture(const af::array& image) const
{
	cuda_utils::copyDataToCudaArray(image.device<float>(), baseCols, baseRows, texArray);
	unlockArrays(image);
}

//move assignment operator
WatermarkCuda& WatermarkCuda::operator=(WatermarkCuda&& other) noexcept
{
	static constexpr auto moveAndDestroyMember = [](auto& thisData, auto& otherData, auto& deleter, auto value) { deleter(thisData); thisData = otherData; otherData = value; };
	if (this != &other) 
	{
		copyParams(other);
		//move texture object/array and arrayfire arrays
		moveAndDestroyMember(texObj, other.texObj, cudaDestroyTextureObject, 0);
		moveAndDestroyMember(texArray, other.texArray, cudaFreeArray, nullptr);
		//move arrayfire arrays
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
		cudaDestroyTextureObject(texObj);
		cudaFreeArray(texArray);
		initializeGpuMemory();
		randomMatrix = other.randomMatrix;
	}
	return *this;
}

//destroy texture data (texture object, cuda array) only if they have not been moved
WatermarkCuda::~WatermarkCuda()
{
	static constexpr auto destroy = [](auto&& resource, auto&& deleter) { if (resource) deleter(resource); };
	destroy(texObj, cudaDestroyTextureObject);
	destroy(texArray, cudaFreeArray);
}

void WatermarkCuda::initializeGpuMemory()
{
	//initialize texture (transposed dimensions, arrayfire is column wise, we skip an extra transpose)
	auto textureData = cuda_utils::createTextureData(baseCols, baseRows);
	texObj = textureData.first;
	texArray = textureData.second;
}

void WatermarkCuda::onReinitialize()
{
	meKernelDims = { ALIGN(baseCols, 64), UINT(baseRows) };
	cudaDestroyTextureObject(texObj);
	cudaFreeArray(texArray);
	initializeGpuMemory();
}

af::array WatermarkCuda::computeCustomMask() const
{
	const dim3 gridSize = cuda_utils::gridSizeCalculate(texKernelBlockSize, baseRows, baseCols, true);
	const af::array customMask(baseRows, baseCols);
	//call NVF kernel
	nvf<3> << <gridSize, texKernelBlockSize, 0, afStream >> > (texObj, customMask.device<float>(), baseCols, baseRows);
	//transfer ownership to arrayfire and return output array
	unlockArrays(customMask);
	return customMask;
}

af::array WatermarkCuda::computeScaledNeighbors(const af::array& coefficients) const
{
	const dim3 gridSize = cuda_utils::gridSizeCalculate(texKernelBlockSize, baseRows, baseCols, true);
	const af::array neighbors(baseRows, baseCols);
	setCoeffs(coefficients.device<float>());
	calculate_scaled_neighbors_p3 << <gridSize, texKernelBlockSize, 0, afStream >> > (texObj, neighbors.device<float>(), baseCols, baseRows);
	//transfer ownership to arrayfire and return output array
	unlockArrays(neighbors, coefficients);
	return neighbors;
}

BufferType WatermarkCuda::makeWatermark(const BufferType& inputImage, const BufferType& outputImage, float& watermarkStrength, MASK_TYPE maskType)
{
	return makeWatermarkGpu(inputImage, outputImage, randomMatrix, strengthFactor, watermarkStrength, maskType);
}

af::array WatermarkCuda::computePredictionErrorMask(const af::array& image, af::array& errorSequence, af::array& coefficients, const bool maskNeeded) const
{
	const dim3 gridSize = cuda_utils::gridSizeCalculate(meKernelBlockSize, meKernelDims.y, meKernelDims.x);
	//call prediction error mask kernel
	const af::array RxPartial(baseRows, meKernelDims.x);
	const af::array rxPartial(baseRows, meKernelDims.x / 8);
	me_p3 <<<gridSize, meKernelBlockSize, 0, afStream >>> (texObj, RxPartial.device<float>(), rxPartial.device<float>(), baseCols, meKernelDims.x, baseRows);
	unlockArrays(RxPartial, rxPartial);
	//calculation of coefficients, error sequence and mask
	const auto correlationArrays = transformCorrelationArrays(RxPartial, rxPartial, p);
	coefficients = af::solve(correlationArrays.first, correlationArrays.second);
	//if system is not solvable, don't waste computing the error sequence, there is no watermark to embed
	if (af::anyTrue<bool>(af::isNaN(coefficients))) 
	{
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

float WatermarkCuda::detectWatermark(const BufferType& inputImage, MASK_TYPE maskType)
{
	return detectWatermarkGpu(inputImage, randomMatrix, maskType);
}