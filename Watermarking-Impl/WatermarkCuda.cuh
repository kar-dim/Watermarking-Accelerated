#pragma once
#include "cuda_utils.hpp"
#include "CudaStreamManager.hpp"
#include "include/WatermarkTypes.hpp"
#include "kernels/kernels.cuh"
#include "WatermarkBase.hpp"
#include "WatermarkGpu.hpp"
#include <algorithm>
#include <arrayfire.h>
#include <cstdint>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <string>
#include <thrust/iterator/transform_iterator.h>

/*!
 *  \brief  Functions for watermark computation and detection, CUDA implementation.
 *  \author Dimitris Karatzas
 */
template <int p>
class WatermarkCuda final : public WatermarkGPU<p> {
  public:
    WatermarkCuda<p>(const unsigned int rows, const unsigned int cols, const std::string& watermarkPassword, const float psnr)
        : WatermarkGPU<p>(rows, cols, watermarkPassword, psnr), afStream(CudaStreamManager::getInstance().getAfStream()), gridOptimalMe(cuda_utils::gridSize1DMeStridedCalculate()) {
        // initialize ME kernel parameters based on image dims, we calculate total blocks in Y dimension and total tasks for optimal configuration
        constexpr unsigned int pixelsPerBlockY = (p == 3) ? (meBlockSize.x * 2) : meBlockSize.x;
        const unsigned int meTotalBlocksY = WatermarkBase::alignUp<pixelsPerBlockY>(this->baseRows) / pixelsPerBlockY;
        meParams = {meTotalBlocksY, meTotalBlocksY * this->baseCols};
        // initialize CUB scratch memory for its reductions
        initializeCubStorage();
    }

  private:
    static constexpr dim3 windowBlockSize{32, 8};
    static constexpr dim3 meBlockSize{p >= 7 ? 128 : 256, 1};
    static constexpr unsigned int corrPartialBlockSize = 768;
    static constexpr unsigned int corrFinalBlockSize = 1024;
    static constexpr unsigned int strWatermarkBlockSize = 768;
    static constexpr unsigned int applyWatermarkBlockSize = 768;
    static constexpr unsigned int maskNormalizationBlockSize = 768;

    dim3 meParams; // used to store ME kernel parameters (total blocks in Y dimension and total tasks) for optimal configuration
    cudaStream_t afStream;
    unsigned int gridOptimalMe;
    af::array cubTempStorage;

    void initializeCubStorage() {
        // ask CUB for required scratch space for the global sum reduction with an abs value transformation
        AbsTransformOp op;
        auto iter = thrust::make_transform_iterator((const float*)nullptr, op);
        size_t tmpBytesTransform = 0;
        cub::DeviceReduce::Max(nullptr, tmpBytesTransform, iter, (float*)nullptr, this->totalPixels, 0);
        // ask CUB for required scratch space for the raw Pointer reduction
        size_t tmpBytesRaw = 0;
        cub::DeviceReduce::Max(nullptr, tmpBytesRaw, (const float*)nullptr, (float*)nullptr, this->totalPixels, 0);
        // allocate the global scratchpad using the maximum required size to fit all cases
        size_t finalTmpBytes = std::max(tmpBytesTransform, tmpBytesRaw);
        cubTempStorage = af::array(finalTmpBytes, u8);
    }

    // helper function to reduce (sum) an array of totalPixel pixels, or an iterated array with a transformation OP
    template <typename InputIteratorT>
    void reduceMaxCub(InputIteratorT in, float* out) const {
        size_t tmpStorageBytes = cubTempStorage.bytes();
        // execute CUB global reduction
        cub::DeviceReduce::Max((void*)cubTempStorage.device<unsigned char>(), tmpStorageBytes, in, out, this->totalPixels, afStream);
        this->unlockArrays(cubTempStorage);
    }

    af::array computeStrengthenedWatermark(const af::array& inputGrayImage, const af::array& inputImage, const MaskMethod maskType) const override {
        const af::array u(inputGrayImage.dims(), f32);
        const af::array output(inputImage.dims(), u8);
        const af::array sumSq = af::constant(0, 1, u64);

        float* uPtr = u.device<float>();
        uint64_t* sumSqPtr = sumSq.device<uint64_t>();
        // fused kernel to compute NVF mask, strengthened watermark (u) and sum of squares of u
        if (maskType == MaskMethod::NVF) {
            const dim3 gridSize = cuda_utils::gridSizeCalculate(windowBlockSize, this->baseCols, this->baseRows);
            nvf_u_and_sumsq_fused<p>
                <<<gridSize, windowBlockSize, 0, afStream>>>(inputGrayImage.template device<float>(), this->randomMatrix.template device<float>(), uPtr, sumSqPtr, this->baseCols, this->baseRows);
            this->unlockArrays(inputGrayImage, this->randomMatrix);
        } else {
            // find max of error sequence
            const af::array errorSeq = computePredictionErrorData(inputGrayImage, true);
            const af::array errorSeqMax(1, f32);
            const float* errSeqPtr = errorSeq.device<float>();
            float* errMaxPtr = errorSeqMax.device<float>();
            reduceMaxCub(errSeqPtr, errMaxPtr);
            // fused kernel to compute ME mask, strengthened watermark (u) and sum of squares of u
            const int blocksComputeU = cuda_utils::gridSize1DStridedCalculate(this->totalPixels, strWatermarkBlockSize);
            me_u_and_sumsq_fused<<<blocksComputeU, strWatermarkBlockSize, 0, afStream>>>(errSeqPtr, this->randomMatrix.template device<float>(), uPtr, sumSqPtr, errMaxPtr, this->totalPixels);
            this->unlockArrays(errorSeq, errorSeqMax, this->randomMatrix);
        }
        // compute and apply watermark
        const int blocksApply = cuda_utils::gridSize1DStridedCalculate(this->totalPixels, applyWatermarkBlockSize);
        apply_watermark_fused<<<blocksApply, applyWatermarkBlockSize, 0, afStream>>>(inputImage.device<float>(), uPtr, sumSqPtr, output.device<unsigned char>(), this->strengthNumerator,
                                                                                     this->totalPixels, static_cast<int>(inputImage.dims(2)));
        this->unlockArrays(inputImage, u, sumSq, output);
        return output;
    }

    af::array computeCustomMask(const af::array& inputImage) const override {
        // transposed grid dimensions because of column-major order in arrayfire
        const dim3 gridSize = cuda_utils::gridSizeCalculate(windowBlockSize, this->baseCols, this->baseRows);
        const af::array customMask(this->baseRows, this->baseCols);
        // call NVF kernel
        nvf<p><<<gridSize, windowBlockSize, 0, afStream>>>(inputImage.device<float>(), customMask.device<float>(), this->baseCols, this->baseRows);
        // transfer ownership to arrayfire and return output array
        this->unlockArrays(inputImage, customMask);
        return customMask;
    }

    af::array computeErrorSequence(const af::array& image, const bool calculateAbs) const override {
        // transposed grid dimensions because of column-major order in arrayfire
        const dim3 gridSize = cuda_utils::gridSizeCalculate(windowBlockSize, this->baseCols, this->baseRows);
        const af::array errorSequence(this->baseRows, this->baseCols);
        // call error sequence kernel
        calculate_error_sequence<p><<<gridSize, windowBlockSize, 0, afStream>>>(image.device<float>(), nullptr, errorSequence.device<float>(), this->coefficients.template device<float>(),
                                                                                       this->baseCols, this->baseRows, calculateAbs, this->stopFlag.template device<int>());
        // transfer ownership to arrayfire and return output array
        this->unlockArrays(image, errorSequence, this->coefficients, this->stopFlag);
        return errorSequence;
    }

    af::array computePredictionErrorData(const af::array& image, const bool calculateAbs) const override {
        constexpr int RxSize = (this->localSize * (this->localSize + 1)) / 2;
        constexpr int rxSize = this->localSize;

        const af::array Rx = af::constant(0, RxSize, 1, u64);
        const af::array rx = af::constant(0, rxSize, 1, u64);
        uint64_t* RxPtr = Rx.device<uint64_t>();
        uint64_t* rxPtr = rx.device<uint64_t>();

        // compute autocorrelation matrix Rx and vector rx for the ME coefficients using grid-stride kernels optimized for the number of SMs on the GPU
        if constexpr (p == 3)
            me_p3<<<gridOptimalMe, meBlockSize.x, 0, afStream>>>(image.device<float>(), RxPtr, rxPtr, this->baseCols, this->baseRows, meParams.x, meParams.y);
        else if constexpr (p == 5)
            me_p5<<<gridOptimalMe, meBlockSize.x, 0, afStream>>>(image.device<float>(), RxPtr, rxPtr, this->baseCols, this->baseRows, meParams.x, meParams.y);
        else if constexpr (p == 7)
            me_p7<<<gridOptimalMe, meBlockSize.x, 0, afStream>>>(image.device<float>(), RxPtr, rxPtr, this->baseCols, this->baseRows, meParams.x, meParams.y);
        else
            me_p9<<<gridOptimalMe, meBlockSize.x, 0, afStream>>>(image.device<float>(), RxPtr, rxPtr, this->baseCols, this->baseRows, meParams.x, meParams.y);
        // solve for coefficients using Cholesky solver, single thread for small p and parallel for larger p
        if constexpr (p <= 5)
            cholesky_solver<p><<<1, 1, 0, afStream>>>(RxPtr, rxPtr, this->coefficients.template device<float>(), this->stopFlag.template device<int>());
        else
            cholesky_solver_parallel<p><<<1, 32, 0, afStream>>>(RxPtr, rxPtr, this->coefficients.template device<float>(), this->stopFlag.template device<int>());
        // calculate error sequence
        this->unlockArrays(image, Rx, rx, this->coefficients, this->stopFlag);
        return computeErrorSequence(image, calculateAbs);
    }

    af::array computePredictionErrorMask(const af::array& errorSequence) const override {
        const af::array mask(errorSequence.dims(), f32);
        const af::array maxVal(1, f32);
        const float* errSeqPtr = errorSequence.device<float>();
        float* maskPtr = mask.device<float>();
        float* maxValPtr = maxVal.device<float>();
        // use cub to find max of absolute values in error sequence, we use a transform iterator to apply abs on the fly
        reduceMaxCub(thrust::make_transform_iterator(errSeqPtr, AbsTransformOp{}), maxValPtr);
        // launch the normalization kernel to compute the mask, dividing each element of the error sequence with the max value we just computed
        const int gridSize = cuda_utils::gridSize1DStridedCalculate(this->totalPixels, maskNormalizationBlockSize);
        compute_abs_normalized_mask<<<gridSize, maskNormalizationBlockSize, 0, afStream>>>(errSeqPtr, maskPtr, maxValPtr, this->totalPixels);
        this->unlockArrays(errorSequence, mask, maxVal);
        return mask;
    }

    float computeCorrelation(const af::array& e_u, const af::array& mask) const override {
        const dim3 gridSize = cuda_utils::gridSizeCalculate(windowBlockSize, this->baseCols, this->baseRows);
        const int numBlocks = gridSize.x * gridSize.y;
        const af::array dotPartial(numBlocks, f32);
        const af::array uNormPartial(numBlocks, f32);
        const af::array zNormPartial(numBlocks, f32);
        // launch fused error sequence + partial correlation
        calculate_error_sequence_and_partial_corr_fused<p><<<gridSize, windowBlockSize, 0, afStream>>>(
            mask.device<float>(), this->randomMatrix.template device<float>(), e_u.device<float>(), this->coefficients.template device<float>(), dotPartial.device<float>(),
            uNormPartial.device<float>(), zNormPartial.device<float>(), this->baseCols, this->baseRows, false, this->stopFlag.template device<int>());
        // reduce partials and compute correlation
        const af::array correlationResult(1, f32);
        calculate_final_correlation<<<1, corrFinalBlockSize, 0, afStream>>>(dotPartial.device<float>(), uNormPartial.device<float>(), zNormPartial.device<float>(), correlationResult.device<float>(),
                                                                            numBlocks);
        // transfer ownership to arrayfire and return output correlation scalar to host
        this->unlockArrays(mask, e_u, dotPartial, uNormPartial, zNormPartial, correlationResult, this->coefficients, this->stopFlag, this->randomMatrix);
        return correlationResult.scalar<float>();
    }
};