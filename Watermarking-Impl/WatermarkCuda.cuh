#pragma once
#include "cuda_utils.hpp"
#include "CudaStreamManager.hpp"
#include "kernels/kernels.cuh"
#include "WatermarkBase.hpp"
#include "WatermarkGpu.hpp"
#include <arrayfire.h>
#include <cuda_runtime.h>
#include <string>

/*!
 *  \brief  Functions for watermark computation and detection, CUDA implementation.
 *  \author Dimitris Karatzas
 */
template <int p>
class WatermarkCuda final : public WatermarkGPU<p> {
  public:
    WatermarkCuda<p>(const unsigned int rows, const unsigned int cols, const std::string& randomMatrixPath, const float psnr)
        : WatermarkGPU<p>(rows, cols, randomMatrixPath, psnr), afStream(CudaStreamManager::getInstance().getAfStream()), gridOptimalMe(cuda_utils::gridSize1DMeStridedCalculate()) {
        const unsigned int totalBlocksY = WatermarkBase::align<meBlockSize.x>(this->baseRows) / meBlockSize.x;
        meParams = {totalBlocksY, totalBlocksY * this->baseCols};
    }

  private:
    static constexpr dim3 windowBlockSize{32, 8}, meBlockSize{p >= 7 ? 128 : 256, 1};
    static constexpr unsigned int corrPartialBlockSize = 768, corrFinalBlockSize = 1024, strWatermarkBlockSize = 768, applyWatermarkBlockSize = 768;
    dim3 meParams; // used to store ME kernel parameters (total blocks in Y dimension and total tasks) for optimal configuration
    cudaStream_t afStream;
    unsigned int gridOptimalMe;

    af::array computeStrengthenedWatermark(const af::array& inputGrayImage, const af::array& inputImage, float& watermarkStrength, const MASK_TYPE maskType) const override {
        const af::array u(inputGrayImage.dims(), f32);
        const af::array output(inputImage.dims(), u8);
        const af::array sumSq = af::constant(0.0f, 1, f32);

        float* uPtr = u.device<float>();
        float* sumSqPtr = sumSq.device<float>();
        // fused kernel to compute NVF mask, strengthened watermark (u) and sum of squares of u
        if (maskType == NVF) {
            const dim3 gridSize = cuda_utils::gridSizeCalculate(windowBlockSize, this->baseCols, this->baseRows);
            nvf_u_and_sumsq_fused<p>
                <<<gridSize, windowBlockSize, 0, afStream>>>(inputGrayImage.template device<float>(), this->randomMatrix.template device<float>(), uPtr, sumSqPtr, this->baseCols, this->baseRows);
            this->unlockArrays(inputGrayImage);
        } else {
            // find max of error sequence, this cannot be fused because it is a global reduction
            const af::array errorSeq = computePredictionErrorData(inputGrayImage, true);
            const af::array errorSeqMax = af::max(af::flat(errorSeq));
            // fused kernel to compute ME mask, strengthened watermark (u) and sum of squares of u
            const int blocksComputeU = cuda_utils::gridSize1DStridedCalculate(this->totalPixels, strWatermarkBlockSize);
            me_u_and_sumsq_fused<<<blocksComputeU, strWatermarkBlockSize, 0, afStream>>>(errorSeq.device<float>(), this->randomMatrix.template device<float>(), uPtr, sumSqPtr,
                                                                                         errorSeqMax.device<float>(), this->totalPixels);
            this->unlockArrays(errorSeqMax, errorSeq);
        }
        // compute and apply watermark
        const int blocksApply = cuda_utils::gridSize1DStridedCalculate(this->totalPixels, applyWatermarkBlockSize);
        apply_watermark_fused<<<blocksApply, applyWatermarkBlockSize, 0, afStream>>>(inputImage.device<float>(), uPtr, sumSqPtr, output.device<unsigned char>(), this->strengthNumerator,
                                                                                     this->totalPixels, static_cast<int>(inputImage.dims(2)));
        this->unlockArrays(inputImage, u, sumSq, output, this->randomMatrix);
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
        calculate_error_sequence<p, false><<<gridSize, windowBlockSize, 0, afStream>>>(image.device<float>(), nullptr, errorSequence.device<float>(), this->coefficients.template device<float>(),
                                                                                       this->baseCols, this->baseRows, calculateAbs, this->stopFlag.template device<int>());
        // transfer ownership to arrayfire and return output array
        this->unlockArrays(image, errorSequence, this->coefficients, this->stopFlag);
        return errorSequence;
    }

    af::array computeErrorSequence(const af::array& inputA, const af::array& inputB) const override {
        // transposed grid dimensions because of column-major order in arrayfire
        const dim3 gridSize = cuda_utils::gridSizeCalculate(windowBlockSize, this->baseCols, this->baseRows);
        const af::array errorSequence(this->baseRows, this->baseCols);
        // call error sequence kernel
        calculate_error_sequence<p, true><<<gridSize, windowBlockSize, 0, afStream>>>(inputA.device<float>(), inputB.device<float>(), errorSequence.device<float>(),
                                                                                      this->coefficients.template device<float>(), this->baseCols, this->baseRows, false,
                                                                                      this->stopFlag.template device<int>());
        // transfer ownership to arrayfire and return output array
        this->unlockArrays(inputA, inputB, errorSequence, this->coefficients, this->stopFlag);
        return errorSequence;
    }

    af::array computePredictionErrorData(const af::array& image, const bool calculateAbs) const override {
        constexpr int RxSize = (this->localSize * (this->localSize + 1)) / 2;
        constexpr int rxSize = this->localSize;

        const af::array Rx = af::constant(0.0f, RxSize, 1, f32);
        const af::array rx = af::constant(0.0f, rxSize, 1, f32);
        float* RxPtr = Rx.device<float>();
        float* rxPtr = rx.device<float>();

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