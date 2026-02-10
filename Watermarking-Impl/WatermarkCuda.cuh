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
template <int p> class WatermarkCuda final : public WatermarkGPU<p> {
  public:
    WatermarkCuda<p>(const unsigned int rows, const unsigned int cols, const std::string& randomMatrixPath, const float psnr)
        : WatermarkGPU<p>(rows, cols, randomMatrixPath, psnr), meKernelDims{WatermarkBase::align<meBlockSize.x>(cols), rows}, afStream(CudaStreamManager::getInstance().getAfStream()),
          gridOptimalMe(cuda_utils::gridSize1DMeStridedCalculate()) {}

  private:
    static constexpr dim3 windowBlockSize{16, 16}, meBlockSize{p >= 7 ? 128 : 256, 1};
    static constexpr unsigned int corrPartialBlockSize = 768, corrFinalBlockSize = 1024, strWatermarkBlockSize = 768, applyWatermarkBlockSize = 768;
    dim3 meKernelDims;
    cudaStream_t afStream;
    unsigned int gridOptimalMe;

    af::array computeStrengthenedWatermark(const af::array& inputGrayImage, const af::array& inputImage, float& watermarkStrength, const MASK_TYPE maskType) const override {
        const af::array u(inputGrayImage.dims(), f32);
        const af::array output(inputImage.dims(), u8);
        const af::array sumSq = af::constant(0.0f, 1, f32);
        // compute mask
        const af::array mask = maskType == ME ? this->template computePredictionErrorMask<false>(computePredictionErrorData(inputGrayImage, true)) : computeCustomMask(inputGrayImage);
        // compute strengthened watermark and sum of squares in one kernel to save bandwidth
        const int blocksComputeU = cuda_utils::gridSize1DStridedCalculate(this->totalPixels, strWatermarkBlockSize);
        float* uPtr = u.device<float>();
        float* sumSqPtr = sumSq.device<float>();
        compute_u_and_sumsq<<<blocksComputeU, strWatermarkBlockSize, 0, afStream>>>(mask.device<float>(), this->randomMatrix.template device<float>(), uPtr, sumSqPtr, this->totalPixels);
        // compute and apply watermark
        const int blocksApply = cuda_utils::gridSize1DStridedCalculate(this->totalPixels, applyWatermarkBlockSize);
        apply_watermark_fused<<<blocksApply, applyWatermarkBlockSize, 0, afStream>>>(inputImage.device<float>(), uPtr, sumSqPtr, output.device<unsigned char>(), this->strengthNumerator,
                                                                                     this->totalPixels, static_cast<int>(inputImage.dims(2)));
        this->unlockArrays(inputImage, u, sumSq, output, mask, this->randomMatrix);
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
        constexpr int threadsPerBlock = (p >= 7) ? 128 : 256;

        const int blocksX = meKernelDims.x / threadsPerBlock;

        const af::array Rx = af::constant(0.0f, RxSize, 1, f32);
        const af::array rx = af::constant(0.0f, rxSize, 1, f32);
        float* RxPtr = Rx.device<float>();
        float* rxPtr = rx.device<float>();

        // compute autocorrelation matrix Rx and vector rx for the ME coefficients using grid-stride kernels optimized for the number of SMs on the GPU
        if constexpr (p == 3)
            me_p3<<<gridOptimalMe, threadsPerBlock, 0, afStream>>>(image.device<float>(), RxPtr, rxPtr, this->baseCols, this->baseRows, blocksX);
        else if constexpr (p == 5)
            me_p5<<<gridOptimalMe, threadsPerBlock, 0, afStream>>>(image.device<float>(), RxPtr, rxPtr, this->baseCols, this->baseRows, blocksX);
        else if constexpr (p == 7)
            me_p7<<<gridOptimalMe, threadsPerBlock, 0, afStream>>>(image.device<float>(), RxPtr, rxPtr, this->baseCols, this->baseRows, blocksX);
        else
            me_p9<<<gridOptimalMe, threadsPerBlock, 0, afStream>>>(image.device<float>(), RxPtr, rxPtr, this->baseCols, this->baseRows, blocksX);
        // solve for coefficients using Cholesky solver, single thread for small p and parallel for larger p
        if constexpr (p <= 5)
            cholesky_solver<p><<<1, 1, 0, afStream>>>(RxPtr, rxPtr, this->coefficients.template device<float>(), this->stopFlag.template device<int>());
        else
            cholesky_solver_parallel<p><<<1, 32, 0, afStream>>>(RxPtr, rxPtr, this->coefficients.template device<float>(), this->stopFlag.template device<int>());

        // calculate error sequence
        this->unlockArrays(image, Rx, rx, this->coefficients, this->stopFlag);
        return computeErrorSequence(image, calculateAbs);
    }

    float computeCorrelation(const af::array& e_u, const af::array& e_z) const override {
        const int blocks = cuda_utils::gridSize1DStridedCalculate(this->totalPixels, corrPartialBlockSize);
        const af::array dotPartial(blocks);
        const af::array uNormPartial(blocks);
        const af::array zNormPartial(blocks);
        const af::array correlationResult(1);
        float* dotPartialPtr = dotPartial.device<float>();
        float* uNormPartialPtr = uNormPartial.device<float>();
        float* zNormPartialPtr = zNormPartial.device<float>();
        // calculate partial dot products and norms
        calculate_partial_correlation<<<blocks, corrPartialBlockSize, 0, afStream>>>(e_u.device<float>(), e_z.device<float>(), dotPartialPtr, uNormPartialPtr, zNormPartialPtr, this->totalPixels);
        // reduce partials and compute correlation
        calculate_final_correlation<<<1, corrFinalBlockSize, 0, afStream>>>(dotPartialPtr, uNormPartialPtr, zNormPartialPtr, correlationResult.device<float>(), blocks);
        // transfer ownership to arrayfire and return output correlation scalar to host
        this->unlockArrays(e_u, e_z, dotPartial, uNormPartial, zNormPartial, correlationResult);
        return correlationResult.scalar<float>();
    }
};