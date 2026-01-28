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
        : WatermarkGPU<p>(rows, cols, randomMatrixPath, psnr), meKernelDims{WatermarkBase::align<meBlockSize.x>(cols), rows}, afStream(CudaStreamManager::getInstance().getAfStream()) {}

  private:
    static constexpr dim3 windowBlockSize{16, 16}, meBlockSize{p == 9 ? 128 : 256, 1};
    static constexpr unsigned int corrPartialBlockSize = 256, corrFinalBlockSize = 1024;
    dim3 meKernelDims;
    cudaStream_t afStream;

    af::array computeCustomMask(const af::array& inputImage) const {
        // transposed grid dimensions because of column-major order in arrayfire
        const dim3 gridSize = cuda_utils::gridSizeCalculate(windowBlockSize, this->baseCols, this->baseRows);
        const af::array customMask(this->baseRows, this->baseCols);
        // call NVF kernel
        nvf<p><<<gridSize, windowBlockSize, 0, afStream>>>(inputImage.device<float>(), customMask.device<float>(), this->baseCols, this->baseRows);
        // transfer ownership to arrayfire and return output array
        this->unlockArrays(inputImage, customMask);
        return customMask;
    }

    af::array computeErrorSequence(const af::array& image, const bool calculateAbs) const {
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

    af::array computeErrorSequence(const af::array& inputA, const af::array& inputB) const {
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

    af::array computePredictionErrorData(const af::array& image, const bool calculateAbs) const {
        const int blocksX = meKernelDims.x / meBlockSize.x;
        const dim3 gridSize = cuda_utils::gridSizeCalculate(meBlockSize, meKernelDims.y, meKernelDims.x);

        af::array RxPartial, rxPartial;
        // call prediction error Rx/rx partials calculation kernel
        if constexpr (p == 3) {
            RxPartial = af::array(this->baseRows, blocksX * 36);
            rxPartial = af::array(this->baseRows, blocksX * 8);
            me_p3<<<gridSize, meBlockSize, 0, afStream>>>(image.device<float>(), RxPartial.device<float>(), rxPartial.device<float>(), this->baseCols, this->baseRows);
        } else if (p == 5) {
            RxPartial = af::array(this->baseRows, blocksX * 300);
            rxPartial = af::array(this->baseRows, blocksX * 24);
            me_p5<<<gridSize, meBlockSize, 0, afStream>>>(image.device<float>(), RxPartial.device<float>(), rxPartial.device<float>(), this->baseCols, this->baseRows);
        } else if (p == 7) {
            RxPartial = af::array(this->baseRows, blocksX * 1176);
            rxPartial = af::array(this->baseRows, blocksX * 48);
            me_p7<<<gridSize, meBlockSize, 0, afStream>>>(image.device<float>(), RxPartial.device<float>(), rxPartial.device<float>(), this->baseCols, this->baseRows);
        } else {
            RxPartial = af::array(this->baseRows, blocksX * 3240);
            rxPartial = af::array(this->baseRows, blocksX * 80);
            me_p9<<<gridSize, meBlockSize, 0, afStream>>>(image.device<float>(), RxPartial.device<float>(), rxPartial.device<float>(), this->baseCols, this->baseRows);
        }
        this->unlockArrays(image, RxPartial, rxPartial);
        // calculation of coefficients and error sequence
        const auto correlationArrays = this->transformCorrelationArrays(RxPartial, rxPartial);
        const af::array& Rx = correlationArrays.first;
        const af::array& rx = correlationArrays.second;
        // very low latency solver for p = 3 and p = 5
        if constexpr (p == 3 || p == 5)
            cholesky_solver<p><<<1, 1, 0, afStream>>>(Rx.device<float>(), rx.device<float>(), this->coefficients.template device<float>(), this->stopFlag.template device<int>());
        // parallel solver for p >= 7 (one warp)
        else
            cholesky_solver_parallel<p><<<1, 32, 0, afStream>>>(Rx.device<float>(), rx.device<float>(), this->coefficients.template device<float>(), this->stopFlag.template device<int>());
        this->unlockArrays(Rx, rx, this->coefficients, this->stopFlag);
        this->unlockArrays(Rx, rx, this->coefficients, this->stopFlag);
        return computeErrorSequence(image, calculateAbs);
    }

    float computeCorrelation(const af::array& e_u, const af::array& e_z) const {
        const int N = static_cast<int>(e_u.elements());
        const int blocks = (N + corrPartialBlockSize - 1) / corrPartialBlockSize;
        const af::array dotPartial(blocks);
        const af::array uNormPartial(blocks);
        const af::array zNormPartial(blocks);
        const af::array correlationResult(1);
        float* dotPartialPtr = dotPartial.device<float>();
        float* uNormPartialPtr = uNormPartial.device<float>();
        float* zNormPartialPtr = zNormPartial.device<float>();

        // calculate partial dot products and norms
        calculate_partial_correlation<<<blocks, corrPartialBlockSize, 0, afStream>>>(e_u.device<float>(), e_z.device<float>(), dotPartialPtr, uNormPartialPtr, zNormPartialPtr, N);
        // reduce partials and compute correlation
        calculate_final_correlation<<<1, corrFinalBlockSize, 0, afStream>>>(dotPartialPtr, uNormPartialPtr, zNormPartialPtr, correlationResult.device<float>(), blocks);
        // transfer ownership to arrayfire and return output correlation scalar to host
        this->unlockArrays(e_u, e_z, dotPartial, uNormPartial, zNormPartial, correlationResult);
        return correlationResult.scalar<float>();
    }
};