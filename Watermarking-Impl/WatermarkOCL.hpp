#pragma once
#include "opencl_init.h"
#include "opencl_utils.hpp"
#include "WatermarkBase.hpp"
#include "WatermarkGpu.hpp"
#include <af/opencl.h>
#include <algorithm>
#include <arrayfire.h>
#include <memory>
#include <string>

struct dim2 {
    dim_t rows;
    dim_t cols;
};

using namespace cl_utils;

/*!
 *  \brief  Functions for watermark computation and detection, OpenCL implementation.
 *  \author Dimitris Karatzas
 */
template <int p> class WatermarkOCL final : public WatermarkGPU<p> {
  public:
    WatermarkOCL<p>(const unsigned int rows, const unsigned int cols, const std::string& randomMatrixPath, const float psnr)
        : WatermarkGPU<p>(rows, cols, randomMatrixPath, psnr), texKernelDims{align<windowLocalSize>(rows), align<windowLocalSize>(cols)}, meKernelDims{rows, align<meLocalSize>(cols)},
          programs(buildKernels(p)) {}

  private:
    using WatermarkBase::align;
    using clMemPtr = std::unique_ptr<cl_mem>;

    static constexpr unsigned int corrPartialLocalSize = 256;
    static constexpr unsigned int windowLocalSize = 16;
    static constexpr unsigned int meLocalSize = 256;
    static constexpr unsigned int choleskyLocalSize = p < 7 ? 1 : 64; // for p >= 7 we use 64-thread cholesky solver, for p < 7 single thread (faster for small p)

    cl::Context context{afcl::getContext(true)};
    cl::CommandQueue queue{afcl::getQueue(true)};
    cl::Device device{afcl::getDeviceId(), true};
    dim2 texKernelDims, meKernelDims;
    unsigned int corrFinalLocalSize = maxPow2WorkGroupSize(device);
    cl::Program programs;

    af::array computeCustomMask(const af::array& image) const override {
        const af::array customMask(this->baseRows, this->baseCols);
        const clMemPtr imageMem(image.device<cl_mem>());
        const clMemPtr outputMem(customMask.device<cl_mem>());
        // transposed global dimensions because of column-major order in arrayfire
        executeKernel(
            [&]() {
                queue.enqueueNDRangeKernel(KernelBuilder(programs, "nvf").args(wrap(imageMem.get()), wrap(outputMem.get()), this->baseCols, this->baseRows).build(), cl::NDRange(),
                                           cl::NDRange(texKernelDims.rows, texKernelDims.cols), cl::NDRange(windowLocalSize, windowLocalSize));
                this->unlockArrays(image, customMask);
            },
            "nvf");
        return customMask;
    }

    af::array computeErrorSequence(const af::array& image, const bool calculateAbs) const override {
        const af::array errorSequence(this->baseRows, this->baseCols);
        const clMemPtr imageMem(image.device<cl_mem>());
        const clMemPtr coeffsMem(this->coefficients.template device<cl_mem>());
        const clMemPtr errorSequenceMem(errorSequence.device<cl_mem>());
        const clMemPtr stopFlagMem(this->stopFlag.template device<cl_mem>());
        // transposed global dimensions because of column-major order in arrayfire
        executeKernel(
            [&]() {
                queue.enqueueNDRangeKernel(
                    KernelBuilder(programs, "error_sequence")
                        .args(wrap(imageMem.get()), wrap(errorSequenceMem.get()), wrap(coeffsMem.get()), this->baseCols, this->baseRows, (int)calculateAbs, wrap(stopFlagMem.get()))
                        .build(),
                    cl::NDRange(), cl::NDRange(texKernelDims.rows, texKernelDims.cols), cl::NDRange(windowLocalSize, windowLocalSize));
                this->unlockArrays(image, errorSequence, this->coefficients, this->stopFlag);
            },
            "error_sequence");
        return errorSequence;
    }

    af::array computeErrorSequence(const af::array& inputA, const af::array& inputB) const override {
        const af::array errorSequence(this->baseRows, this->baseCols);
        const clMemPtr inputAmem(inputA.device<cl_mem>());
        const clMemPtr inputBmem(inputB.device<cl_mem>());
        const clMemPtr coeffsMem(this->coefficients.template device<cl_mem>());
        const clMemPtr errorSequenceMem(errorSequence.device<cl_mem>());
        const clMemPtr stopFlagMem(this->stopFlag.template device<cl_mem>());
        // transposed global dimensions because of column-major order in arrayfire
        executeKernel(
            [&]() {
                queue.enqueueNDRangeKernel(
                    KernelBuilder(programs, "error_sequence_fused")
                        .args(wrap(inputAmem.get()), wrap(inputBmem.get()), wrap(errorSequenceMem.get()), wrap(coeffsMem.get()), this->baseCols, this->baseRows, wrap(stopFlagMem.get()))
                        .build(),
                    cl::NDRange(), cl::NDRange(texKernelDims.rows, texKernelDims.cols), cl::NDRange(windowLocalSize, windowLocalSize));
                this->unlockArrays(inputA, inputB, errorSequence, this->coefficients, this->stopFlag);
            },
            "error_sequence_fused");
        return errorSequence;
    }

    af::array computePredictionErrorData(const af::array& image, const bool calculateAbs) const override {
        const auto meArraysBaseWidth = meKernelDims.cols / meLocalSize;
        const clMemPtr imageMem(image.device<cl_mem>());
        af::array RxPartial, rxPartial;
        return executeKernel(
            [&]() -> af::array {
                if constexpr (p == 3) {
                    RxPartial = af::array(this->baseRows, meArraysBaseWidth * 36);
                    rxPartial = af::array(this->baseRows, meArraysBaseWidth * 8);
                } else if (p == 5) {
                    RxPartial = af::array(this->baseRows, meArraysBaseWidth * 300);
                    rxPartial = af::array(this->baseRows, meArraysBaseWidth * 24);
                } else if (p == 7) {
                    RxPartial = af::array(this->baseRows, meArraysBaseWidth * 1176);
                    rxPartial = af::array(this->baseRows, meArraysBaseWidth * 48);
                } else {
                    RxPartial = af::array(this->baseRows, meArraysBaseWidth * 3240);
                    rxPartial = af::array(this->baseRows, meArraysBaseWidth * 80);
                }
                const clMemPtr RxPartialMem(RxPartial.device<cl_mem>());
                const clMemPtr rxPartialMem(rxPartial.device<cl_mem>());
                // call prediction error Rx/rx partials calculation kernel
                queue.enqueueNDRangeKernel(KernelBuilder(programs, "me").args(wrap(imageMem.get()), wrap(RxPartialMem.get()), wrap(rxPartialMem.get()), this->baseCols, this->baseRows).build(),
                                           cl::NDRange(), cl::NDRange(meKernelDims.cols, meKernelDims.rows), cl::NDRange(meLocalSize, 1));
                // return memory to arrayfire
                this->unlockArrays(image, RxPartial, rxPartial);

                // calculation of coefficients and error sequence
                const auto correlationArrays = this->transformCorrelationArrays(RxPartial, rxPartial);
                const af::array& Rx = correlationArrays.first;
                const af::array& rx = correlationArrays.second;
                const clMemPtr RxMemPtr(Rx.device<cl_mem>());
                const clMemPtr rxMemPtr(rx.device<cl_mem>());
                const clMemPtr coeffsMem(this->coefficients.template device<cl_mem>());
                const clMemPtr stopFlagMem(this->stopFlag.template device<cl_mem>());
                queue.enqueueNDRangeKernel(KernelBuilder(programs, "cholesky_solver").args(wrap(RxMemPtr.get()), wrap(rxMemPtr.get()), wrap(coeffsMem.get()), wrap(stopFlagMem.get())).build(),
                                           cl::NDRange(), cl::NDRange(choleskyLocalSize), cl::NDRange(choleskyLocalSize));
                // return memory to arrayfire
                this->unlockArrays(Rx, rx, this->coefficients, this->stopFlag);
                return computeErrorSequence(image, calculateAbs);
            },
            "me");
    }

    float computeCorrelation(const af::array& e_u, const af::array& e_z) const override {
        const int N = static_cast<int>(e_u.elements());
        const int neededBlocks = (N + corrPartialLocalSize - 1) / corrPartialLocalSize;
        const int blocks = std::min(neededBlocks, 2560);
        const int globalSizePartials = blocks * corrPartialLocalSize;
        const af::array dotPartial(blocks);
        const af::array uNormPartial(blocks);
        const af::array zNormPartial(blocks);
        const af::array correlationResult(1);
        const clMemPtr euMem(e_u.device<cl_mem>());
        const clMemPtr ezMem(e_z.device<cl_mem>());
        const clMemPtr dotPartialMem(dotPartial.device<cl_mem>());
        const clMemPtr uNormPartialMem(uNormPartial.device<cl_mem>());
        const clMemPtr zNormPartialMem(zNormPartial.device<cl_mem>());
        const clMemPtr correlationResultMem(correlationResult.device<cl_mem>());
        float correlation = 0.0f;
        executeKernel(
            [&]() {
                // calculate partial dot products and norms
                queue.enqueueNDRangeKernel(KernelBuilder(programs, "calculate_partial_correlation")
                                               .args(wrap(euMem.get()), wrap(ezMem.get()), wrap(dotPartialMem.get()), wrap(uNormPartialMem.get()), wrap(zNormPartialMem.get()), N)
                                               .build(),
                                           cl::NDRange(), cl::NDRange(globalSizePartials), cl::NDRange(corrPartialLocalSize));
                // reduce partials and compute correlation
                queue.enqueueNDRangeKernel(KernelBuilder(programs, "calculate_final_correlation")
                                               .args(wrap(dotPartialMem.get()), wrap(uNormPartialMem.get()), wrap(zNormPartialMem.get()), wrap(correlationResultMem.get()), blocks)
                                               .build(),
                                           cl::NDRange(), cl::NDRange(corrFinalLocalSize), cl::NDRange(corrFinalLocalSize));
                // retrieve the correlation result
                this->unlockArrays(e_u, e_z, dotPartial, uNormPartial, zNormPartial, correlationResult);
                correlation = correlationResult.scalar<float>();
            },
            "compute correlation kernels");
        return correlation;
    }
};