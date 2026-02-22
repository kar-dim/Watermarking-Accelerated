#pragma once
#include "opencl_init.h"
#include "opencl_utils.hpp"
#include "WatermarkBase.hpp"
#include "WatermarkGpu.hpp"
#include <af/opencl.h>
#include <arrayfire.h>
#include <memory>
#include <string>
#include <utility>

struct dim2 {
    dim_t rows;
    dim_t cols;
};

/*!
 *  \brief  Functions for watermark computation and detection, OpenCL implementation.
 *  \author Dimitris Karatzas
 */
template <int p>
class WatermarkOCL final : public WatermarkGPU<p> {
  public:
    WatermarkOCL<p>(const unsigned int rows, const unsigned int cols, const std::string& randomMatrixPath, const float psnr)
        : WatermarkGPU<p>(rows, cols, randomMatrixPath, psnr), texKernelDims{align<windowLocalSize.rows>(rows), align<windowLocalSize.cols>(cols)}, meKernelDims{rows, align<meLocalSize>(cols)},
          programs(cl_utils::buildKernels(p)) {}

  private:
    using WatermarkBase::align;
    using clMemPtr = std::unique_ptr<cl_mem>;

    static constexpr unsigned int corrPartialLocalSize = 256;
    static constexpr dim2 windowLocalSize = {32, 8};
    static constexpr unsigned int meLocalSize = 256;
    static constexpr unsigned int applyWatermarkLocalSize = 256;      // safe universal local size for OpenCL, used in apply watermark kernel and in compute_u_and_sumsq kernel
    static constexpr unsigned int choleskyLocalSize = p < 7 ? 1 : 64; // for p >= 7 we use 64-thread cholesky solver, for p < 7 single thread (faster for small p)

    cl::Context context{afcl::getContext(true)};
    cl::CommandQueue queue{afcl::getQueue(true)};
    cl::Device device{afcl::getDeviceId(), true};
    dim2 texKernelDims, meKernelDims;
    unsigned int corrFinalLocalSize = cl_utils::maxPow2WorkGroupSize(device); // we could use the safe universal of 256, but the kernel that uses this benefits from larger local sizes
    cl::Program programs;

    af::array computeStrengthenedWatermark(const af::array& inputGrayImage, const af::array& inputImage, float& watermarkStrength, const MASK_TYPE maskType) const override {
        using namespace cl_utils;
        const af::array u(inputGrayImage.dims(), f32);
        const af::array output(inputImage.dims(), u8);
        const af::array sumSq = af::constant(0.0f, 1, f32);

        const clMemPtr uMem(u.device<cl_mem>());
        const clMemPtr outputMem(output.device<cl_mem>());
        const clMemPtr sumSqMem(sumSq.device<cl_mem>());
        const clMemPtr randMem(this->randomMatrix.template device<cl_mem>());
        const clMemPtr inputMem(inputImage.device<cl_mem>());
        const clMemPtr inputGrayMem(inputGrayImage.device<cl_mem>());

        if (maskType == NVF) {
            const int workGroups = static_cast<int>((this->texKernelDims.rows / windowLocalSize.rows) * (this->texKernelDims.cols / windowLocalSize.cols));
            const af::array partials(workGroups, f32);
            const clMemPtr partialsMem(partials.device<cl_mem>());

            executeKernel(
                [&]() {
                    // fused kernel to compute NVF mask, strengthened watermark (u) and sum of squares of u
                    queue.enqueueNDRangeKernel(KernelBuilder(programs, "nvf_u_and_partial_sumsq_fused")
                                                   .args(wrap(inputGrayMem.get()), wrap(randMem.get()), wrap(uMem.get()), wrap(partialsMem.get()), this->baseCols, this->baseRows)
                                                   .build(),
                                               cl::NDRange(), cl::NDRange(texKernelDims.rows, texKernelDims.cols), cl::NDRange(windowLocalSize.rows, windowLocalSize.cols));

                    // reduce the partial sums (single workgroup)
                    queue.enqueueNDRangeKernel(KernelBuilder(programs, "reduce_sumsq_partials").args(wrap(partialsMem.get()), wrap(sumSqMem.get()), workGroups).build(), cl::NDRange(),
                                               cl::NDRange(applyWatermarkLocalSize), cl::NDRange(applyWatermarkLocalSize));

                    // apply watermark
                    const int workGroupsApply = calculateLocalGroupsNumber(this->totalPixels, applyWatermarkLocalSize);
                    const int globalSizeApply = workGroupsApply * applyWatermarkLocalSize;

                    queue.enqueueNDRangeKernel(
                        KernelBuilder(programs, "apply_watermark_fused")
                            .args(wrap(inputMem.get()), wrap(uMem.get()), wrap(sumSqMem.get()), wrap(outputMem.get()), this->strengthNumerator, this->totalPixels, static_cast<int>(inputImage.dims(2)))
                            .build(),
                        cl::NDRange(), cl::NDRange(globalSizeApply), cl::NDRange(applyWatermarkLocalSize));

                    this->unlockArrays(inputGrayImage, partials);
                },
                "NVF_computeStrengthenedWatermark");

        } else {
            // find max of error sequence, this cannot be fused because it is a global reduction
            const af::array errorSeq = computePredictionErrorData(inputGrayImage, true);
            const af::array errorSeqMax = af::max(af::flat(errorSeq));

            // fused kernel to compute ME mask, strengthened watermark (u) and sum of squares of u
            const int workGroups = calculateLocalGroupsNumber(this->totalPixels, applyWatermarkLocalSize);
            const int globalSize = workGroups * applyWatermarkLocalSize;

            const af::array partials(workGroups, f32);
            const clMemPtr partialsMem(partials.device<cl_mem>());
            const clMemPtr errorSeqMem(errorSeq.device<cl_mem>());
            const clMemPtr errorSeqMaxMem(errorSeqMax.device<cl_mem>());

            executeKernel(
                [&]() {
                    queue.enqueueNDRangeKernel(KernelBuilder(programs, "me_u_and_partial_sumsq_fused")
                                                   .args(wrap(errorSeqMem.get()), wrap(randMem.get()), wrap(uMem.get()), wrap(partialsMem.get()), wrap(errorSeqMaxMem.get()), this->totalPixels)
                                                   .build(),
                                               cl::NDRange(), cl::NDRange(globalSize), cl::NDRange(applyWatermarkLocalSize));

                    queue.enqueueNDRangeKernel(KernelBuilder(programs, "reduce_sumsq_partials").args(wrap(partialsMem.get()), wrap(sumSqMem.get()), workGroups).build(), cl::NDRange(),
                                               cl::NDRange(applyWatermarkLocalSize), cl::NDRange(applyWatermarkLocalSize));

                    // apply watermark
                    queue.enqueueNDRangeKernel(
                        KernelBuilder(programs, "apply_watermark_fused")
                            .args(wrap(inputMem.get()), wrap(uMem.get()), wrap(sumSqMem.get()), wrap(outputMem.get()), this->strengthNumerator, this->totalPixels, static_cast<int>(inputImage.dims(2)))
                            .build(),
                        cl::NDRange(), cl::NDRange(globalSize), cl::NDRange(applyWatermarkLocalSize));

                    this->unlockArrays(errorSeq, errorSeqMax, partials);
                },
                "ME_computeStrengthenedWatermark");
        }

        this->unlockArrays(inputImage, u, sumSq, output, this->randomMatrix);
        return output;
    }

    af::array computeCustomMask(const af::array& image) const override {
        using namespace cl_utils;
        const af::array customMask(this->baseRows, this->baseCols);
        const clMemPtr imageMem(image.device<cl_mem>());
        const clMemPtr outputMem(customMask.device<cl_mem>());
        // transposed global dimensions because of column-major order in arrayfire
        executeKernel(
            [&]() {
                queue.enqueueNDRangeKernel(KernelBuilder(programs, "nvf").args(wrap(imageMem.get()), wrap(outputMem.get()), this->baseCols, this->baseRows).build(), cl::NDRange(),
                                           cl::NDRange(texKernelDims.rows, texKernelDims.cols), cl::NDRange(windowLocalSize.rows, windowLocalSize.cols));
                this->unlockArrays(image, customMask);
            },
            "nvf");
        return customMask;
    }

    af::array computeErrorSequence(const af::array& image, const bool calculateAbs) const override {
        using namespace cl_utils;
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
                    cl::NDRange(), cl::NDRange(texKernelDims.rows, texKernelDims.cols), cl::NDRange(windowLocalSize.rows, windowLocalSize.cols));
                this->unlockArrays(image, errorSequence, this->coefficients, this->stopFlag);
            },
            "error_sequence");
        return errorSequence;
    }

    af::array computeErrorSequence(const af::array& inputA, const af::array& inputB) const override {
        using namespace cl_utils;
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
                    cl::NDRange(), cl::NDRange(texKernelDims.rows, texKernelDims.cols), cl::NDRange(windowLocalSize.rows, windowLocalSize.cols));
                this->unlockArrays(inputA, inputB, errorSequence, this->coefficients, this->stopFlag);
            },
            "error_sequence_fused");
        return errorSequence;
    }

    af::array computePredictionErrorData(const af::array& image, const bool calculateAbs) const override {
        using namespace cl_utils;
        constexpr int RxSize = (this->localSize * (this->localSize + 1)) / 2;
        constexpr int rxSize = this->localSize;

        return executeKernel(
            [&]() -> af::array {
                const auto meArraysBaseWidth = meKernelDims.cols / meLocalSize;
                const af::array RxPartial(this->baseRows, meArraysBaseWidth * RxSize);
                const af::array rxPartial(this->baseRows, meArraysBaseWidth * rxSize);
                const clMemPtr RxPartialMem(RxPartial.device<cl_mem>());
                const clMemPtr rxPartialMem(rxPartial.device<cl_mem>());
                const clMemPtr imageMem(image.device<cl_mem>());

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
        using namespace cl_utils;
        const int workGroups = calculateLocalGroupsNumber(this->totalPixels, corrPartialLocalSize);
        const int globalSizePartials = workGroups * corrPartialLocalSize;
        const af::array dotPartial(workGroups);
        const af::array uNormPartial(workGroups);
        const af::array zNormPartial(workGroups);
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
                                               .args(wrap(euMem.get()), wrap(ezMem.get()), wrap(dotPartialMem.get()), wrap(uNormPartialMem.get()), wrap(zNormPartialMem.get()), this->totalPixels)
                                               .build(),
                                           cl::NDRange(), cl::NDRange(globalSizePartials), cl::NDRange(corrPartialLocalSize));
                // reduce partials and compute correlation
                queue.enqueueNDRangeKernel(KernelBuilder(programs, "calculate_final_correlation")
                                               .args(wrap(dotPartialMem.get()), wrap(uNormPartialMem.get()), wrap(zNormPartialMem.get()), wrap(correlationResultMem.get()), workGroups)
                                               .build(),
                                           cl::NDRange(), cl::NDRange(corrFinalLocalSize), cl::NDRange(corrFinalLocalSize));
                // retrieve the correlation result
                this->unlockArrays(e_u, e_z, dotPartial, uNormPartial, zNormPartial, correlationResult);
                correlation = correlationResult.scalar<float>();
            },
            "compute correlation kernels");
        return correlation;
    }

    // helper method to sum the incomplete RxPartial and rxPartial arrays which were produced from the custom "me" kernel
    // and to transform them to the correct size, so that they can be used by the system solver
    std::pair<af::array, af::array> transformCorrelationArrays(const af::array& RxPartial, const af::array& rxPartial) const {
        // reduction sum of blocks
        // all [p^2-1,1] blocks will be summed in rx
        // all [((p^2-1)(p^2))/2] vector blocks will be summed in Rx
        const auto totalBlocks = rxPartial.elements() / this->localSize;
        const auto RxStride = RxPartial.elements() / totalBlocks;
        const af::array rx = af::sum(af::moddims(rxPartial, this->localSize, totalBlocks), 1);
        const af::array Rx = af::sum(af::moddims(RxPartial, RxStride, totalBlocks), 1);
        return std::make_pair(Rx, rx);
    }
};