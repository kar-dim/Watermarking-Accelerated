#pragma once
#include "include/WatermarkTypes.hpp"
#include "opencl_init.h"
#include "opencl_utils.hpp"
#include "WatermarkBase.hpp"
#include "WatermarkGpu.hpp"
#include <af/opencl.h>
#include <arrayfire.h>
#include <memory>
#include <string>
#include <utility>

/*!
 *  \brief  Functions for watermark computation and detection, OpenCL implementation.
 *  \author Dimitris Karatzas
 */
template <int p>
class WatermarkOCL final : public WatermarkGPU<p> {
  public:
    WatermarkOCL<p>(const unsigned int rows, const unsigned int cols, const std::string& watermarkPassword, const float psnr)
        : WatermarkGPU<p>(rows, cols, watermarkPassword, psnr), texKernelDims{alignUp<windowLocalSize.first>(rows), alignUp<windowLocalSize.second>(cols)},
          meKernelDims{rows, alignUp<optimalLocalSize>(cols)}, programs(cl_utils::buildKernels(p)) {}

  private:
    using WatermarkBase::alignUp;
    using clMemPtr = std::unique_ptr<cl_mem>;

    static constexpr unsigned int optimalLocalSize = 256; // safe universal local size for OpenCL, used almost anywhere
    static constexpr unsigned int rxReduceLocalSize = 64;
    static constexpr std::pair windowLocalSize = {32, 8};
    static constexpr unsigned int choleskyLocalSize = p < 7 ? 1 : 64; // for p >= 7 we use 64-thread cholesky solver, for p < 7 single thread (faster for small p)

    cl::Context context{afcl::getContext(true)};
    cl::CommandQueue queue{afcl::getQueue(true)};
    cl::Device device{afcl::getDeviceId(), true};
    std::pair<int, int> texKernelDims, meKernelDims;
    unsigned int corrFinalLocalSize = cl_utils::maxPow2WorkGroupSize(device); // we could use the safe universal of 256, but the kernel that uses this benefits from larger local sizes
    cl::Program programs;

    af::array computeStrengthenedWatermark(const af::array& inputGrayImage, const af::array& inputImage, const MaskMethod maskType) const override {
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

        if (maskType == MaskMethod::NVF) {
            const int workGroups = static_cast<int>((this->texKernelDims.first / windowLocalSize.first) * (this->texKernelDims.second / windowLocalSize.second));
            const af::array partials(workGroups, f32);
            const clMemPtr partialsMem(partials.device<cl_mem>());

            executeKernel(
                [&]() {
                    // fused kernel to compute NVF mask, strengthened watermark (u) and sum of squares of u
                    queue.enqueueNDRangeKernel(KernelBuilder(programs, "nvf_u_and_partial_sumsq_fused")
                                                   .args(wrap(inputGrayMem.get()), wrap(randMem.get()), wrap(uMem.get()), wrap(partialsMem.get()), this->baseCols, this->baseRows)
                                                   .build(),
                                               cl::NDRange(), cl::NDRange(texKernelDims.first, texKernelDims.second), cl::NDRange(windowLocalSize.first, windowLocalSize.second));

                    // reduce the partial sums (single workgroup)
                    queue.enqueueNDRangeKernel(KernelBuilder(programs, "reduce_sumsq_partials").args(wrap(partialsMem.get()), wrap(sumSqMem.get()), workGroups).build(), cl::NDRange(),
                                               cl::NDRange(optimalLocalSize), cl::NDRange(optimalLocalSize));

                    // apply watermark
                    const int workGroupsApply = calculateLocalGroupsNumber(this->totalPixels, optimalLocalSize);
                    const int globalSizeApply = workGroupsApply * optimalLocalSize;

                    queue.enqueueNDRangeKernel(
                        KernelBuilder(programs, "apply_watermark_fused")
                            .args(wrap(inputMem.get()), wrap(uMem.get()), wrap(sumSqMem.get()), wrap(outputMem.get()), this->strengthNumerator, this->totalPixels, static_cast<int>(inputImage.dims(2)))
                            .build(),
                        cl::NDRange(), cl::NDRange(globalSizeApply), cl::NDRange(optimalLocalSize));

                    this->unlockArrays(inputGrayImage, partials);
                },
                "NVF_computeStrengthenedWatermark");

        } else {
            // find max of error sequence, this cannot be fused because it is a global reduction
            const af::array errorSeq = computePredictionErrorData(inputGrayImage, true);
            const af::array errorSeqMax = af::max(af::flat(errorSeq));

            // fused kernel to compute ME mask, strengthened watermark (u) and sum of squares of u
            const int workGroups = calculateLocalGroupsNumber(this->totalPixels, optimalLocalSize);
            const int globalSize = workGroups * optimalLocalSize;

            const af::array partials(workGroups, f32);
            const clMemPtr partialsMem(partials.device<cl_mem>());
            const clMemPtr errorSeqMem(errorSeq.device<cl_mem>());
            const clMemPtr errorSeqMaxMem(errorSeqMax.device<cl_mem>());

            executeKernel(
                [&]() {
                    queue.enqueueNDRangeKernel(KernelBuilder(programs, "me_u_and_partial_sumsq_fused")
                                                   .args(wrap(errorSeqMem.get()), wrap(randMem.get()), wrap(uMem.get()), wrap(partialsMem.get()), wrap(errorSeqMaxMem.get()), this->totalPixels)
                                                   .build(),
                                               cl::NDRange(), cl::NDRange(globalSize), cl::NDRange(optimalLocalSize));

                    queue.enqueueNDRangeKernel(KernelBuilder(programs, "reduce_sumsq_partials").args(wrap(partialsMem.get()), wrap(sumSqMem.get()), workGroups).build(), cl::NDRange(),
                                               cl::NDRange(optimalLocalSize), cl::NDRange(optimalLocalSize));

                    // apply watermark
                    queue.enqueueNDRangeKernel(
                        KernelBuilder(programs, "apply_watermark_fused")
                            .args(wrap(inputMem.get()), wrap(uMem.get()), wrap(sumSqMem.get()), wrap(outputMem.get()), this->strengthNumerator, this->totalPixels, static_cast<int>(inputImage.dims(2)))
                            .build(),
                        cl::NDRange(), cl::NDRange(globalSize), cl::NDRange(optimalLocalSize));

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
                                           cl::NDRange(texKernelDims.first, texKernelDims.second), cl::NDRange(windowLocalSize.first, windowLocalSize.second));
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
                    cl::NDRange(), cl::NDRange(texKernelDims.first, texKernelDims.second), cl::NDRange(windowLocalSize.first, windowLocalSize.second));
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
                    cl::NDRange(), cl::NDRange(texKernelDims.first, texKernelDims.second), cl::NDRange(windowLocalSize.first, windowLocalSize.second));
                this->unlockArrays(inputA, inputB, errorSequence, this->coefficients, this->stopFlag);
            },
            "error_sequence_fused");
        return errorSequence;
    }

    af::array computePredictionErrorData(const af::array& image, const bool calculateAbs) const override {
        using namespace cl_utils;
        //"me" kernel constants
        constexpr int RxSize = (this->localSize * (this->localSize + 1)) / 2;
        constexpr int rxSize = this->localSize;
        // reduce Rx/rx kernel constants
        constexpr size_t RxGlobalX = ((RxSize + rxReduceLocalSize - 1) / rxReduceLocalSize) * rxReduceLocalSize;
        constexpr size_t rxGlobalX = ((rxSize + rxReduceLocalSize - 1) / rxReduceLocalSize) * rxReduceLocalSize;
        constexpr int numChunks = 256;

        return executeKernel(
            [&]() -> af::array {
                const auto meArraysBaseWidth = meKernelDims.second / optimalLocalSize;
                const af::array RxPartial(this->baseRows, meArraysBaseWidth * RxSize);
                const af::array rxPartial(this->baseRows, meArraysBaseWidth * rxSize);
                const clMemPtr RxPartialMem(RxPartial.device<cl_mem>());
                const clMemPtr rxPartialMem(rxPartial.device<cl_mem>());
                const clMemPtr imageMem(image.device<cl_mem>());

                // call prediction error Rx/rx partials calculation kernel
                queue.enqueueNDRangeKernel(KernelBuilder(programs, "me").args(wrap(imageMem.get()), wrap(RxPartialMem.get()), wrap(rxPartialMem.get()), this->baseCols, this->baseRows).build(),
                                           cl::NDRange(), cl::NDRange(meKernelDims.second, meKernelDims.first), cl::NDRange(optimalLocalSize, 1));

                // setup and call reductions of the Rx/rx partials
                const int totalBlocks = this->baseRows * meArraysBaseWidth;
                const int blocksPerChunk = (totalBlocks + numChunks - 1) / numChunks;
                // intermediate partial arrays for partial sums and final arrays
                const af::array RxPartialsTemp(numChunks * RxSize);
                const af::array rxPartialsTemp(numChunks * rxSize);
                const af::array Rx(RxSize);
                const af::array rx(rxSize);
                const clMemPtr RxPartialsTempMem(RxPartialsTemp.device<cl_mem>());
                const clMemPtr rxPartialsTempMem(rxPartialsTemp.device<cl_mem>());
                const clMemPtr RxMem(Rx.device<cl_mem>());
                const clMemPtr rxMem(rx.device<cl_mem>());

                // call Rx/rx partial reduces
                queue.enqueueNDRangeKernel(KernelBuilder(programs, "partial_reduce").args(wrap(RxPartialMem.get()), wrap(RxPartialsTempMem.get()), RxSize, totalBlocks, blocksPerChunk).build(),
                                           cl::NDRange(), cl::NDRange(RxGlobalX, numChunks), cl::NDRange(rxReduceLocalSize, 1));

                queue.enqueueNDRangeKernel(KernelBuilder(programs, "partial_reduce").args(wrap(rxPartialMem.get()), wrap(rxPartialsTempMem.get()), rxSize, totalBlocks, blocksPerChunk).build(),
                                           cl::NDRange(), cl::NDRange(rxGlobalX, numChunks), cl::NDRange(rxReduceLocalSize, 1));

                // call final Rx/rx reduces, 1 workgroup per coefficient (256 threads each)
                queue.enqueueNDRangeKernel(KernelBuilder(programs, "final_reduce").args(wrap(RxPartialsTempMem.get()), wrap(RxMem.get()), numChunks, RxSize).build(), cl::NDRange(),
                                           cl::NDRange(RxSize * optimalLocalSize), cl::NDRange(optimalLocalSize));

                queue.enqueueNDRangeKernel(KernelBuilder(programs, "final_reduce").args(wrap(rxPartialsTempMem.get()), wrap(rxMem.get()), numChunks, rxSize).build(), cl::NDRange(),
                                           cl::NDRange(rxSize * optimalLocalSize), cl::NDRange(optimalLocalSize));

                // return memory to arrayfire
                this->unlockArrays(image, RxPartial, rxPartial, RxPartialsTemp, rxPartialsTemp);

                // calculation of coefficients
                const clMemPtr coeffsMem(this->coefficients.template device<cl_mem>());
                const clMemPtr stopFlagMem(this->stopFlag.template device<cl_mem>());

                queue.enqueueNDRangeKernel(KernelBuilder(programs, "cholesky_solver").args(wrap(RxMem.get()), wrap(rxMem.get()), wrap(coeffsMem.get()), wrap(stopFlagMem.get())).build(), cl::NDRange(),
                                           cl::NDRange(choleskyLocalSize), cl::NDRange(choleskyLocalSize));
                // return memory to arrayfire
                this->unlockArrays(Rx, rx, this->coefficients, this->stopFlag);
                // calculation of error sequence which use the coefficients we just computed
                return computeErrorSequence(image, calculateAbs);
            },
            "me");
    }

    af::array computePredictionErrorMask(const af::array& errorSequence) const {
        using namespace cl_utils;
        const int workGroups = calculateLocalGroupsNumber(this->totalPixels, optimalLocalSize);
        const int globalSize = workGroups * optimalLocalSize;
        const af::array mask(errorSequence.dims(), f32);
        const af::array maxVal(1, f32);
        const af::array partialMax(workGroups, f32);
        const clMemPtr errMem(errorSequence.device<cl_mem>());
        const clMemPtr maskMem(mask.device<cl_mem>());
        const clMemPtr maxValMem(maxVal.device<cl_mem>());
        const clMemPtr partialMaxMem(partialMax.device<cl_mem>());

        executeKernel(
            [&]() {
                // transform to abs and find "partial maxes"
                queue.enqueueNDRangeKernel(KernelBuilder(programs, "reduce_abs_max_partials").args(wrap(errMem.get()), wrap(partialMaxMem.get()), this->totalPixels).build(), cl::NDRange(),
                                           cl::NDRange(globalSize), cl::NDRange(optimalLocalSize));
                // final max reduction
                queue.enqueueNDRangeKernel(KernelBuilder(programs, "reduce_abs_max_final").args(wrap(partialMaxMem.get()), wrap(maxValMem.get()), workGroups).build(), cl::NDRange(),
                                           cl::NDRange(optimalLocalSize), cl::NDRange(optimalLocalSize));
                // normalize mask
                queue.enqueueNDRangeKernel(KernelBuilder(programs, "compute_abs_normalized_mask").args(wrap(errMem.get()), wrap(maskMem.get()), wrap(maxValMem.get()), this->totalPixels).build(),
                                           cl::NDRange(), cl::NDRange(globalSize), cl::NDRange(optimalLocalSize));
                this->unlockArrays(errorSequence, mask, maxVal, partialMax);
            },
            "compute_prediction_error_mask_fused");

        return mask;
    }

    float computeCorrelation(const af::array& e_u, const af::array& mask) const override {
        using namespace cl_utils;
        const int workGroups = static_cast<int>((texKernelDims.first / windowLocalSize.first) * (texKernelDims.second / windowLocalSize.second));
        const af::array dotPartial(workGroups, f32);
        const af::array uNormPartial(workGroups, f32);
        const af::array zNormPartial(workGroups, f32);
        const af::array correlationResult(1, f32);
        const clMemPtr maskMem(mask.device<cl_mem>());
        const clMemPtr randMem(this->randomMatrix.template device<cl_mem>());
        const clMemPtr euMem(e_u.device<cl_mem>());
        const clMemPtr coeffsMem(this->coefficients.template device<cl_mem>());
        const clMemPtr stopFlagMem(this->stopFlag.template device<cl_mem>());
        const clMemPtr dotPartialMem(dotPartial.device<cl_mem>());
        const clMemPtr uNormPartialMem(uNormPartial.device<cl_mem>());
        const clMemPtr zNormPartialMem(zNormPartial.device<cl_mem>());
        const clMemPtr correlationResultMem(correlationResult.device<cl_mem>());
        float correlation = 0.0f;
        executeKernel(
            [&]() {
                // launch fused rrror sequence + partial correlation
                queue.enqueueNDRangeKernel(KernelBuilder(programs, "calculate_error_sequence_and_partial_corr_fused")
                                               .args(wrap(maskMem.get()), wrap(randMem.get()), wrap(euMem.get()), wrap(coeffsMem.get()), wrap(dotPartialMem.get()), wrap(uNormPartialMem.get()),
                                                     wrap(zNormPartialMem.get()), this->baseCols, this->baseRows, 0, wrap(stopFlagMem.get()))
                                               .build(),
                                           cl::NDRange(), cl::NDRange(this->texKernelDims.first, this->texKernelDims.second), cl::NDRange(windowLocalSize.first, windowLocalSize.second));
                // reduce partials and compute correlation
                queue.enqueueNDRangeKernel(KernelBuilder(programs, "calculate_final_correlation")
                                               .args(wrap(dotPartialMem.get()), wrap(uNormPartialMem.get()), wrap(zNormPartialMem.get()), wrap(correlationResultMem.get()), workGroups)
                                               .build(),
                                           cl::NDRange(), cl::NDRange(corrFinalLocalSize), cl::NDRange(corrFinalLocalSize));
                // retrieve the correlation result
                this->unlockArrays(mask, e_u, dotPartial, uNormPartial, zNormPartial, correlationResult, this->coefficients, this->stopFlag, this->randomMatrix);
                correlation = correlationResult.scalar<float>();
            },
            "compute correlation kernels");
        return correlation;
    }
};