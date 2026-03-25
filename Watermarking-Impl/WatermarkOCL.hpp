#pragma once
#include "AfclBuffer.hpp"
#include "include/WatermarkTypes.hpp"
#include "opencl_init.h"
#include "opencl_utils.hpp"
#include "WatermarkBase.hpp"
#include "WatermarkGpu.hpp"
#include <af/opencl.h>
#include <arrayfire.h>
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
          meKernelDims{rows, alignUp<optimalLocalSize>(cols)}, programs(cl_utils::OpenCLKernelCache<p>::getProgram()) {}

  private:
    using WatermarkBase::alignUp;

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
        const AfclBuffer inputGrayBuf(inputGrayImage);
        const AfclBuffer inputBuf(inputImage);
        const AfclBuffer randBuf(this->randomMatrix);
        const AfclBuffer uBuf(inputGrayImage.dims(), f32);
        const AfclBuffer sumSqBuf(1, f32);
        AfclBuffer outputBuf(inputImage.dims(), u8);
        if (maskType == MaskMethod::NVF) {
            const int workGroups = static_cast<int>((this->texKernelDims.first / windowLocalSize.first) * (this->texKernelDims.second / windowLocalSize.second));
            const AfclBuffer partialsBuf(workGroups, f32);
            executeKernel(
                [&]() {
                    // fused kernel to compute NVF mask, strengthened watermark (u) and sum of squares of u
                    queue.enqueueNDRangeKernel(
                        KernelBuilder(programs, "nvf_u_and_partial_sumsq_fused").args(inputGrayBuf.get(), randBuf.get(), uBuf.get(), partialsBuf.get(), this->baseCols, this->baseRows).build(),
                        cl::NDRange(), cl::NDRange(texKernelDims.first, texKernelDims.second), cl::NDRange(windowLocalSize.first, windowLocalSize.second));
                    // reduce the partial sums (single workgroup)
                    queue.enqueueNDRangeKernel(KernelBuilder(programs, "reduce_sumsq_partials").args(partialsBuf.get(), sumSqBuf.get(), workGroups).build(), cl::NDRange(),
                                               cl::NDRange(optimalLocalSize), cl::NDRange(optimalLocalSize));
                    // apply watermark
                    const int workGroupsApply = calculateLocalGroupsNumber(this->totalPixels, optimalLocalSize);
                    const int globalSizeApply = workGroupsApply * optimalLocalSize;
                    queue.enqueueNDRangeKernel(KernelBuilder(programs, "apply_watermark_fused")
                                                   .args(inputBuf.get(), uBuf.get(), sumSqBuf.get(), outputBuf.get(), this->strengthNumerator, this->totalPixels, static_cast<int>(inputImage.dims(2)))
                                                   .build(),
                                               cl::NDRange(), cl::NDRange(globalSizeApply), cl::NDRange(optimalLocalSize));
                },
                "NVF_computeStrengthenedWatermark");
        } else {
            // compute prediction error
            const AfclBuffer errorSeqBuf(computePredictionErrorData(inputGrayImage, true));
            const int maxWorkGroups = calculateLocalGroupsNumber(this->totalPixels, optimalLocalSize);
            const int maxGlobalSize = maxWorkGroups * optimalLocalSize;
            const AfclBuffer errorSeqMaxBuf(1, f32);
            const AfclBuffer maxPartialsBuf(maxWorkGroups, f32);
            const AfclBuffer partialsBuf(maxWorkGroups, f32);
            executeKernel(
                [&]() {
                    // compute max error sequence partials
                    queue.enqueueNDRangeKernel(KernelBuilder(programs, "partial_max_reduce").args(errorSeqBuf.get(), maxPartialsBuf.get(), this->totalPixels).build(), cl::NDRange(),
                                               cl::NDRange(maxGlobalSize), cl::NDRange(optimalLocalSize));
                    // compute final error sequence max
                    queue.enqueueNDRangeKernel(KernelBuilder(programs, "final_max_reduce").args(maxPartialsBuf.get(), errorSeqMaxBuf.get(), maxWorkGroups).build(), cl::NDRange(),
                                               cl::NDRange(optimalLocalSize), cl::NDRange(optimalLocalSize));
                    // fused kernel to compute ME mask, strengthened watermark (u) and sum of squares of u
                    queue.enqueueNDRangeKernel(
                        KernelBuilder(programs, "me_u_and_partial_sumsq_fused").args(errorSeqBuf.get(), randBuf.get(), uBuf.get(), partialsBuf.get(), errorSeqMaxBuf.get(), this->totalPixels).build(),
                        cl::NDRange(), cl::NDRange(maxGlobalSize), cl::NDRange(optimalLocalSize));
                    // reduce sumsq partials
                    queue.enqueueNDRangeKernel(KernelBuilder(programs, "reduce_sumsq_partials").args(partialsBuf.get(), sumSqBuf.get(), maxWorkGroups).build(), cl::NDRange(),
                                               cl::NDRange(optimalLocalSize), cl::NDRange(optimalLocalSize));
                    // apply watermark
                    queue.enqueueNDRangeKernel(KernelBuilder(programs, "apply_watermark_fused")
                                                   .args(inputBuf.get(), uBuf.get(), sumSqBuf.get(), outputBuf.get(), this->strengthNumerator, this->totalPixels, static_cast<int>(inputImage.dims(2)))
                                                   .build(),
                                               cl::NDRange(), cl::NDRange(maxGlobalSize), cl::NDRange(optimalLocalSize));
                },
                "ME_computeStrengthenedWatermark");
        }
        outputBuf.unlock();
        return outputBuf.getArray();
    }

    af::array computeCustomMask(const af::array& image) const override {
        using namespace cl_utils;
        const AfclBuffer imageBuf(image);
        AfclBuffer customMaskBuf(this->baseRows, this->baseCols, f32);
        // transposed global dimensions because of column-major order in arrayfire
        executeKernel(
            [&]() {
                queue.enqueueNDRangeKernel(KernelBuilder(programs, "nvf").args(imageBuf.get(), customMaskBuf.get(), this->baseCols, this->baseRows).build(), cl::NDRange(),
                                           cl::NDRange(texKernelDims.first, texKernelDims.second), cl::NDRange(windowLocalSize.first, windowLocalSize.second));
            },
            "nvf");
        customMaskBuf.unlock();
        return customMaskBuf.getArray();
    }

    af::array computeErrorSequence(const af::array& image, const bool calculateAbs) const override {
        using namespace cl_utils;
        const AfclBuffer imageBuf(image);
        const AfclBuffer coeffsBuf(this->coefficients);
        const AfclBuffer stopFlagBuf(this->stopFlag);
        AfclBuffer errorSequenceBuf(this->baseRows, this->baseCols, f32);
        // transposed global dimensions because of column-major order in arrayfire
        executeKernel(
            [&]() {
                queue.enqueueNDRangeKernel(KernelBuilder(programs, "error_sequence")
                                               .args(imageBuf.get(), errorSequenceBuf.get(), coeffsBuf.get(), this->baseCols, this->baseRows, static_cast<int>(calculateAbs), stopFlagBuf.get())
                                               .build(),
                                           cl::NDRange(), cl::NDRange(texKernelDims.first, texKernelDims.second), cl::NDRange(windowLocalSize.first, windowLocalSize.second));
            },
            "error_sequence");
        errorSequenceBuf.unlock();
        return errorSequenceBuf.getArray();
    }

    af::array computePredictionErrorData(const af::array& image, const bool calculateAbs) const override {
        using namespace cl_utils;
        constexpr int RxSize = (this->localSize * (this->localSize + 1)) / 2;
        constexpr int rxSize = this->localSize;
        constexpr size_t RxGlobalX = ((RxSize + rxReduceLocalSize - 1) / rxReduceLocalSize) * rxReduceLocalSize;
        constexpr size_t rxGlobalX = ((rxSize + rxReduceLocalSize - 1) / rxReduceLocalSize) * rxReduceLocalSize;
        constexpr int numChunks = 256;
        return executeKernel(
            [&]() -> af::array {
                const auto meArraysBaseWidth = meKernelDims.second / optimalLocalSize;
                const int totalBlocks = this->baseRows * meArraysBaseWidth;
                const int blocksPerChunk = (totalBlocks + numChunks - 1) / numChunks;
                const AfclBuffer imageBuf(image);
                const AfclBuffer coeffsBuf(this->coefficients);
                const AfclBuffer stopFlagBuf(this->stopFlag);
                AfclBuffer RxPartialBuf(this->baseRows, meArraysBaseWidth * RxSize, f32);
                AfclBuffer rxPartialBuf(this->baseRows, meArraysBaseWidth * rxSize, f32);
                AfclBuffer RxPartialsTempBuf(numChunks * RxSize, f32);
                AfclBuffer rxPartialsTempBuf(numChunks * rxSize, f32);
                AfclBuffer RxBuf(RxSize, f32);
                AfclBuffer rxBuf(rxSize, f32);
                // call prediction error Rx/rx partials calculation kernel
                queue.enqueueNDRangeKernel(KernelBuilder(programs, "me").args(imageBuf.get(), RxPartialBuf.get(), rxPartialBuf.get(), this->baseCols, this->baseRows).build(), cl::NDRange(),
                                           cl::NDRange(meKernelDims.second, meKernelDims.first), cl::NDRange(optimalLocalSize, 1));
                // call Rx and rx partial reduce
                queue.enqueueNDRangeKernel(KernelBuilder(programs, "partial_reduce").args(RxPartialBuf.get(), RxPartialsTempBuf.get(), RxSize, totalBlocks, blocksPerChunk).build(), cl::NDRange(),
                                           cl::NDRange(RxGlobalX, numChunks), cl::NDRange(rxReduceLocalSize, 1));
                queue.enqueueNDRangeKernel(KernelBuilder(programs, "partial_reduce").args(rxPartialBuf.get(), rxPartialsTempBuf.get(), rxSize, totalBlocks, blocksPerChunk).build(), cl::NDRange(),
                                           cl::NDRange(rxGlobalX, numChunks), cl::NDRange(rxReduceLocalSize, 1));
                // call final Rx and rx reduce, 1 workgroup per coefficient
                queue.enqueueNDRangeKernel(KernelBuilder(programs, "final_reduce").args(RxPartialsTempBuf.get(), RxBuf.get(), numChunks, RxSize).build(), cl::NDRange(),
                                           cl::NDRange(RxSize * optimalLocalSize), cl::NDRange(optimalLocalSize));
                queue.enqueueNDRangeKernel(KernelBuilder(programs, "final_reduce").args(rxPartialsTempBuf.get(), rxBuf.get(), numChunks, rxSize).build(), cl::NDRange(),
                                           cl::NDRange(rxSize * optimalLocalSize), cl::NDRange(optimalLocalSize));
                // return the partial buffers to the arrayfire pool (they may be huge for large p)
                AfclBuffer::unlockArrays(RxPartialBuf, rxPartialBuf, RxPartialsTempBuf, rxPartialsTempBuf);
                // calculation of coefficients
                queue.enqueueNDRangeKernel(KernelBuilder(programs, "cholesky_solver").args(RxBuf.get(), rxBuf.get(), coeffsBuf.get(), stopFlagBuf.get()).build(), cl::NDRange(),
                                           cl::NDRange(choleskyLocalSize), cl::NDRange(choleskyLocalSize));
                // unlock the remaining buffers, optinal but let's help arrayfire manage the memory better
                AfclBuffer::unlockArrays(RxBuf, rxBuf);
                // calculation of error sequence which use the coefficients we just computed
                return computeErrorSequence(image, calculateAbs);
            },
            "me");
    }

    af::array computePredictionErrorMask(const af::array& errorSequence) const {
        using namespace cl_utils;
        const int workGroups = calculateLocalGroupsNumber(this->totalPixels, optimalLocalSize);
        const int globalSize = workGroups * optimalLocalSize;
        const AfclBuffer errSeqBuf(errorSequence);
        const AfclBuffer maxValBuf(1, f32);
        const AfclBuffer partialMaxBuf(workGroups, f32);
        AfclBuffer maskBuf(errorSequence.dims(), f32);
        executeKernel(
            [&]() {
                // transform to abs and find "partial maxes"
                queue.enqueueNDRangeKernel(KernelBuilder(programs, "reduce_abs_max_partials").args(errSeqBuf.get(), partialMaxBuf.get(), this->totalPixels).build(), cl::NDRange(),
                                           cl::NDRange(globalSize), cl::NDRange(optimalLocalSize));

                // final max reduction
                queue.enqueueNDRangeKernel(KernelBuilder(programs, "final_max_reduce").args(partialMaxBuf.get(), maxValBuf.get(), workGroups).build(), cl::NDRange(), cl::NDRange(optimalLocalSize),
                                           cl::NDRange(optimalLocalSize));

                // normalize mask
                queue.enqueueNDRangeKernel(KernelBuilder(programs, "compute_abs_normalized_mask").args(errSeqBuf.get(), maskBuf.get(), maxValBuf.get(), this->totalPixels).build(), cl::NDRange(),
                                           cl::NDRange(globalSize), cl::NDRange(optimalLocalSize));
            },
            "compute_prediction_error_mask_fused");
        maskBuf.unlock();
        return maskBuf.getArray();
    }

    float computeCorrelation(const af::array& e_u, const af::array& mask) const override {
        using namespace cl_utils;
        const int workGroups = static_cast<int>((texKernelDims.first / windowLocalSize.first) * (texKernelDims.second / windowLocalSize.second));
        const AfclBuffer maskBuf(mask);
        const AfclBuffer wBuf(this->randomMatrix);
        const AfclBuffer euBuf(e_u);
        const AfclBuffer coeffsBuf(this->coefficients);
        const AfclBuffer stopFlagBuf(this->stopFlag);
        const AfclBuffer dotPartialBuf(workGroups, f32);
        const AfclBuffer uNormPartialBuf(workGroups, f32);
        const AfclBuffer zNormPartialBuf(workGroups, f32);
        AfclBuffer corrResultBuf(1, f32);
        executeKernel(
            [&]() {
                // launch fused rrror sequence + partial correlation
                queue.enqueueNDRangeKernel(KernelBuilder(programs, "calculate_error_sequence_and_partial_corr_fused")
                                               .args(maskBuf.get(), wBuf.get(), euBuf.get(), coeffsBuf.get(), dotPartialBuf.get(), uNormPartialBuf.get(), zNormPartialBuf.get(), this->baseCols,
                                                     this->baseRows, 0, stopFlagBuf.get())
                                               .build(),
                                           cl::NDRange(), cl::NDRange(this->texKernelDims.first, this->texKernelDims.second), cl::NDRange(windowLocalSize.first, windowLocalSize.second));
                // reduce partials and compute correlation
                queue.enqueueNDRangeKernel(
                    KernelBuilder(programs, "calculate_final_correlation").args(dotPartialBuf.get(), uNormPartialBuf.get(), zNormPartialBuf.get(), corrResultBuf.get(), workGroups).build(),
                    cl::NDRange(), cl::NDRange(corrFinalLocalSize), cl::NDRange(corrFinalLocalSize));
            },
            "compute correlation kernels");
        // retrieve the correlation result back to host and return it
        return corrResultBuf.scalar<float>();
    }
};