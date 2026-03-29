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
          programs(cl_utils::OpenCLKernelCache<p>::getProgram(af::getDevice())) {
        // calculate optimal grid size for ME kernel based on the number of SMs on the GPU
        gridOptimalMe = cl_utils::gridSizeMeCalculate(device, cl_utils::KernelBuilder(programs, "me").build());
        constexpr unsigned int pixelsPerBlockX = optimalLocalSize;
        const unsigned int meTotalBlocksX = alignUp<pixelsPerBlockX>(this->baseCols) / pixelsPerBlockX;
        meParams = {meTotalBlocksX, meTotalBlocksX * this->baseRows};
    }

  private:
    using WatermarkBase::alignUp;

    static constexpr unsigned int optimalLocalSize = 256; // safe universal local size for OpenCL, used almost anywhere
    static constexpr std::pair windowLocalSize = {32, 8};
    static constexpr unsigned int choleskyLocalSize = p < 7 ? 1 : 64; // for p >= 7 we use 64-thread cholesky solver, for p < 7 single thread (faster for small p)

    cl::Context context{afcl::getContext(true)};
    cl::CommandQueue queue{afcl::getQueue(true)};
    cl::Device device{afcl::getDeviceId(), true};
    std::pair<int, int> texKernelDims;
    unsigned int gridOptimalMe;
    std::pair<int, int> meParams;
    unsigned int corrFinalLocalSize = cl_utils::maxPow2WorkGroupSize(device); // we could use the safe universal of 256, but the kernel that uses this benefits from larger local sizes
    cl::Program programs;

    af::array computeStrengthenedWatermark(const af::array& inputGrayImage, const af::array& inputImage, const MaskMethod maskType) const override {
        using namespace cl_utils;
        const int maxWorkGroups = calculateLocalGroupsNumber(this->totalPixels, optimalLocalSize);
        const int maxGlobalSize = maxWorkGroups * optimalLocalSize;
        const AfclBuffer inputGrayBuf(inputGrayImage);
        const AfclBuffer inputBuf(inputImage);
        const AfclBuffer randBuf(this->randomMatrix);
        const AfclBuffer uBuf(inputGrayImage.dims(), f32);
        const AfclBuffer sumSqBuf(af::constant(0, 1, u64));
        AfclBuffer outputBuf(inputImage.dims(), u8);
        executeKernel(
            [&]() {
                if (maskType == MaskMethod::NVF) {
                    // fused kernel to compute NVF mask, strengthened watermark (u) and sum of squares of u
                    queue.enqueueNDRangeKernel(
                        KernelBuilder(programs, "nvf_u_and_sumsq_fused").args(inputGrayBuf.get(), randBuf.get(), uBuf.get(), sumSqBuf.get(), this->baseCols, this->baseRows).build(), cl::NDRange(),
                        cl::NDRange(texKernelDims.first, texKernelDims.second), cl::NDRange(windowLocalSize.first, windowLocalSize.second));

                } else {
                    // compute prediction error
                    const AfclBuffer errorSeqBuf(computePredictionErrorData(inputGrayImage, true));
                    const AfclBuffer errorSeqMaxBuf(1, f32);
                    const AfclBuffer maxPartialsBuf(maxWorkGroups, f32);

                    // compute max error sequence partials
                    queue.enqueueNDRangeKernel(KernelBuilder(programs, "partial_max_reduce").args(errorSeqBuf.get(), maxPartialsBuf.get(), this->totalPixels).build(), cl::NDRange(),
                                               cl::NDRange(maxGlobalSize), cl::NDRange(optimalLocalSize));
                    // compute final error sequence max
                    queue.enqueueNDRangeKernel(KernelBuilder(programs, "final_max_reduce").args(maxPartialsBuf.get(), errorSeqMaxBuf.get(), maxWorkGroups).build(), cl::NDRange(),
                                               cl::NDRange(optimalLocalSize), cl::NDRange(optimalLocalSize));
                    // fused kernel to compute ME mask, strengthened watermark (u) and sum of squares of u
                    queue.enqueueNDRangeKernel(
                        KernelBuilder(programs, "me_u_and_sumsq_fused").args(errorSeqBuf.get(), randBuf.get(), uBuf.get(), sumSqBuf.get(), errorSeqMaxBuf.get(), this->totalPixels).build(),
                        cl::NDRange(), cl::NDRange(maxGlobalSize), cl::NDRange(optimalLocalSize));
                }
                // apply watermark
                queue.enqueueNDRangeKernel(KernelBuilder(programs, "apply_watermark_fused")
                                               .args(inputBuf.get(), uBuf.get(), sumSqBuf.get(), outputBuf.get(), this->strengthNumerator, this->totalPixels, static_cast<int>(inputImage.dims(2)))
                                               .build(),
                                           cl::NDRange(), cl::NDRange(maxGlobalSize), cl::NDRange(optimalLocalSize));
            },
            "computeStrengthenedWatermark");
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
        return executeKernel(
            [&]() -> af::array {
                const AfclBuffer imageBuf(image);
                const AfclBuffer coeffsBuf(this->coefficients);
                const AfclBuffer stopFlagBuf(this->stopFlag);
                const AfclBuffer RxBuf(af::constant(0, RxSize, u64));
                const AfclBuffer rxBuf(af::constant(0, rxSize, u64));
                // call prediction error Rx/rx matrices calculation kernel
                queue.enqueueNDRangeKernel(KernelBuilder(programs, "me").args(imageBuf.get(), RxBuf.get(), rxBuf.get(), this->baseCols, this->baseRows, meParams.first, meParams.second).build(),
                                           cl::NDRange(), cl::NDRange(gridOptimalMe * optimalLocalSize), cl::NDRange(optimalLocalSize));
                // calculation of coefficients
                queue.enqueueNDRangeKernel(KernelBuilder(programs, "cholesky_solver").args(RxBuf.get(), rxBuf.get(), coeffsBuf.get(), stopFlagBuf.get()).build(), cl::NDRange(),
                                           cl::NDRange(choleskyLocalSize), cl::NDRange(choleskyLocalSize));
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