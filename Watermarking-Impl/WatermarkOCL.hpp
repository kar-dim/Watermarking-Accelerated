#pragma once
#include "buffer.hpp"
#include "include/WatermarkTypes.hpp"
#include "OclArray.hpp"
#include "OclQueueManager.hpp"
#include "opencl_init.h"
#include "opencl_utils.hpp"
#include "WatermarkBase.hpp"
#include <cmath>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

/*!
 *  \brief  Functions for watermark computation and detection, OpenCL implementation.
 *          Local OclArray allocations are pool-backed (OclMemPool), no OpenCL driver allocation overhead
 *  \author Dimitris Karatzas
 */
template <int p>
class WatermarkOCL final : public WatermarkBase {
  public:
    WatermarkOCL<p>(const int rows, const int cols, const std::string& watermarkPassword, const float psnr)
        : WatermarkBase(rows, cols, watermarkPassword, psnr, initializeRandomMatrix), strengthNumerator(strengthFactor * std::sqrt(static_cast<float>(this->totalPixels))),
          coefficients(localSize, OclQueueManager::getInstance().getQueueRaw()), stopFlag(FlagBuffer::zeros(1, OclQueueManager::getInstance().getQueueRaw())),
          queue(OclQueueManager::getInstance().getQueue()), device(OclQueueManager::getInstance().getDevice()),
          texKernelDims{alignUp<windowLocalSize.first>(rows), alignUp<windowLocalSize.second>(cols)}, meKernelDims{rows, alignUp<optimalLocalSize>(cols)},
          corrFinalLocalSize(cl_utils::maxPow2WorkGroupSize(device)), programs(cl_utils::OpenCLKernelCache<p>::getProgram()) {}

    // Embed: compute strengthened watermark u (via NVF or ME mask), then apply to all channels of inputImage
    void makeWatermark(const ImageBuffer& inputGrayImage, const ImageBuffer& inputImage, ImageOutputBuffer& output, const MaskMethod maskType) override {
        using namespace cl_utils;
        const int maxWorkGroups = calculateLocalGroupsNumber(this->totalPixels, optimalLocalSize);
        const int maxGlobalSize = maxWorkGroups * optimalLocalSize;
        const OclArray<float> u(this->baseRows, this->baseCols, this->queue.get());
        const OclArray<uint64_t> sumSq = OclArray<uint64_t>::zeros(1, this->queue.get());
        // reuse output buffer, only (re)allocate when dimensions or channel count change
        if (output.empty() || output.getRows() != this->baseRows || output.getCols() != this->baseCols || output.getChannels() != inputImage.getChannels())
            output = ImageOutputBuffer(inputImage.getRows(), inputImage.getCols(), inputImage.getChannels(), this->queue.get());

        executeKernel(
            [&]() {
                if (maskType == MaskMethod::NVF) {
                    // fused NVF: local variance mask x watermark -> strengthened watermark u + sum(u^2)
                    queue.enqueueNDRangeKernel(KernelBuilder(programs, "nvf_u_and_sumsq_fused")
                                                   .args(inputGrayImage.clBuffer(), this->randomMatrix.clBuffer(), u.clBuffer(), sumSq.clBuffer(), this->baseCols, this->baseRows)
                                                   .build(),
                                               cl::NDRange(), cl::NDRange(texKernelDims.first, texKernelDims.second), cl::NDRange(windowLocalSize.first, windowLocalSize.second));
                } else {
                    // ME path: solve prediction error model, compute error sequence, normalize, fuse with watermark
                    const OclArray<uint64_t> Rx = OclArray<uint64_t>::zeros(RxSize, this->queue.get());
                    const OclArray<uint64_t> rx = OclArray<uint64_t>::zeros(rxSize, this->queue.get());
                    launchMeKernel(inputGrayImage.clBuffer(), Rx.clBuffer(), rx.clBuffer());
                    launchCholeskySolver(Rx.clBuffer(), rx.clBuffer());
                    const OclArray<float> errorSeq(this->baseRows, this->baseCols, this->queue.get());
                    queue.enqueueNDRangeKernel(
                        KernelBuilder(programs, "error_sequence")
                            .args(inputGrayImage.clBuffer(), errorSeq.clBuffer(), this->coefficients.clBuffer(), this->baseCols, this->baseRows, static_cast<int>(true), this->stopFlag.clBuffer())
                            .build(),
                        cl::NDRange(), cl::NDRange(texKernelDims.first, texKernelDims.second), cl::NDRange(windowLocalSize.first, windowLocalSize.second));
                    // two-pass max-reduce for normalization
                    const OclArray<float> errorSeqMax(1, this->queue.get());
                    const OclArray<float> maxPartials(maxWorkGroups, this->queue.get());
                    queue.enqueueNDRangeKernel(KernelBuilder(programs, "partial_max_reduce").args(errorSeq.clBuffer(), maxPartials.clBuffer(), this->totalPixels).build(), cl::NDRange(),
                                               cl::NDRange(maxGlobalSize), cl::NDRange(optimalLocalSize));
                    queue.enqueueNDRangeKernel(KernelBuilder(programs, "final_max_reduce").args(maxPartials.clBuffer(), errorSeqMax.clBuffer(), maxWorkGroups).build(), cl::NDRange(),
                                               cl::NDRange(optimalLocalSize), cl::NDRange(optimalLocalSize));
                    // fused ME: normalized error x watermark -> strengthened watermark u + sum(u^2)
                    queue.enqueueNDRangeKernel(KernelBuilder(programs, "me_u_and_sumsq_fused")
                                                   .args(errorSeq.clBuffer(), this->randomMatrix.clBuffer(), u.clBuffer(), sumSq.clBuffer(), errorSeqMax.clBuffer(), this->totalPixels)
                                                   .build(),
                                               cl::NDRange(), cl::NDRange(maxGlobalSize), cl::NDRange(optimalLocalSize));
                }
                // scale u by strength factor and add to each channel of the input image
                queue.enqueueNDRangeKernel(
                    KernelBuilder(programs, "apply_watermark_fused")
                        .args(inputImage.clBuffer(), u.clBuffer(), sumSq.clBuffer(), output.clBuffer(), this->strengthNumerator, this->totalPixels, inputImage.getChannels())
                        .build(),
                    cl::NDRange(), cl::NDRange(maxGlobalSize), cl::NDRange(optimalLocalSize));
            },
            "makeWatermark");
    }

    // Detect: compute prediction error, detection mask, then correlate with watermark
    float detectWatermark(const ImageBuffer& inputImage, const MaskMethod maskType) override {
        using namespace cl_utils;
        const int maxWorkGroups = calculateLocalGroupsNumber(this->totalPixels, optimalLocalSize);
        const int maxGlobalSize = maxWorkGroups * optimalLocalSize;
        const int corrWorkGroups = static_cast<int>((texKernelDims.first / windowLocalSize.first) * (texKernelDims.second / windowLocalSize.second));

        OclArray<float> corrResult(1, this->queue.get());

        executeKernel(
            [&]() {
                // solve prediction error model (Rx, rx -> coefficients via Cholesky)
                const OclArray<uint64_t> Rx = OclArray<uint64_t>::zeros(RxSize, this->queue.get());
                const OclArray<uint64_t> rx = OclArray<uint64_t>::zeros(rxSize, this->queue.get());
                launchMeKernel(inputImage.clBuffer(), Rx.clBuffer(), rx.clBuffer());
                launchCholeskySolver(Rx.clBuffer(), rx.clBuffer());

                // compute prediction error sequence (non-abs, needed for correlation sign)
                const OclArray<float> errorSeq(this->baseRows, this->baseCols, this->queue.get());
                queue.enqueueNDRangeKernel(
                    KernelBuilder(programs, "error_sequence")
                        .args(inputImage.clBuffer(), errorSeq.clBuffer(), this->coefficients.clBuffer(), this->baseCols, this->baseRows, static_cast<int>(false), this->stopFlag.clBuffer())
                        .build(),
                    cl::NDRange(), cl::NDRange(texKernelDims.first, texKernelDims.second), cl::NDRange(windowLocalSize.first, windowLocalSize.second));

                // compute detection mask (ME: abs-normalized error, NVF: local variance)
                OclArray<float> mask(this->baseRows, this->baseCols, this->queue.get());
                if (maskType == MaskMethod::ME) {
                    const OclArray<float> partialMax(maxWorkGroups, this->queue.get());
                    const OclArray<float> maxVal(1, this->queue.get());
                    queue.enqueueNDRangeKernel(KernelBuilder(programs, "reduce_abs_max_partials").args(errorSeq.clBuffer(), partialMax.clBuffer(), this->totalPixels).build(), cl::NDRange(),
                                               cl::NDRange(maxGlobalSize), cl::NDRange(optimalLocalSize));
                    queue.enqueueNDRangeKernel(KernelBuilder(programs, "final_max_reduce").args(partialMax.clBuffer(), maxVal.clBuffer(), maxWorkGroups).build(), cl::NDRange(),
                                               cl::NDRange(optimalLocalSize), cl::NDRange(optimalLocalSize));
                    queue.enqueueNDRangeKernel(KernelBuilder(programs, "compute_abs_normalized_mask").args(errorSeq.clBuffer(), mask.clBuffer(), maxVal.clBuffer(), this->totalPixels).build(),
                                               cl::NDRange(), cl::NDRange(maxGlobalSize), cl::NDRange(optimalLocalSize));
                } else {
                    queue.enqueueNDRangeKernel(KernelBuilder(programs, "nvf").args(inputImage.clBuffer(), mask.clBuffer(), this->baseCols, this->baseRows).build(), cl::NDRange(),
                                               cl::NDRange(texKernelDims.first, texKernelDims.second), cl::NDRange(windowLocalSize.first, windowLocalSize.second));
                }

                // fused: recompute error sequence from (mask * watermark), accumulate partial dot / normU / normZ
                const OclArray<float> dotPartial(corrWorkGroups, this->queue.get());
                const OclArray<float> uNormPartial(corrWorkGroups, this->queue.get());
                const OclArray<float> zNormPartial(corrWorkGroups, this->queue.get());
                queue.enqueueNDRangeKernel(KernelBuilder(programs, "calculate_error_sequence_and_partial_corr_fused")
                                               .args(mask.clBuffer(), this->randomMatrix.clBuffer(), errorSeq.clBuffer(), this->coefficients.clBuffer(), dotPartial.clBuffer(), uNormPartial.clBuffer(),
                                                     zNormPartial.clBuffer(), this->baseCols, this->baseRows, this->stopFlag.clBuffer())
                                               .build(),
                                           cl::NDRange(), cl::NDRange(texKernelDims.first, texKernelDims.second), cl::NDRange(windowLocalSize.first, windowLocalSize.second));
                // reduce partials -> final normalized correlation
                queue.enqueueNDRangeKernel(
                    KernelBuilder(programs, "calculate_final_correlation").args(dotPartial.clBuffer(), uNormPartial.clBuffer(), zNormPartial.clBuffer(), corrResult.clBuffer(), corrWorkGroups).build(),
                    cl::NDRange(), cl::NDRange(corrFinalLocalSize), cl::NDRange(corrFinalLocalSize));
            },
            "detectWatermark");

        const float correlation = corrResult.scalar();
        return std::isfinite(correlation) ? correlation : 0.0f;
    }

  private:
    using WatermarkBase::alignUp;

    static constexpr int localSize = (p * p) - 1;
    static constexpr unsigned int optimalLocalSize = 256;
    static constexpr std::pair windowLocalSize = {32, 8};
    static constexpr unsigned int choleskyLocalSize = p < 7 ? 1 : 64;
    static constexpr int RxSize = (localSize * (localSize + 1)) / 2;
    static constexpr int rxSize = localSize;

    float strengthNumerator;
    ImageBuffer coefficients;
    FlagBuffer stopFlag;

    cl::CommandQueue queue;
    cl::Device device;
    std::pair<int, int> texKernelDims, meKernelDims;
    unsigned int corrFinalLocalSize;
    cl::Program programs;

    static ImageBuffer initializeRandomMatrix(const std::vector<float>& watermarkVec, const int rows, const int cols) {
        return ImageBuffer(rows, cols, watermarkVec.data(), OclQueueManager::getInstance().getQueueRaw());
    }

    // dispatch the ME kernel for the given prediction order p
    void launchMeKernel(const cl::Buffer& image, const cl::Buffer& RxBuf, const cl::Buffer& rxBuf) const {
        queue.enqueueNDRangeKernel(cl_utils::KernelBuilder(programs, "me").args(image, RxBuf, rxBuf, this->baseCols, this->baseRows).build(), cl::NDRange(),
                                   cl::NDRange(meKernelDims.second, meKernelDims.first), cl::NDRange(optimalLocalSize));
    }

    // solve Rx*a = rx via Cholesky decomposition to get prediction coefficients
    void launchCholeskySolver(const cl::Buffer& RxBuf, const cl::Buffer& rxBuf) const {
        queue.enqueueNDRangeKernel(cl_utils::KernelBuilder(programs, "cholesky_solver").args(RxBuf, rxBuf, this->coefficients.clBuffer(), this->stopFlag.clBuffer()).build(), cl::NDRange(),
                                   cl::NDRange(choleskyLocalSize), cl::NDRange(choleskyLocalSize));
    }
};
