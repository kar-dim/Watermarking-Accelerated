#pragma once
#include "buffer.hpp"
#include "OclArray.hpp"
#include "OclQueueManager.hpp"
#include "opencl_init.h"
#include "opencl_utils.hpp"
#include "WatermarkBase.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

/*!
 *  \brief  Watermark embedding and detection, OpenCL implementation (the same steps as the CUDA one)
 *  \author Dimitris Karatzas
 */
template <int p>
class WatermarkOCL final : public WatermarkBase {
  public:
    WatermarkOCL<p>(const int rows, const int cols, WatermarkBuffer watermark, const float psnr)
        : WatermarkBase(rows, cols, std::move(watermark), psnr), queue(OclQueueManager::getInstance().getQueue()), program(cl_utils::OpenCLKernelCache<p>::getProgram()),
          groupSize(cl_utils::workGroupSize(program)) {
        using namespace cl_utils;
        const cl_command_queue rawQueue = queue.get();
        const cl::Device& device = OclQueueManager::getInstance().getDevice();
        coefficients = ImageBuffer(localSize, rawQueue);
        stopFlag = FlagBuffer::zeros(1, rawQueue);

        // shift sums grid: border runs run alongside inner shift sum workgroups
        borderBlocksPerLine = (((std::max(this->baseRows, this->baseCols) + interiorRun - 1) / interiorRun) + groupSize - 1) / groupSize;
        const int residentGroups = static_cast<int>(device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>()) * residentGroupsPerUnit;
        const int borderBlocks = borderBlocksPerLine * 2 * borderSize * shiftGroups;
        const int interiorSlots = 4 * borderBlocks <= residentGroups ? residentGroups - borderBlocks : residentGroups;
        const int interiorTasks = ((this->baseRows + interiorRun - 1) / interiorRun) * ((this->baseCols - 2 * pad + interiorCols - 1) / interiorCols);
        const int blocksPerGroup = std::min(interiorSlots / shiftGroups, (interiorTasks + groupSize - 1) / groupSize);
        const int shiftInteriorBlocks = std::max(blocksPerGroup, 1) * shiftGroups;
        shiftSumsGlobal = cl::NDRange(static_cast<size_t>(shiftInteriorBlocks + borderBlocks) * groupSize);

        // prediction error tiles: 32 x 4 rows x (groupSize / 32) x 2 columns per workgroup
        const int tilesFast = (this->baseRows + 127) / 128;
        const int tilesSlow = (this->baseCols + (groupSize / 16) - 1) / (groupSize / 16);
        errorGlobal = cl::NDRange(static_cast<size_t>(tilesFast) * groupSize, tilesSlow);
        const int corrGroups = tilesFast * tilesSlow;

        // buffers: shift sums, border rows, solver system, u, stats and correlation sums
        shiftSums = OclArray<uint64_t>::zeros(shiftSumsSize, rawQueue);
        if constexpr (copyBorderRows)
            borderRowsCopy = OclArray<float>(borderCopyRows * this->baseCols, rawQueue);
        solverSystem = OclArray<uint64_t>(solverSystemSize, rawQueue);
        u = OclArray<cl_half>(this->baseRows, this->baseCols, rawQueue);
        sumSq = OclArray<uint64_t>::zeros(2, rawQueue);
        errorSeq = ImageBuffer(this->baseRows, this->baseCols, rawQueue);
        dotPartial = ImageBuffer(corrGroups, rawQueue);
        uNormPartial = ImageBuffer(corrGroups, rawQueue);
        zNormPartial = ImageBuffer(corrGroups, rawQueue);
        correlation = ImageBuffer(1, rawQueue);

        // kernels with per-call buffer bindings
        const cl::Buffer borderCopy = copyBorderRows ? borderRowsCopy.clBuffer() : cl::Buffer();
        copyBorderRowsKernel = cl::Kernel(program, "me_copy_border_rows");
        shiftSumsKernel = cl::Kernel(program, "me_shift_sums");
        const size_t shiftScratch = reductionMode(program) == ReductionMode::Subgroup ? reductionScratchBytes(program, "me_shift_sums", cl::NDRange(groupSize), maxGroupShifts)
                                                                                      : static_cast<size_t>(16 * groupSize + groupSize) * sizeof(float);
        setArgs(shiftSumsKernel, cl::Buffer(), borderCopy, shiftSums.clBuffer(), this->baseCols, this->baseRows, shiftInteriorBlocks, borderBlocksPerLine, cl::Local(shiftScratch));
        buildSystemKernel = cl::Kernel(program, "me_build_system");
        setArgs(buildSystemKernel, cl::Buffer(), shiftSums.clBuffer(), solverSystem.clBuffer(), this->baseCols, this->baseRows);
        solveSystemKernel = cl::Kernel(program, "me_solve_system");
        setArgs(solveSystemKernel, solverSystem.clBuffer(), shiftSums.clBuffer(), sumSq.clBuffer(), coefficients.clBuffer(), stopFlag.clBuffer());
        if constexpr (copyBorderRows)
            setArgs(copyBorderRowsKernel, cl::Buffer(), borderCopy, this->baseCols, this->baseRows);

        const cl::NDRange errorLocal(groupSize);
        errorSequenceKernel = cl::Kernel(program, "calculate_error_sequence");
        setArgs(errorSequenceKernel, cl::Buffer(), errorSeq.clBuffer(), coefficients.clBuffer(), this->baseCols, this->baseRows, stopFlag.clBuffer());
        meErrorKernel = cl::Kernel(program, "me_error_sequence_u_sumsq_fused");
        setArgs(meErrorKernel, cl::Buffer(), this->randomMatrix.clBuffer(), u.clBuffer(), coefficients.clBuffer(), sumSq.clBuffer(), this->baseCols, this->baseRows, stopFlag.clBuffer(),
            cl::Local(reductionScratchBytes(program, "me_error_sequence_u_sumsq_fused", errorLocal, 1)));
        corrKernel = cl::Kernel(program, "calculate_error_sequence_and_partial_corr_fused");
        setArgs(corrKernel, errorSeq.clBuffer(), this->randomMatrix.clBuffer(), coefficients.clBuffer(), dotPartial.clBuffer(), uNormPartial.clBuffer(), zNormPartial.clBuffer(), this->baseCols,
            this->baseRows, stopFlag.clBuffer(), cl::Local(3 * groupSize * sizeof(float)));
        corrFinalizeKernel = cl::Kernel(program, "finalize_correlation");
        setArgs(corrFinalizeKernel, dotPartial.clBuffer(), uNormPartial.clBuffer(), zNormPartial.clBuffer(), correlation.clBuffer(), corrGroups, cl::Local(3 * groupSize * sizeof(float)));

        applyGlobal = cl::NDRange(static_cast<size_t>(calculateLocalGroupsNumber((this->totalPixels + 3) / 4, groupSize)) * groupSize);
        applyRgbKernel = cl::Kernel(program, "apply_watermark_rgb");
        applyGrayKernel = cl::Kernel(program, "apply_watermark_gray");
        applyRowMajorKernel = cl::Kernel(program, "apply_watermark_row_major");
    }

    // RGB embedding: computes watermark u from luma mask, adds to all channels
    void makeWatermark(const ImageBuffer& inputGrayImage, const ImageOutputBuffer& inputImage, ImageOutputBuffer& output) override { embed(inputGrayImage, &inputImage, output, Layout::ColMajor); }

    // grayscale embedding: computes watermark u and adds to luma
    void makeWatermark(const ImageBuffer& inputGrayImage, ImageOutputBuffer& output, const Layout outputLayout) override { embed(inputGrayImage, nullptr, output, outputLayout); }

    // detection: correlation between image prediction error and watermark error
    float detectWatermark(const ImageBuffer& inputImage) override {
        cl_utils::executeKernel(
            [&]() {
                const cl::Buffer input = inputImage.clBuffer();
                solvePredictionCoefficients(input);
                // prediction error of image with sign for correlation
                errorSequenceKernel.setArg(0, input);
                enqueue(errorSequenceKernel, errorGlobal, cl::NDRange(groupSize));
                // prediction error of mask * watermark, correlated with image prediction error
                enqueue(corrKernel, errorGlobal, cl::NDRange(groupSize));
                enqueue(corrFinalizeKernel, cl::NDRange(groupSize), cl::NDRange(groupSize));
            },
            "detectWatermark");
        const float result = correlation.scalar();
        return std::isfinite(result) ? result : 0.0f;
    }

  private:
    using WatermarkBase::alignUp;

    static constexpr int localSize = (p * p) - 1;
    // shift sums layout, must match kernel defines in kernels.hpp
    static constexpr int pad = p / 2;
    static constexpr int maxShift = p - 1;
    static constexpr int numShifts = ((((2 * maxShift) + 1) * ((2 * maxShift) + 1)) + 1) / 2;
    static constexpr int borderSize = 4 * pad;
    static constexpr int shiftSumsSize = numShifts * (1 + (2 * borderSize));
    static constexpr int shiftGroups = p >= 9 ? 3 : (p >= 7 ? 2 : 1);
    static constexpr int firstShift(const int dc) { return dc == 0 ? 0 : (maxShift + 1) + ((dc - 1) * (2 * maxShift + 1)); }
    static constexpr int maxGroupShifts = [] {
        int largest = 0;
        for (int group = 0; group < shiftGroups; ++group) {
            const int firstDc = group * (maxShift + 1) / shiftGroups;
            const int lastDc = (group + 1) * (maxShift + 1) / shiftGroups;
            const int end = lastDc > maxShift ? numShifts : firstShift(lastDc);
            largest = std::max(largest, end - firstShift(firstDc));
        }
        return largest;
    }();
    static constexpr int interiorCols = p >= 7 ? 2 : 1;
    static constexpr int interiorRun = 8;
    static constexpr bool copyBorderRows = p >= 7;
    static constexpr int borderCopyRows = 6 * pad;
    // ME solver system (Rx packed lower triangular, then rx), exact fixed point
    static constexpr int solverSystemSize = ((localSize * (localSize + 1)) / 2) + localSize;
    // inner shift sums work-groups per compute unit
    static constexpr int residentGroupsPerUnit = 2;

    cl::CommandQueue queue;
    cl::Program program;
    int groupSize;
    ImageBuffer coefficients;
    FlagBuffer stopFlag;

    int borderBlocksPerLine;
    cl::NDRange shiftSumsGlobal, errorGlobal, applyGlobal;
    OclArray<uint64_t> shiftSums;
    OclArray<float> borderRowsCopy;
    OclArray<uint64_t> solverSystem;
    OclArray<cl_half> u;
    OclArray<uint64_t> sumSq;
    ImageBuffer errorSeq;
    ImageBuffer dotPartial;
    ImageBuffer uNormPartial;
    ImageBuffer zNormPartial;
    ImageBuffer correlation;

    cl::Kernel copyBorderRowsKernel, shiftSumsKernel, buildSystemKernel, solveSystemKernel;
    cl::Kernel errorSequenceKernel, meErrorKernel, corrKernel, corrFinalizeKernel;
    cl::Kernel applyRgbKernel, applyGrayKernel, applyRowMajorKernel;

    // enqueues kernel on in-order queue
    void enqueue(const cl::Kernel& kernel, const cl::NDRange& global, const cl::NDRange& local) { queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local); }

    // watermark added to 8-bit RGB or luma when inputImage is null
    void embed(const ImageBuffer& inputGrayImage, const ImageOutputBuffer* inputImage, ImageOutputBuffer& output, const Layout outputLayout) {
        using namespace cl_utils;
        const int channels = inputImage ? inputImage->getChannels() : 1;
        // output buffer reused, reallocated only when dimensions or channels change
        if (output.empty() || output.getRows() != this->baseRows || output.getCols() != this->baseCols || output.getChannels() != channels)
            output = ImageOutputBuffer(this->baseRows, this->baseCols, channels, queue.get());
        executeKernel(
            [&]() {
                const cl::Buffer gray = inputGrayImage.clBuffer();
                // solve prediction coefficients and compute u, sum(u^2) and max|e|
                solvePredictionCoefficients(gray);
                meErrorKernel.setArg(0, gray);
                enqueue(meErrorKernel, errorGlobal, cl::NDRange(groupSize));
                // scale u by strength and add to channels or luma
                cl::Kernel& apply = inputImage ? applyRgbKernel : (outputLayout == Layout::ColMajor ? applyGrayKernel : applyRowMajorKernel);
                const cl::Buffer source = inputImage ? inputImage->clBuffer() : gray;
                if (&apply == &applyRowMajorKernel) {
                    setArgs(apply, source, u.clBuffer(), sumSq.clBuffer(), output.clBuffer(), this->strengthNumerator, this->baseCols, this->baseRows);
                    const int tileRows = groupSize / 16;
                    enqueue(apply, cl::NDRange(roundUp(this->baseRows, 16), ((this->baseCols + 15) / 16) * tileRows), cl::NDRange(16, tileRows));
                } else {
                    setArgs(apply, source, u.clBuffer(), sumSq.clBuffer(), output.clBuffer(), this->strengthNumerator, this->totalPixels);
                    enqueue(apply, applyGlobal, cl::NDRange(groupSize));
                }
            },
            "makeWatermark");
    }

    // compute shift sums and solve prediction system, p >= 7 copies border rows first
    void solvePredictionCoefficients(const cl::Buffer& image) {
        const cl::NDRange local(groupSize);
        if constexpr (copyBorderRows) {
            copyBorderRowsKernel.setArg(0, image);
            enqueue(copyBorderRowsKernel, cl::NDRange(cl_utils::roundUp(borderCopyRows * this->baseCols, groupSize)), local);
        }
        shiftSumsKernel.setArg(0, image);
        enqueue(shiftSumsKernel, shiftSumsGlobal, local);
        buildSystemKernel.setArg(0, image);
        enqueue(buildSystemKernel, cl::NDRange(cl_utils::roundUp(solverSystemSize, groupSize)), local);
        enqueue(solveSystemKernel, local, local);
    }
};
