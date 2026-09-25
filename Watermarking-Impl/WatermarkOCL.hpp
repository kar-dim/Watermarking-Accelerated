#pragma once
#include "buffer.hpp"
#include "include/WatermarkTypes.hpp"
#include "OclArray.hpp"
#include "OclQueueManager.hpp"
#include "opencl_init.h"
#include "opencl_utils.hpp"
#include "WatermarkBase.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

/*!
 *  \brief  Watermark embedding and detection, OpenCL implementation (the same steps as the CUDA one)
 *  \author Dimitris Karatzas
 */
template <int p>
class WatermarkOCL final : public WatermarkBase {
  public:
    WatermarkOCL<p>(const int rows, const int cols, const std::string& watermarkPassword, const float psnr)
        : WatermarkBase(rows, cols, watermarkPassword, psnr, initializeRandomMatrix), queue(OclQueueManager::getInstance().getQueue()), program(cl_utils::OpenCLKernelCache<p>::getProgram()),
          groupSize(cl_utils::workGroupSize(program)) {
        using namespace cl_utils;
        const cl_command_queue rawQueue = queue.get();
        const cl::Device& device = OclQueueManager::getInstance().getDevice();
        coefficients = ImageBuffer(localSize, rawQueue);
        stopFlag = FlagBuffer::zeros(1, rawQueue);

        // shift sums grid: border lines are summed in runs, one run per workitem (borderBlocksPerLine groups per line and column shift
        // group) and the inner sums use as many workgroups at once (of course NEVER more than tasks), one set per
        // column shift group. When there are few border groups, room is left for them to run together with the inner groups
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

        // buffers: the shift sums (zeroed once), the border rows copy, the solver system, u, [0] = sum(u^2) and [1] = max|e| float bits (ME only),
        // the detection prediction error, the per group correlation sums, the last group ticket counter and the result
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
        corrGroupCounter = OclArray<uint32_t>::zeros(1, rawQueue);
        correlation = ImageBuffer(1, rawQueue);

        // kernels, the input / output buffers are set per call
        const cl::Buffer borderCopy = copyBorderRows ? borderRowsCopy.clBuffer() : cl::Buffer();
        copyBorderRowsKernel = cl::Kernel(program, "me_copy_border_rows");
        shiftSumsKernel = cl::Kernel(program, "me_shift_sums");
        setArgs(shiftSumsKernel, cl::Buffer(), borderCopy, shiftSums.clBuffer(), this->baseCols, this->baseRows, shiftInteriorBlocks, borderBlocksPerLine);
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
        setArgs(corrKernel, cl::Buffer(), this->randomMatrix.clBuffer(), errorSeq.clBuffer(), coefficients.clBuffer(), dotPartial.clBuffer(), uNormPartial.clBuffer(), zNormPartial.clBuffer(),
            corrGroupCounter.clBuffer(), correlation.clBuffer(), this->baseCols, this->baseRows, stopFlag.clBuffer(), 1,
            cl::Local(reductionScratchBytes(program, "calculate_error_sequence_and_partial_corr_fused", errorLocal, 3)));

        nvfLocal = cl::NDRange(32, groupSize / 32);
        nvfGlobal = cl::NDRange(roundUp(this->baseRows, 32), roundUp(this->baseCols, groupSize / 32));
        nvfKernel = cl::Kernel(program, "nvf");
        nvfEmbedKernel = cl::Kernel(program, "nvf_u_and_sumsq_fused");
        setArgs(nvfEmbedKernel, cl::Buffer(), this->randomMatrix.clBuffer(), u.clBuffer(), sumSq.clBuffer(), this->baseCols, this->baseRows,
            cl::Local(reductionScratchBytes(program, "nvf_u_and_sumsq_fused", nvfLocal, 1)));

        applyGlobal = cl::NDRange(static_cast<size_t>(calculateLocalGroupsNumber((this->totalPixels + 3) / 4, groupSize)) * groupSize);
        applyRgbKernel = cl::Kernel(program, "apply_watermark_rgb");
        applyGrayKernel = cl::Kernel(program, "apply_watermark_gray");
        applyRowMajorKernel = cl::Kernel(program, "apply_watermark_row_major");
    }

    // RGB embedding: computes the strengthened watermark u from the luma (NVF or ME mask), then adds it to all channels of the 8-bit image
    void makeWatermark(const ImageBuffer& inputGrayImage, const ImageOutputBuffer& inputImage, ImageOutputBuffer& output, const MaskMethod maskType) override {
        embed(inputGrayImage, &inputImage, output, maskType, Layout::ColMajor);
    }

    // grayscale embedding: computes the strengthened watermark u (NVF or ME mask), then adds it to the luma itself
    void makeWatermark(const ImageBuffer& inputGrayImage, ImageOutputBuffer& output, const MaskMethod maskType, const Layout outputLayout) override {
        embed(inputGrayImage, nullptr, output, maskType, outputLayout);
    }

    // detection: correlation between the prediction error of the image and the prediction error of (mask * watermark)
    float detectWatermark(const ImageBuffer& inputImage, const MaskMethod maskType) override {
        const bool isME = maskType == MaskMethod::ME;
        if (!isME && nvfMask.empty())
            nvfMask = ImageBuffer(this->baseRows, this->baseCols, queue.get());
        cl_utils::executeKernel(
            [&]() {
                const cl::Buffer input = inputImage.clBuffer();
                solvePredictionCoefficients(input);
                // prediction error of the image (with its sign, the correlation needs it)
                errorSequenceKernel.setArg(0, input);
                enqueue(errorSequenceKernel, errorGlobal, cl::NDRange(groupSize));
                // prediction error of (mask x watermark), correlated with the image's prediction error, the last workgroup writes the result
                // ME: the mask |e| is formed while loading, NVF: the mask is computed first by its own kernel
                if (!isME) {
                    cl_utils::setArgs(nvfKernel, input, nvfMask.clBuffer(), this->baseCols, this->baseRows);
                    enqueue(nvfKernel, nvfGlobal, nvfLocal);
                }
                corrKernel.setArg(0, isME ? errorSeq.clBuffer() : nvfMask.clBuffer());
                corrKernel.setArg(12, static_cast<int>(isME));
                enqueue(corrKernel, errorGlobal, cl::NDRange(groupSize));
            },
            "detectWatermark");
        const float result = correlation.scalar();
        return std::isfinite(result) ? result : 0.0f;
    }

  private:
    using WatermarkBase::alignUp;

    static constexpr int localSize = (p * p) - 1;
    // shift sums layout, must match the kernel defines (kernels.hpp)
    static constexpr int pad = p / 2;
    static constexpr int maxShift = p - 1;
    static constexpr int numShifts = ((((2 * maxShift) + 1) * ((2 * maxShift) + 1)) + 1) / 2;
    static constexpr int borderSize = 4 * pad;
    static constexpr int shiftSumsSize = numShifts * (1 + (2 * borderSize));
    static constexpr int shiftGroups = p >= 9 ? 3 : (p >= 7 ? 2 : 1);
    static constexpr int interiorCols = p >= 7 ? 2 : 1;
    static constexpr int interiorRun = 8;
    static constexpr bool copyBorderRows = p >= 7;
    static constexpr int borderCopyRows = 6 * pad;
    // the ME solver system (Rx packed lower triangular, then rx), exact ulong fixed point
    static constexpr int solverSystemSize = ((localSize * (localSize + 1)) / 2) + localSize;
    // inner shift sums work-groups per compute unit
    static constexpr int residentGroupsPerUnit = 2;

    cl::CommandQueue queue;
    cl::Program program;
    int groupSize;
    ImageBuffer coefficients;
    FlagBuffer stopFlag;

    int borderBlocksPerLine;
    cl::NDRange shiftSumsGlobal, errorGlobal, applyGlobal, nvfGlobal, nvfLocal;
    OclArray<uint64_t> shiftSums;
    OclArray<float> borderRowsCopy;
    OclArray<uint64_t> solverSystem;
    OclArray<cl_half> u;
    OclArray<uint64_t> sumSq;
    ImageBuffer errorSeq;
    ImageBuffer nvfMask;
    ImageBuffer dotPartial;
    ImageBuffer uNormPartial;
    ImageBuffer zNormPartial;
    OclArray<uint32_t> corrGroupCounter;
    ImageBuffer correlation;

    cl::Kernel copyBorderRowsKernel, shiftSumsKernel, buildSystemKernel, solveSystemKernel;
    cl::Kernel errorSequenceKernel, meErrorKernel, corrKernel, nvfKernel, nvfEmbedKernel;
    cl::Kernel applyRgbKernel, applyGrayKernel, applyRowMajorKernel;

    // enqueues a kernel on the in-order queue
    void enqueue(const cl::Kernel& kernel, const cl::NDRange& global, const cl::NDRange& local) { queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local); }

    static WatermarkBuffer initializeRandomMatrix(const std::span<const uint16_t> watermarkHalfBits, const int rows, const int cols) {
        return WatermarkBuffer(rows, cols, watermarkHalfBits.data(), OclQueueManager::getInstance().getQueueRaw());
    }

    // embedding: the watermark is added to the 8-bit "inputImage" (RGB), or to the luma when inputImage is null
    void embed(const ImageBuffer& inputGrayImage, const ImageOutputBuffer* inputImage, ImageOutputBuffer& output, const MaskMethod maskType, const Layout outputLayout) {
        using namespace cl_utils;
        const bool isME = maskType == MaskMethod::ME;
        const int channels = inputImage ? inputImage->getChannels() : 1;
        // the output buffer is reused, allocated again only when its size or channel count changes
        if (output.empty() || output.getRows() != this->baseRows || output.getCols() != this->baseCols || output.getChannels() != channels)
            output = ImageOutputBuffer(this->baseRows, this->baseCols, channels, queue.get());
        executeKernel(
            [&]() {
                const cl::Buffer gray = inputGrayImage.clBuffer();
                if (!isME) {
                    // NVF: mask (local variance) x watermark -> u and sum(u^2)
                    sumSq.fillZero();
                    nvfEmbedKernel.setArg(0, gray);
                    enqueue(nvfEmbedKernel, nvfGlobal, nvfLocal);
                } else {
                    // ME: solve the prediction coefficients (this also zeroes the sums), then prediction error x watermark -> u, sum(u^2)
                    // and max|e|
                    solvePredictionCoefficients(gray);
                    meErrorKernel.setArg(0, gray);
                    enqueue(meErrorKernel, errorGlobal, cl::NDRange(groupSize));
                }
                // scale u by the strength and add it to each channel of the 8-bit image, or to the luma
                cl::Kernel& apply = inputImage ? applyRgbKernel : (outputLayout == Layout::ColMajor ? applyGrayKernel : applyRowMajorKernel);
                const cl::Buffer source = inputImage ? inputImage->clBuffer() : gray;
                if (&apply == &applyRowMajorKernel) {
                    setArgs(apply, source, u.clBuffer(), sumSq.clBuffer(), output.clBuffer(), this->strengthNumerator, this->baseCols, this->baseRows, static_cast<int>(isME));
                    const int tileRows = groupSize / 16;
                    enqueue(apply, cl::NDRange(roundUp(this->baseRows, 16), ((this->baseCols + 15) / 16) * tileRows), cl::NDRange(16, tileRows));
                } else {
                    setArgs(apply, source, u.clBuffer(), sumSq.clBuffer(), output.clBuffer(), this->strengthNumerator, this->totalPixels, static_cast<int>(isME));
                    enqueue(apply, applyGlobal, cl::NDRange(groupSize));
                }
            },
            "makeWatermark");
    }

    // prediction coefficients (+ stopFlag): all shift sums, build the system, solve it + zero the shift sums and the embedding sums
    // For p >= 7 the border rows are copied first
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
