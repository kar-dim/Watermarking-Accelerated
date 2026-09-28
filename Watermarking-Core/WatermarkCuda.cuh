#pragma once
#include "buffer.hpp"
#include "cuda_utils.hpp"
#include "CudaArray.hpp"
#include "CudaStreamManager.hpp"
#include "CudaCheck.hpp"
#include "kernels/kernels.cuh"
#include "WatermarkBase.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <memory>
#include <utility>
#include <vector>

// CUDA graph of one pipeline (embed or detect): recorded once and replayed with ONE launch while its buffers and kernel arguments (the
// key) stay the same, recorded again when they change (the existing graph is updated in place when possible). It is recorded on its own
// stream: capture is per stream, work that another thread puts on the compute stream (the image prefetch uploads) would end up in the graph or break the capture
class PipelineGraph {
  public:
    struct Key {
        std::array<const void*, 3> buffers;
        int channels;
        int layout;
        float strength;
        bool operator==(const Key&) const = default;
    };

    PipelineGraph() = default;
    PipelineGraph(const PipelineGraph&) = delete;
    PipelineGraph& operator=(const PipelineGraph&) = delete;
    ~PipelineGraph() {
        if (exec)
            cudaGraphExecDestroy(exec);
        if (captureStream)
            cudaStreamDestroy(captureStream);
    }

    // records the kernels that "enqueue" puts on the stream it receives (when the key changed), then launches the graph on "stream"
    template <typename Enqueue>
    void launch(cudaStream_t stream, const Key& newKey, Enqueue&& enqueue) {
        if (!exec || !(newKey == key)) {
            if (!captureStream)
                CUDA_CHECK(cudaStreamCreateWithFlags(&captureStream, cudaStreamNonBlocking));
            cudaGraph_t graph = nullptr;
            CUDA_CHECK(cudaStreamBeginCapture(captureStream, cudaStreamCaptureModeThreadLocal));
            try {
                enqueue(captureStream);
            } catch (...) {
                if (cudaStreamEndCapture(captureStream, &graph) == cudaSuccess && graph)
                    cudaGraphDestroy(graph);
                // failed captures must not stay on the stream and the next capture starts on a new one
                cudaStreamDestroy(captureStream);
                captureStream = nullptr;
                (void)cudaGetLastError();
                throw;
            }
            CUDA_CHECK(cudaStreamEndCapture(captureStream, &graph));
            cudaGraphExecUpdateResultInfo updateInfo;
            if (exec && cudaGraphExecUpdate(exec, graph, &updateInfo) != cudaSuccess) {
                // different kernels (another output layout or channel count): create a new graph
                (void)cudaGetLastError();
                cudaGraphExecDestroy(exec);
                exec = nullptr;
            }
            const cudaError_t instantiated = exec ? cudaSuccess : cudaGraphInstantiate(&exec, graph, 0);
            cudaGraphDestroy(graph);
            CUDA_CHECK(instantiated);
            key = newKey;
        }
        CUDA_CHECK(cudaGraphLaunch(exec, stream));
    }

  private:
    cudaStream_t captureStream = nullptr;
    cudaGraphExec_t exec = nullptr;
    Key key{};
};

/*!
 *  \brief  Watermark embedding and detection, CUDA implementation
 *  \author Dimitris Karatzas
 */
template <int p>
class WatermarkCuda final : public WatermarkBase {
  public:
    WatermarkCuda<p>(const int rows, const int cols, WatermarkBuffer watermark, const float psnr)
        : WatermarkBase(rows, cols, std::move(watermark), psnr), stream(CudaStreamManager::getInstance().getComputeStream()), coefficients(localSize, stream), stopFlag(FlagBuffer::zeros(1, stream)) {
        using L = MeShiftLayout<p>;
        // border lines are summed in runs, one run per thread (borderBlocksPerLine blocks per line and column shift group)
        borderBlocksPerLine = (((std::max(this->baseRows, this->baseCols) + L::interiorRun - 1) / L::interiorRun) + shiftBlockSize - 1) / shiftBlockSize;
        // the inner sums use as many blocks as fit on the GPU at once, NEVER more than tasks, one set of blocks per column shift
        // group. When there are few border blocks, room is left for them so they run together with the inner blocks, not after them
        const int residentBlocks = static_cast<int>(cuda_utils::gridSizeMeCalculate(me_shift_sums<p>, shiftBlockSize));
        const int borderBlocks = borderBlocksPerLine * 2 * L::borderSize * L::shiftGroups;
        const int interiorSlots = 4 * borderBlocks <= residentBlocks ? residentBlocks - borderBlocks : residentBlocks;
        const int interiorTasks = ((this->baseRows + L::interiorRun - 1) / L::interiorRun) * ((this->baseCols - 2 * L::pad + L::interiorCols - 1) / L::interiorCols);
        const int blocksPerGroup = std::min<int>(interiorSlots / L::shiftGroups, (interiorTasks + shiftBlockSize - 1) / shiftBlockSize);
        shiftInteriorBlocks = std::max(blocksPerGroup, 1) * L::shiftGroups;
        shiftSumsBlocks = shiftInteriorBlocks + borderBlocks;
        // zeroed once, after that me_solve_system zeroes them after they are read
        shiftSums = CudaArray<uint64_t>::zeros(L::shiftSumsSize, stream);
        if constexpr (L::copyBorderRows)
            borderRowsCopy = CudaArray<float>(L::borderCopyRows * this->baseCols, stream);
        solverSystem = CudaArray<uint64_t>(solverSystemSize, stream);
        const dim3 errorGrid = ErrorTile<p>::grid(this->baseCols, this->baseRows);
        const int corrNumBlocks = errorGrid.x * errorGrid.y;
        // per frame buffers live as long as the object (CUDA graphs need fixed pointers!)
        u = CudaArray<__half>(this->baseRows, this->baseCols, stream);
        sumSq = CudaArray<uint64_t>::zeros(2, stream);
        errorSeq = ImageBuffer(this->baseRows, this->baseCols, stream);
        dotPartial = ImageBuffer(corrNumBlocks, stream);
        uNormPartial = ImageBuffer(corrNumBlocks, stream);
        zNormPartial = ImageBuffer(corrNumBlocks, stream);
        corrBlockCounter = CudaArray<unsigned int>::zeros(1, stream);
        float* pinned = nullptr;
        CUDA_CHECK(cudaHostAlloc(&pinned, sizeof(float), cudaHostAllocMapped));
        correlationHost.reset(pinned);
    }

    // RGB embedding: computes the strengthened watermark u from the luma (ME mask), then adds it to all channels of the 8-bit image
    void makeWatermark(const ImageBuffer& inputGrayImage, const ImageOutputBuffer& inputImage, ImageOutputBuffer& output) override {
        embed(inputGrayImage, inputImage.data(), inputImage.getChannels(), output, Layout::ColMajor);
    }

    // grayscale embedding: computes the strengthened watermark u (ME mask), then adds it to the luma itself
    void makeWatermark(const ImageBuffer& inputGrayImage, ImageOutputBuffer& output, const Layout outputLayout) override { embed(inputGrayImage, nullptr, 1, output, outputLayout); }

    // detection: correlation between the prediction error of the image and the prediction error of (mask * watermark)
    float detectWatermark(const ImageBuffer& inputImage) override {
        const PipelineGraph::Key key{
            {inputImage.data(), nullptr, nullptr},
            1, 0, 0.0f
        };
        detectGraph.launch(stream, key, [&](const cudaStream_t graphStream) { enqueueDetect(inputImage, graphStream); });
        CUDA_CHECK(cudaStreamSynchronize(stream));
        const float correlation = *correlationHost;
        return std::isfinite(correlation) ? correlation : 0.0f;
    }

  private:
    static constexpr int localSize = (p * p) - 1;
    static constexpr int choleskyBlockSize = 256;
    static constexpr int shiftBlockSize = 256;
    static constexpr unsigned int applyWatermarkBlockSize = 768;

    cudaStream_t stream;
    ImageBuffer coefficients;
    FlagBuffer stopFlag;

    // shift sums grid: the inner blocks first, then the border blocks
    int shiftInteriorBlocks;
    int borderBlocksPerLine;
    int shiftSumsBlocks;
    CudaArray<uint64_t> shiftSums;
    // row-major copy of the image rows used by the border row sums
    CudaArray<float> borderRowsCopy;
    // the ME solver system (Rx packed lower triangular, then rx), exact uint64 fixed point
    static constexpr int solverSystemSize = ((localSize * (localSize + 1)) / 2) + localSize;
    CudaArray<uint64_t> solverSystem;

    // per frame buffers: u, [0] = sum(u^2) (fixed point) and [1] = max|e| float bits, the detection prediction error, the per block
    // correlation sums, the last block ticket counter and the result (mapped pinned host memory, written by the correlation kernel)
    CudaArray<__half> u;
    CudaArray<uint64_t> sumSq;
    ImageBuffer errorSeq;
    ImageBuffer dotPartial;
    ImageBuffer uNormPartial;
    ImageBuffer zNormPartial;
    CudaArray<unsigned int> corrBlockCounter;
    struct PinnedDeleter {
        void operator()(float* ptr) const { cudaFreeHost(ptr); }
    };
    std::unique_ptr<float, PinnedDeleter> correlationHost;

    PipelineGraph embedGraph;
    PipelineGraph detectGraph;

    // embedding: the watermark is added to the 8-bit "inputImage" (RGB), or to the luma when inputImage is null
    void embed(const ImageBuffer& inputGrayImage, const uint8_t* inputImage, const int channels, ImageOutputBuffer& output, const Layout outputLayout) {
        // the output buffer is reused, allocated again only when its size or channel count changes
        if (output.empty() || output.getRows() != this->baseRows || output.getCols() != this->baseCols || output.getChannels() != channels)
            output = CudaArray<uint8_t>(this->baseRows, this->baseCols, channels, stream);
        const PipelineGraph::Key key{
            {inputGrayImage.data(), inputImage, output.data()},
            channels, static_cast<int>(outputLayout), this->strengthNumerator
        };
        embedGraph.launch(stream, key, [&](const cudaStream_t graphStream) { enqueueEmbed(inputGrayImage, inputImage, channels, output, outputLayout, graphStream); });
    }

    // embedding kernels (recorded into embedGraph on graphStream)
    void enqueueEmbed(const ImageBuffer& inputGrayImage, const uint8_t* inputImage, const int channels, ImageOutputBuffer& output, const Layout outputLayout, const cudaStream_t graphStream) {
        // solve the prediction coefficients (this also zeroes the sums), then prediction error x watermark -> u, sum(u^2) and max|e|
        solvePredictionCoefficients(inputGrayImage.data(), graphStream);
        me_error_sequence_u_sumsq_fused<p><<<ErrorTile<p>::grid(this->baseCols, this->baseRows), ErrorTile<p>::threads, 0, graphStream>>>(
            inputGrayImage.data(), this->randomMatrix.data(), u.data(), this->coefficients.data(), sumSq.data(), this->baseCols, this->baseRows, this->stopFlag.data());
        CUDA_CHECK(cudaGetLastError());
        // scale u by the strength and add it to each channel of the 8-bit image, or to the luma
        const int blocksApply = cuda_utils::gridSize1DStridedCalculate((this->totalPixels + 3) / 4, applyWatermarkBlockSize);
        if (inputImage) {
            apply_watermark_fused<<<blocksApply, applyWatermarkBlockSize, 0, graphStream>>>(inputImage, u.data(), sumSq.data(), output.data(), this->strengthNumerator, this->totalPixels, channels);
        } else if (outputLayout == Layout::ColMajor) {
            apply_watermark_fused<<<blocksApply, applyWatermarkBlockSize, 0, graphStream>>>(
                inputGrayImage.data(), u.data(), sumSq.data(), output.data(), this->strengthNumerator, this->totalPixels, 1);
        } else {
            const dim3 tileGrid((this->baseRows + 31) / 32, (this->baseCols + 31) / 32);
            apply_watermark_row_major<<<tileGrid, dim3(32, 8), 0, graphStream>>>(inputGrayImage.data(), u.data(), sumSq.data(), output.data(), this->strengthNumerator, this->baseCols, this->baseRows);
        }
        CUDA_CHECK(cudaGetLastError());
    }

    // detection kernels (recorded into detectGraph on graphStream), the correlation kernel writes the result to mapped pinned host memory
    void enqueueDetect(const ImageBuffer& inputImage, const cudaStream_t graphStream) {
        // prediction coefficients (Rx, rx, Cholesky)
        solvePredictionCoefficients(inputImage.data(), graphStream);

        // prediction error of the image (with its sign, the correlation needs it)
        const dim3 errorGrid = ErrorTile<p>::grid(this->baseCols, this->baseRows);
        calculate_error_sequence<p>
            <<<errorGrid, ErrorTile<p>::threads, 0, graphStream>>>(inputImage.data(), errorSeq.data(), this->coefficients.data(), this->baseCols, this->baseRows, this->stopFlag.data());
        CUDA_CHECK(cudaGetLastError());

        // prediction error of (mask x watermark), correlated with the image's prediction error. The mask |e| is formed while loading, not
        // divided by max|e| (the correlation does not depend on the scale)
        calculate_error_sequence_and_partial_corr_fused<p><<<errorGrid, ErrorTile<p>::threads, 0, graphStream>>>(errorSeq.data(), this->randomMatrix.data(), this->coefficients.data(),
            dotPartial.data(), uNormPartial.data(), zNormPartial.data(), corrBlockCounter.data(), correlationHost.get(), this->baseCols, this->baseRows, this->stopFlag.data());
        CUDA_CHECK(cudaGetLastError());
    }

    // prediction coefficients (+ stopFlag): all shift sums, build the system, solve it (+ zero the sums). For p >= 7 the border rows are copied first
    void solvePredictionCoefficients(const float* imageData, const cudaStream_t graphStream) {
        constexpr int buildBlockSize = 128;
        using L = MeShiftLayout<p>;
        if constexpr (L::copyBorderRows) {
            const int borderCopySize = L::borderCopyRows * this->baseCols;
            me_copy_border_rows<p><<<(borderCopySize + shiftBlockSize - 1) / shiftBlockSize, shiftBlockSize, 0, graphStream>>>(imageData, borderRowsCopy.data(), this->baseCols, this->baseRows);
            CUDA_CHECK(cudaGetLastError());
        }
        me_shift_sums<p>
            <<<shiftSumsBlocks, shiftBlockSize, 0, graphStream>>>(imageData, borderRowsCopy.data(), shiftSums.data(), this->baseCols, this->baseRows, shiftInteriorBlocks, borderBlocksPerLine);
        CUDA_CHECK(cudaGetLastError());
        me_build_system<p>
            <<<(solverSystemSize + buildBlockSize - 1) / buildBlockSize, buildBlockSize, 0, graphStream>>>(imageData, shiftSums.data(), solverSystem.data(), this->baseCols, this->baseRows);
        CUDA_CHECK(cudaGetLastError());
        me_solve_system<p, choleskyBlockSize><<<1, choleskyBlockSize, 0, graphStream>>>(solverSystem.data(), shiftSums.data(), sumSq.data(), this->coefficients.data(), this->stopFlag.data());
        CUDA_CHECK(cudaGetLastError());
    }
};
