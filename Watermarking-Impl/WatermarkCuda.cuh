#pragma once
#include "cuda_utils.hpp"
#include "CudaStreamManager.hpp"
#include "GpuArray.hpp"
#include "include/WatermarkTypes.hpp"
#include "kernels/kernels.cuh"
#include "WatermarkBase.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <thrust/iterator/transform_iterator.h>

/*!
 *  \brief  Functions for watermark computation and detection, CUDA implementation.
 *          All scratch buffers are preallocated in the constructor with zero cudaMallocAsync/cudaFreeAsync calls per frame
 *  \author Dimitris Karatzas
 */
template <int p>
class WatermarkCuda final : public WatermarkBase {
  public:
    WatermarkCuda<p>(const unsigned int rows, const unsigned int cols, const std::string& watermarkPassword, const float psnr)
        : WatermarkBase(rows, cols, watermarkPassword, psnr, initializeRandomMatrix), stream(CudaStreamManager::getInstance().getComputeStream()),
          strengthNumerator(strengthFactor * std::sqrt(static_cast<float>(this->totalPixels))), coefficients(localSize, stream), stopFlag(FlagBuffer::zeros(1, stream)) {
        if constexpr (p == 3)
            gridOptimalMe = cuda_utils::gridSizeMeCalculate(me_p3, meBlockSize.x);
        else if constexpr (p == 5)
            gridOptimalMe = cuda_utils::gridSizeMeCalculate(me_p5, meBlockSize.x);
        else if constexpr (p == 7)
            gridOptimalMe = cuda_utils::gridSizeMeCalculate(me_p7, meBlockSize.x);
        else
            gridOptimalMe = cuda_utils::gridSizeMeCalculate(me_p9, meBlockSize.x);
        constexpr unsigned int pixelsPerBlockY = (p == 3) ? (meBlockSize.x * 2) : meBlockSize.x;
        const unsigned int meTotalBlocksY = WatermarkBase::alignUp<pixelsPerBlockY>(this->baseRows) / pixelsPerBlockY;
        meParams = {meTotalBlocksY, meTotalBlocksY * this->baseCols};
        initializeCubStorage();
        initializePreallocatedBuffers();
    }

    // Embed: compute strengthened watermark u (via NVF or ME mask), then apply to all channels of inputImage
    void makeWatermark(const ImageBuffer& inputGrayImage, const ImageBuffer& inputImage, ImageOutputBuffer& output, const MaskMethod maskType) override {
        // reuse output buffer, only (re)allocate when dimensions or channel count change
        if (output.empty() || output.getRows() != this->baseRows || output.getCols() != this->baseCols || output.getChannels() != inputImage.getChannels())
            output = GpuArray<uint8_t>(this->baseRows, this->baseCols, inputImage.getChannels(), stream);

        sumSq.fillZero();

        if (maskType == MaskMethod::NVF) {
            // fused NVF: local variance mask x watermark -> strengthened watermark u + sum(u^2)
            const dim3 gridSize = cuda_utils::gridSizeCalculate(windowBlockSize, this->baseCols, this->baseRows);
            nvf_u_and_sumsq_fused<p><<<gridSize, windowBlockSize, 0, stream>>>(inputGrayImage.data(), this->randomMatrix.data(), u.data(), sumSq.data(), this->baseCols, this->baseRows);
        } else {
            // ME path: solve prediction error model, compute error sequence, normalize, fuse with watermark
            Rx.fillZero();
            rx.fillZero();
            launchMeKernel(inputGrayImage.data());
            launchCholeskySolver();
            const dim3 errorGridSize = cuda_utils::gridSizeCalculate(windowBlockSize, this->baseCols, this->baseRows);
            calculate_error_sequence<p><<<errorGridSize, windowBlockSize, 0, stream>>>(inputGrayImage.data(), nullptr, errorSeq.data(), this->coefficients.data(), this->baseCols, this->baseRows, true,
                                                                                       this->stopFlag.data());
            // max-reduce for normalization
            reduceMaxCub(errorSeq.data(), errorSeqMax.data());
            // fused ME: normalized error x watermark -> strengthened watermark u + sum(u^2)
            const int blocksComputeU = cuda_utils::gridSize1DStridedCalculate(this->totalPixels, strWatermarkBlockSize);
            me_u_and_sumsq_fused<<<blocksComputeU, strWatermarkBlockSize, 0, stream>>>(errorSeq.data(), this->randomMatrix.data(), u.data(), sumSq.data(), errorSeqMax.data(), this->totalPixels);
        }
        // scale u by strength factor and add to each channel of the input image
        const int blocksApply = cuda_utils::gridSize1DStridedCalculate(this->totalPixels, applyWatermarkBlockSize);
        apply_watermark_fused<<<blocksApply, applyWatermarkBlockSize, 0, stream>>>(inputImage.data(), u.data(), sumSq.data(), output.data(), this->strengthNumerator, this->totalPixels,
                                                                                   static_cast<int>(inputImage.getChannels()));
    }

    // Detect: compute prediction error, detection mask, then correlate with watermark
    float detectWatermark(const ImageBuffer& inputImage, const MaskMethod maskType) override {
        // solve prediction error model (Rx, rx -> coefficients via Cholesky)
        Rx.fillZero();
        rx.fillZero();
        launchMeKernel(inputImage.data());
        launchCholeskySolver();

        const dim3 windowGrid = cuda_utils::gridSizeCalculate(windowBlockSize, this->baseCols, this->baseRows);

        // compute prediction error sequence (non-abs, needed for correlation sign)
        calculate_error_sequence<p>
            <<<windowGrid, windowBlockSize, 0, stream>>>(inputImage.data(), nullptr, errorSeq.data(), this->coefficients.data(), this->baseCols, this->baseRows, false, this->stopFlag.data());

        // compute detection mask (ME: abs-normalized error, NVF: local variance)
        if (maskType == MaskMethod::ME) {
            reduceMaxCub(thrust::make_transform_iterator(errorSeq.data(), AbsTransformOp{}), errorSeqMax.data());
            const int gridSize = cuda_utils::gridSize1DStridedCalculate(this->totalPixels, maskNormalizationBlockSize);
            compute_abs_normalized_mask<<<gridSize, maskNormalizationBlockSize, 0, stream>>>(errorSeq.data(), mask.data(), errorSeqMax.data(), this->totalPixels);
        } else {
            nvf<p><<<windowGrid, windowBlockSize, 0, stream>>>(inputImage.data(), mask.data(), this->baseCols, this->baseRows);
        }

        // fused: recompute error sequence from (mask x watermark), accumulate partial dot / normU / normZ
        calculate_error_sequence_and_partial_corr_fused<p><<<windowGrid, windowBlockSize, 0, stream>>>(mask.data(), this->randomMatrix.data(), errorSeq.data(), this->coefficients.data(),
                                                                                                       dotPartial.data(), uNormPartial.data(), zNormPartial.data(), this->baseCols, this->baseRows,
                                                                                                       this->stopFlag.data());
        // reduce partials -> final normalized correlation
        calculate_final_correlation<<<1, corrFinalBlockSize, 0, stream>>>(dotPartial.data(), uNormPartial.data(), zNormPartial.data(), corrResult.data(), corrNumBlocks);

        const float correlation = corrResult.scalar();
        return std::isfinite(correlation) ? correlation : 0.0f;
    }

  private:
    static constexpr int localSize = (p * p) - 1;
    static constexpr dim3 windowBlockSize{32, 8};
    static constexpr dim3 meBlockSize{p >= 7 ? 128 : 256, 1};
    static constexpr unsigned int corrFinalBlockSize = 1024;
    static constexpr unsigned int strWatermarkBlockSize = 768;
    static constexpr unsigned int applyWatermarkBlockSize = 768;
    static constexpr unsigned int maskNormalizationBlockSize = 768;

    cudaStream_t stream;
    float strengthNumerator;
    ImageBuffer coefficients;
    FlagBuffer stopFlag;

    dim3 meParams;
    unsigned int gridOptimalMe;
    unsigned int corrNumBlocks;
    GpuArray<uint8_t> cubTempStorage;

    // preallocated scratch buffers, shared across embed and detect paths
    GpuArray<float> u;
    GpuArray<uint64_t> sumSq;
    GpuArray<uint64_t> Rx;
    GpuArray<uint64_t> rx;
    GpuArray<float> errorSeq;
    GpuArray<float> errorSeqMax;
    GpuArray<float> mask;
    GpuArray<float> dotPartial;
    GpuArray<float> uNormPartial;
    GpuArray<float> zNormPartial;
    GpuArray<float> corrResult;

    static ImageBuffer initializeRandomMatrix(const std::vector<float>& watermarkVec, const unsigned int rows, const unsigned int cols) {
        return ImageBuffer(rows, cols, watermarkVec.data(), CudaStreamManager::getInstance().getComputeStream());
    }

    void initializeCubStorage() {
        AbsTransformOp op;
        auto iter = thrust::make_transform_iterator((const float*)nullptr, op);
        size_t tmpBytesTransform = 0;
        cub::DeviceReduce::Max(nullptr, tmpBytesTransform, iter, (float*)nullptr, this->totalPixels, 0);
        size_t tmpBytesRaw = 0;
        cub::DeviceReduce::Max(nullptr, tmpBytesRaw, (const float*)nullptr, (float*)nullptr, this->totalPixels, 0);
        cubTempStorage = GpuArray<uint8_t>(static_cast<unsigned int>(std::max(tmpBytesTransform, tmpBytesRaw)), stream);
    }

    void initializePreallocatedBuffers() {
        constexpr unsigned int RxSize = static_cast<unsigned int>((localSize * (localSize + 1)) / 2);
        constexpr unsigned int rxSize = static_cast<unsigned int>(localSize);
        const dim3 corrGrid = cuda_utils::gridSizeCalculate(windowBlockSize, this->baseCols, this->baseRows);
        corrNumBlocks = corrGrid.x * corrGrid.y;

        u = GpuArray<float>(this->baseRows, this->baseCols, stream);
        sumSq = GpuArray<uint64_t>(1, stream);
        Rx = GpuArray<uint64_t>(RxSize, stream);
        rx = GpuArray<uint64_t>(rxSize, stream);
        errorSeq = GpuArray<float>(this->baseRows, this->baseCols, stream);
        errorSeqMax = GpuArray<float>(1, stream);
        mask = GpuArray<float>(this->baseRows, this->baseCols, stream);
        dotPartial = GpuArray<float>(corrNumBlocks, stream);
        uNormPartial = GpuArray<float>(corrNumBlocks, stream);
        zNormPartial = GpuArray<float>(corrNumBlocks, stream);
        corrResult = GpuArray<float>(1, stream);
    }

    template <typename InputIteratorT>
    void reduceMaxCub(InputIteratorT in, float* out) const {
        size_t tmpStorageBytes = cubTempStorage.bytes();
        cub::DeviceReduce::Max(const_cast<uint8_t*>(cubTempStorage.data()), tmpStorageBytes, in, out, this->totalPixels, stream);
    }

    // dispatch the correct ME kernel variant based on prediction order p
    void launchMeKernel(const float* imageData) {
        if constexpr (p == 3)
            me_p3<<<gridOptimalMe, meBlockSize.x, 0, stream>>>(imageData, Rx.data(), rx.data(), this->baseCols, this->baseRows, meParams.x, meParams.y);
        else if constexpr (p == 5)
            me_p5<<<gridOptimalMe, meBlockSize.x, 0, stream>>>(imageData, Rx.data(), rx.data(), this->baseCols, this->baseRows, meParams.x, meParams.y);
        else if constexpr (p == 7)
            me_p7<<<gridOptimalMe, meBlockSize.x, 0, stream>>>(imageData, Rx.data(), rx.data(), this->baseCols, this->baseRows, meParams.x, meParams.y);
        else
            me_p9<<<gridOptimalMe, meBlockSize.x, 0, stream>>>(imageData, Rx.data(), rx.data(), this->baseCols, this->baseRows, meParams.x, meParams.y);
    }

    // solve Rx*a = rx via Cholesky decomposition to get prediction coefficients
    void launchCholeskySolver() {
        if constexpr (p <= 5)
            cholesky_solver<p><<<1, 1, 0, stream>>>(Rx.data(), rx.data(), const_cast<float*>(this->coefficients.data()), const_cast<int32_t*>(this->stopFlag.data()));
        else
            cholesky_solver_parallel<p><<<1, 32, 0, stream>>>(Rx.data(), rx.data(), const_cast<float*>(this->coefficients.data()), const_cast<int32_t*>(this->stopFlag.data()));
    }
};
