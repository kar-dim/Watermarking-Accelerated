#pragma once
#include "cuda_utils.hpp"
#include "CudaStreamManager.hpp"
#include "GpuArray.hpp"
#include "include/WatermarkTypes.hpp"
#include "kernels/kernels.cuh"
#include "WatermarkBase.hpp"
#include "WatermarkGpu.hpp"
#include <algorithm>
#include <cstdint>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <string>
#include <thrust/iterator/transform_iterator.h>

/*!
 *  \brief  Functions for watermark computation and detection, CUDA implementation.
 *  \author Dimitris Karatzas
 */
template <int p>
class WatermarkCuda final : public WatermarkGPU<p> {
  public:
    WatermarkCuda<p>(const unsigned int rows, const unsigned int cols, const std::string& watermarkPassword, const float psnr)
        : WatermarkGPU<p>(rows, cols, watermarkPassword, psnr), stream(this->stream_) {
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
    }

  private:
    static constexpr dim3 windowBlockSize{32, 8};
    static constexpr dim3 meBlockSize{p >= 7 ? 128 : 256, 1};
    static constexpr unsigned int corrPartialBlockSize = 768;
    static constexpr unsigned int corrFinalBlockSize = 1024;
    static constexpr unsigned int strWatermarkBlockSize = 768;
    static constexpr unsigned int applyWatermarkBlockSize = 768;
    static constexpr unsigned int maskNormalizationBlockSize = 768;

    dim3 meParams;
    cudaStream_t stream;
    unsigned int gridOptimalMe;
    GpuArray<uint8_t> cubTempStorage;

    void initializeCubStorage() {
        AbsTransformOp op;
        auto iter = thrust::make_transform_iterator((const float*)nullptr, op);
        size_t tmpBytesTransform = 0;
        cub::DeviceReduce::Max(nullptr, tmpBytesTransform, iter, (float*)nullptr, this->totalPixels, 0);
        size_t tmpBytesRaw = 0;
        cub::DeviceReduce::Max(nullptr, tmpBytesRaw, (const float*)nullptr, (float*)nullptr, this->totalPixels, 0);
        size_t finalTmpBytes = std::max(tmpBytesTransform, tmpBytesRaw);
        cubTempStorage = GpuArray<uint8_t>(static_cast<unsigned int>(finalTmpBytes), stream);
    }

    template <typename InputIteratorT>
    void reduceMaxCub(InputIteratorT in, float* out) const {
        size_t tmpStorageBytes = cubTempStorage.bytes();
        cub::DeviceReduce::Max(const_cast<uint8_t*>(cubTempStorage.data()), tmpStorageBytes, in, out, this->totalPixels, stream);
    }

    ImageOutputBuffer computeStrengthenedWatermark(const ImageBuffer& inputGrayImage, const ImageBuffer& inputImage, const MaskMethod maskType) const override {
        GpuArray<float> u(this->baseRows, this->baseCols, stream);
        GpuArray<uint8_t> output(inputImage.rows(), inputImage.cols(), inputImage.channels(), stream);
        auto sumSq = GpuArray<uint64_t>::zeros(1, stream);

        float* uPtr = u.data();
        uint64_t* sumSqPtr = sumSq.data();
        if (maskType == MaskMethod::NVF) {
            const dim3 gridSize = cuda_utils::gridSizeCalculate(windowBlockSize, this->baseCols, this->baseRows);
            nvf_u_and_sumsq_fused<p><<<gridSize, windowBlockSize, 0, stream>>>(inputGrayImage.data(), this->randomMatrix.data(), uPtr, sumSqPtr, this->baseCols, this->baseRows);
        } else {
            auto errorSeq = computePredictionErrorData(inputGrayImage, true);
            GpuArray<float> errorSeqMax(1, stream);
            reduceMaxCub(errorSeq.data(), errorSeqMax.data());
            const int blocksComputeU = cuda_utils::gridSize1DStridedCalculate(this->totalPixels, strWatermarkBlockSize);
            me_u_and_sumsq_fused<<<blocksComputeU, strWatermarkBlockSize, 0, stream>>>(errorSeq.data(), this->randomMatrix.data(), uPtr, sumSqPtr, errorSeqMax.data(), this->totalPixels);
        }
        const int blocksApply = cuda_utils::gridSize1DStridedCalculate(this->totalPixels, applyWatermarkBlockSize);
        apply_watermark_fused<<<blocksApply, applyWatermarkBlockSize, 0, stream>>>(inputImage.data(), uPtr, sumSqPtr, output.data(), this->strengthNumerator, this->totalPixels,
                                                                                   static_cast<int>(inputImage.channels()));
        return output;
    }

    ImageBuffer computeCustomMask(const ImageBuffer& inputImage) const override {
        const dim3 gridSize = cuda_utils::gridSizeCalculate(windowBlockSize, this->baseCols, this->baseRows);
        GpuArray<float> customMask(this->baseRows, this->baseCols, stream);
        nvf<p><<<gridSize, windowBlockSize, 0, stream>>>(inputImage.data(), customMask.data(), this->baseCols, this->baseRows);
        return customMask;
    }

    ImageBuffer computeErrorSequence(const ImageBuffer& image, const bool calculateAbs) const override {
        const dim3 gridSize = cuda_utils::gridSizeCalculate(windowBlockSize, this->baseCols, this->baseRows);
        GpuArray<float> errorSequence(this->baseRows, this->baseCols, stream);
        calculate_error_sequence<p><<<gridSize, windowBlockSize, 0, stream>>>(image.data(), nullptr, errorSequence.data(), this->coefficients.data(), this->baseCols, this->baseRows, calculateAbs,
                                                                              this->stopFlag.data());
        return errorSequence;
    }

    ImageBuffer computePredictionErrorData(const ImageBuffer& image, const bool calculateAbs) const override {
        constexpr int RxSize = (this->localSize * (this->localSize + 1)) / 2;
        constexpr int rxSize = this->localSize;

        auto Rx = GpuArray<uint64_t>::zeros(static_cast<unsigned int>(RxSize), stream);
        auto rx = GpuArray<uint64_t>::zeros(static_cast<unsigned int>(rxSize), stream);

        if constexpr (p == 3)
            me_p3<<<gridOptimalMe, meBlockSize.x, 0, stream>>>(image.data(), Rx.data(), rx.data(), this->baseCols, this->baseRows, meParams.x, meParams.y);
        else if constexpr (p == 5)
            me_p5<<<gridOptimalMe, meBlockSize.x, 0, stream>>>(image.data(), Rx.data(), rx.data(), this->baseCols, this->baseRows, meParams.x, meParams.y);
        else if constexpr (p == 7)
            me_p7<<<gridOptimalMe, meBlockSize.x, 0, stream>>>(image.data(), Rx.data(), rx.data(), this->baseCols, this->baseRows, meParams.x, meParams.y);
        else
            me_p9<<<gridOptimalMe, meBlockSize.x, 0, stream>>>(image.data(), Rx.data(), rx.data(), this->baseCols, this->baseRows, meParams.x, meParams.y);
        if constexpr (p <= 5)
            cholesky_solver<p><<<1, 1, 0, stream>>>(Rx.data(), rx.data(), const_cast<float*>(this->coefficients.data()), const_cast<int32_t*>(this->stopFlag.data()));
        else
            cholesky_solver_parallel<p><<<1, 32, 0, stream>>>(Rx.data(), rx.data(), const_cast<float*>(this->coefficients.data()), const_cast<int32_t*>(this->stopFlag.data()));
        return computeErrorSequence(image, calculateAbs);
    }

    ImageBuffer computePredictionErrorMask(const ImageBuffer& errorSequence) const override {
        GpuArray<float> mask(this->baseRows, this->baseCols, stream);
        GpuArray<float> maxVal(1, stream);
        reduceMaxCub(thrust::make_transform_iterator(errorSequence.data(), AbsTransformOp{}), maxVal.data());
        const int gridSize = cuda_utils::gridSize1DStridedCalculate(this->totalPixels, maskNormalizationBlockSize);
        compute_abs_normalized_mask<<<gridSize, maskNormalizationBlockSize, 0, stream>>>(errorSequence.data(), mask.data(), maxVal.data(), this->totalPixels);
        return mask;
    }

    float computeCorrelation(const ImageBuffer& e_u, const ImageBuffer& mask) const override {
        const dim3 gridSize = cuda_utils::gridSizeCalculate(windowBlockSize, this->baseCols, this->baseRows);
        const int numBlocks = gridSize.x * gridSize.y;
        GpuArray<float> dotPartial(static_cast<unsigned int>(numBlocks), stream);
        GpuArray<float> uNormPartial(static_cast<unsigned int>(numBlocks), stream);
        GpuArray<float> zNormPartial(static_cast<unsigned int>(numBlocks), stream);
        calculate_error_sequence_and_partial_corr_fused<p><<<gridSize, windowBlockSize, 0, stream>>>(
            mask.data(), this->randomMatrix.data(), e_u.data(), this->coefficients.data(), dotPartial.data(), uNormPartial.data(), zNormPartial.data(), this->baseCols, this->baseRows,
            this->stopFlag.data());
        GpuArray<float> correlationResult(1, stream);
        calculate_final_correlation<<<1, corrFinalBlockSize, 0, stream>>>(dotPartial.data(), uNormPartial.data(), zNormPartial.data(), correlationResult.data(), numBlocks);
        return correlationResult.scalar();
    }
};
