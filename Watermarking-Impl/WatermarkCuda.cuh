#pragma once
#include "buffer.hpp"
#include "cuda_utils.hpp"
#include "CudaArray.hpp"
#include "CudaStreamManager.hpp"
#include "../CudaCheck.hpp"
#include "include/WatermarkTypes.hpp"
#include "kernels/kernels.cuh"
#include "WatermarkBase.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <string>
#include <thrust/iterator/transform_iterator.h>
#include <vector>

/*!
 *  \brief  Functions for watermark computation and detection, CUDA implementation
 *  \author Dimitris Karatzas
 */
template <int p>
class WatermarkCuda final : public WatermarkBase {
  public:
    WatermarkCuda<p>(const int rows, const int cols, const std::string& watermarkPassword, const float psnr)
        : WatermarkBase(rows, cols, watermarkPassword, psnr, initializeRandomMatrix), stream(CudaStreamManager::getInstance().getComputeStream()), coefficients(localSize, stream),
          stopFlag(FlagBuffer::zeros(1, stream)) {
        if constexpr (p == 3)
            gridOptimalMe = cuda_utils::gridSizeMeCalculate(me_p3, meBlockSize.x);
        else if constexpr (p == 5)
            gridOptimalMe = cuda_utils::gridSizeMeCalculate(me_p5, meBlockSize.x);
        else if constexpr (p == 7)
            gridOptimalMe = cuda_utils::gridSizeMeCalculate(me_p7, meBlockSize.x);
        else
            gridOptimalMe = cuda_utils::gridSizeMeCalculate(me_p9, meBlockSize.x);
        constexpr int pixelsPerBlockY = (p == 3) ? (meBlockSize.x * 2) : meBlockSize.x;
        const int meTotalBlocksY = WatermarkBase::alignUp<pixelsPerBlockY>(this->baseRows) / pixelsPerBlockY;
        meParams = dim3(meTotalBlocksY, meTotalBlocksY * this->baseCols);
        const dim3 corrGrid = cuda_utils::gridSizeCalculate(windowBlockSize, this->baseCols, this->baseRows);
        corrNumBlocks = corrGrid.x * corrGrid.y;
        initializeCubStorage();
    }

    // Embed: compute strengthened watermark u (via NVF or ME mask), then apply to all channels of inputImage
    void makeWatermark(const ImageBuffer& inputGrayImage, const ImageBuffer& inputImage, ImageOutputBuffer& output, const MaskMethod maskType) override {
        // reuse output buffer, only (re)allocate when dimensions or channel count change
        if (output.empty() || output.getRows() != this->baseRows || output.getCols() != this->baseCols || output.getChannels() != inputImage.getChannels())
            output = CudaArray<uint8_t>(this->baseRows, this->baseCols, inputImage.getChannels(), stream);

        CudaArray<float> u(this->baseRows, this->baseCols, stream);
        CudaArray<uint64_t> sumSq = CudaArray<uint64_t>::zeros(1, stream);

        if (maskType == MaskMethod::NVF) {
            // fused NVF: local variance mask x watermark -> strengthened watermark u + sum(u^2)
            const dim3 gridSize = cuda_utils::gridSizeCalculate(windowBlockSize, this->baseCols, this->baseRows);
            nvf_u_and_sumsq_fused<p><<<gridSize, windowBlockSize, 0, stream>>>(inputGrayImage.data(), this->randomMatrix.data(), u.data(), sumSq.data(), this->baseCols, this->baseRows);
            CUDA_CHECK(cudaGetLastError());
        } else {
            // ME path: solve prediction error model, compute error sequence, normalize, fuse with watermark
            constexpr int RxSize = (localSize * (localSize + 1)) / 2;
            constexpr int rxSize = localSize;
            CudaArray<uint64_t> Rx = CudaArray<uint64_t>::zeros(RxSize, stream);
            CudaArray<uint64_t> rx = CudaArray<uint64_t>::zeros(rxSize, stream);
            launchMeKernel(inputGrayImage.data(), Rx, rx);
            launchCholeskySolver(Rx, rx);
            const dim3 errorGridSize = cuda_utils::gridSizeCalculate(windowBlockSize, this->baseCols, this->baseRows);
            CudaArray<float> errorSeq(this->baseRows, this->baseCols, stream);
            calculate_error_sequence<p><<<errorGridSize, windowBlockSize, 0, stream>>>(inputGrayImage.data(), nullptr, errorSeq.data(), this->coefficients.data(), this->baseCols, this->baseRows, true,
                                                                                       this->stopFlag.data());
            CUDA_CHECK(cudaGetLastError());
            // max-reduce for normalization
            CudaArray<float> errorSeqMax(1, stream);
            reduceMaxCub(errorSeq.data(), errorSeqMax.data());
            // fused ME: normalized error x watermark -> strengthened watermark u + sum(u^2)
            const int blocksComputeU = cuda_utils::gridSize1DStridedCalculate(this->totalPixels, strWatermarkBlockSize);
            me_u_and_sumsq_fused<<<blocksComputeU, strWatermarkBlockSize, 0, stream>>>(errorSeq.data(), this->randomMatrix.data(), u.data(), sumSq.data(), errorSeqMax.data(), this->totalPixels);
            CUDA_CHECK(cudaGetLastError());
        }
        // scale u by strength factor and add to each channel of the input image
        const int blocksApply = cuda_utils::gridSize1DStridedCalculate(this->totalPixels, applyWatermarkBlockSize);
        apply_watermark_fused<<<blocksApply, applyWatermarkBlockSize, 0, stream>>>(inputImage.data(), u.data(), sumSq.data(), output.data(), this->strengthNumerator, this->totalPixels,
                                                                                   inputImage.getChannels());
        CUDA_CHECK(cudaGetLastError());
    }

    // Detect: compute prediction error, detection mask, then correlate with watermark
    float detectWatermark(const ImageBuffer& inputImage, const MaskMethod maskType) override {
        // solve prediction error model (Rx, rx -> coefficients via Cholesky)
        constexpr unsigned int RxSize = static_cast<unsigned int>((localSize * (localSize + 1)) / 2);
        constexpr unsigned int rxSize = static_cast<unsigned int>(localSize);
        CudaArray<uint64_t> Rx = CudaArray<uint64_t>::zeros(RxSize, stream);
        CudaArray<uint64_t> rx = CudaArray<uint64_t>::zeros(rxSize, stream);
        launchMeKernel(inputImage.data(), Rx, rx);
        launchCholeskySolver(Rx, rx);

        const dim3 windowGrid = cuda_utils::gridSizeCalculate(windowBlockSize, this->baseCols, this->baseRows);

        // compute prediction error sequence (non-abs, needed for correlation sign)
        CudaArray<float> errorSeq(this->baseRows, this->baseCols, stream);
        calculate_error_sequence<p>
            <<<windowGrid, windowBlockSize, 0, stream>>>(inputImage.data(), nullptr, errorSeq.data(), this->coefficients.data(), this->baseCols, this->baseRows, false, this->stopFlag.data());
        CUDA_CHECK(cudaGetLastError());

        // compute detection mask (ME: abs-normalized error, NVF: local variance)
        CudaArray<float> mask(this->baseRows, this->baseCols, stream);
        CudaArray<float> errorSeqMax(1, stream);
        if (maskType == MaskMethod::ME) {
            reduceMaxCub(thrust::make_transform_iterator(errorSeq.data(), AbsTransformOp{}), errorSeqMax.data());
            const int gridSize = cuda_utils::gridSize1DStridedCalculate(this->totalPixels, maskNormalizationBlockSize);
            compute_abs_normalized_mask<<<gridSize, maskNormalizationBlockSize, 0, stream>>>(errorSeq.data(), mask.data(), errorSeqMax.data(), this->totalPixels);
            CUDA_CHECK(cudaGetLastError());
        } else {
            nvf<p><<<windowGrid, windowBlockSize, 0, stream>>>(inputImage.data(), mask.data(), this->baseCols, this->baseRows);
            CUDA_CHECK(cudaGetLastError());
        }

        // fused: recompute error sequence from (mask x watermark), accumulate partial dot / normU / normZ
        CudaArray<float> dotPartial(corrNumBlocks, stream);
        CudaArray<float> uNormPartial(corrNumBlocks, stream);
        CudaArray<float> zNormPartial(corrNumBlocks, stream);
        calculate_error_sequence_and_partial_corr_fused<p><<<windowGrid, windowBlockSize, 0, stream>>>(mask.data(), this->randomMatrix.data(), errorSeq.data(), this->coefficients.data(),
                                                                                                       dotPartial.data(), uNormPartial.data(), zNormPartial.data(), this->baseCols, this->baseRows,
                                                                                                       this->stopFlag.data());
        CUDA_CHECK(cudaGetLastError());
        // reduce partials -> final normalized correlation
        CudaArray<float> corrResult(1, stream);
        calculate_final_correlation<<<1, corrFinalBlockSize, 0, stream>>>(dotPartial.data(), uNormPartial.data(), zNormPartial.data(), corrResult.data(), corrNumBlocks);
        CUDA_CHECK(cudaGetLastError());

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
    ImageBuffer coefficients;
    FlagBuffer stopFlag;

    dim3 meParams;
    int gridOptimalMe;
    int corrNumBlocks;
    CudaArray<uint8_t> cubTempStorage;

    static ImageBuffer initializeRandomMatrix(const std::vector<float>& watermarkVec, const int rows, const int cols) {
        return ImageBuffer(rows, cols, watermarkVec.data(), CudaStreamManager::getInstance().getComputeStream());
    }

    void initializeCubStorage() {
        AbsTransformOp op;
        auto iter = thrust::make_transform_iterator((const float*)nullptr, op);
        size_t tmpBytesTransform = 0;
        CUDA_CHECK(cub::DeviceReduce::Max(nullptr, tmpBytesTransform, iter, (float*)nullptr, this->totalPixels, 0));
        size_t tmpBytesRaw = 0;
        CUDA_CHECK(cub::DeviceReduce::Max(nullptr, tmpBytesRaw, (const float*)nullptr, (float*)nullptr, this->totalPixels, 0));
        cubTempStorage = CudaArray<uint8_t>(static_cast<int>(std::max(tmpBytesTransform, tmpBytesRaw)), stream);
    }

    template <typename InputIteratorT>
    void reduceMaxCub(InputIteratorT in, float* out) const {
        size_t tmpStorageBytes = cubTempStorage.bytes();
        CUDA_CHECK(cub::DeviceReduce::Max(const_cast<uint8_t*>(cubTempStorage.data()), tmpStorageBytes, in, out, this->totalPixels, stream));
    }

    // dispatch the correct ME kernel variant based on prediction order p
    void launchMeKernel(const float* imageData, CudaArray<uint64_t>& Rx, CudaArray<uint64_t>& rx) {
        if constexpr (p == 3)
            me_p3<<<gridOptimalMe, meBlockSize.x, 0, stream>>>(imageData, Rx.data(), rx.data(), this->baseCols, this->baseRows, meParams.x, meParams.y);
        else if constexpr (p == 5)
            me_p5<<<gridOptimalMe, meBlockSize.x, 0, stream>>>(imageData, Rx.data(), rx.data(), this->baseCols, this->baseRows, meParams.x, meParams.y);
        else if constexpr (p == 7)
            me_p7<<<gridOptimalMe, meBlockSize.x, 0, stream>>>(imageData, Rx.data(), rx.data(), this->baseCols, this->baseRows, meParams.x, meParams.y);
        else
            me_p9<<<gridOptimalMe, meBlockSize.x, 0, stream>>>(imageData, Rx.data(), rx.data(), this->baseCols, this->baseRows, meParams.x, meParams.y);
        CUDA_CHECK(cudaGetLastError());
    }

    // solve Rx*a = rx via Cholesky decomposition to get prediction coefficients
    void launchCholeskySolver(CudaArray<uint64_t>& Rx, CudaArray<uint64_t>& rx) {
        if constexpr (p <= 5)
            cholesky_solver<p><<<1, 1, 0, stream>>>(Rx.data(), rx.data(), this->coefficients.data(), this->stopFlag.data());
        else
            cholesky_solver_parallel<p><<<1, 32, 0, stream>>>(Rx.data(), rx.data(), this->coefficients.data(), this->stopFlag.data());
        CUDA_CHECK(cudaGetLastError());
    }
};
