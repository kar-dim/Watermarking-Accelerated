#pragma once

#include "buffer.hpp"
#include "Eigen/Core"
#include "include/WatermarkTypes.hpp"
#include "PredictionErrorMatrixData.hpp"
#include "WatermarkBase.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <omp.h>
#include <optional>
#include <string>
#include <vector>

/*!
 *  \brief  Functions for watermark computation and detection, Eigen implementation.
 *  \author Dimitris Karatzas
 */
template <int p>
class WatermarkEigen final : public WatermarkBase {
  private:
    enum class Op { ADD, SUB };

    static constexpr int pSquared = p * p;
    static constexpr int pad = p / 2;
    static constexpr int localSize = pSquared - 1;
    static constexpr int startRow = pad;
    static constexpr int startCol = pad;
    static constexpr int center = pad;

    const int endRow = baseRows - pad;
    const int endCol = baseCols - pad;
    const int stripHeight = endRow - startRow;
    const bool hasCenterRegion = (endRow > startRow) && (endCol > startCol);

    using LocalVector = Eigen::Matrix<float, localSize, 1>;
    using ArrayXXf = Eigen::ArrayXXf;
    using VectorXf = Eigen::VectorXf;
    template <typename T>
    using Map = Eigen::Map<T>;

  public:
    WatermarkEigen<p>(const int rows, const int cols, const std::string& watermarkPassword, const float psnr)
        : WatermarkBase(rows, cols, watermarkPassword, psnr, initializeRandomMatrix), errorSequence(rows, cols), u(rows, cols), meMatrixData(omp_get_max_threads(), rows) {}

    // main watermark embedding method
    void makeWatermark(const ImageBuffer& inputGrayImage, const ImageBuffer& inputImage, ImageOutputBuffer& output, const MaskMethod maskType) override {
        // compute the unscaled strengthened watermark and its scale factor, if it fails assign input to output and return
        const auto scale = computeStrengthenedWatermark(inputGrayImage.getGray(), maskType);
        if (!scale) {
            inputImage.assignTo(output);
            return;
        }
        // embed the watermark into the input image
        inputImage.applyWatermark(u, *scale, output);
    }

    // main watermark detection method
    float detectWatermark(const ImageBuffer& inputImage, MaskMethod maskType) override {
        const auto& watermarkedBuffer = inputImage.getGray();
        if (maskType == MaskMethod::NVF) {
            if (!computePredictionErrorData<false>(watermarkedBuffer))
                return 0.0f;
            // fused: NVF mask + (u = mask*w)
            computeCustomMaskFused<false>(watermarkedBuffer, u);
        } else {
            // ME detect uses fused computations
            const auto maxAbsOpt = computePredictionErrorData<true>(watermarkedBuffer);
            if (!maxAbsOpt)
                return 0.0f;
            const auto& w = randomMatrix.getGray();
            const float invMax = (*maxAbsOpt > 0.0f) ? (1.0f / *maxAbsOpt) : 0.0f;
#pragma omp parallel for schedule(static)
            for (int i = 0; i < u.size(); i++)
                u(i) = std::abs(errorSequence(i)) * invMax * w(i);
        }
        const auto [dot, sqEz, sqEu] = computeDetectionCorrelation(u);
        const float correlation = dot / (std::sqrt(sqEz) * std::sqrt(sqEu));
        return std::isfinite(correlation) ? correlation : 0.0f;
    }

  private:
    // embedding and detection use this watermark plane at separate times (reused)
    ArrayXXf errorSequence, u;
    PredictionErrorMatrixData<p> meMatrixData;

    // initialize the watermark random matrix into an Eigen buffer
    static ImageBuffer initializeRandomMatrix(const std::vector<float>& watermarkVec, const int rows, const int cols) {
        return ImageBuffer(ArrayXXf(Map<const ArrayXXf>(watermarkVec.data(), rows, cols)));
    }

    // helper method to clamp the pixel value to the image boundaries if out of bounds
    inline float clampedValue(const ArrayXXf& img, int r, int c, const int rows, const int cols) { return img(std::clamp(r, 0, rows - 1), std::clamp(c, 0, cols - 1)); }

    // helper method for custom mask sums accumulation
    template <Op OP>
    inline void computeCustomMaskSums(const float pixelValue, double& sum, double& sumSq) {
        if constexpr (OP == Op::ADD) {
            sum += pixelValue;
            sumSq += pixelValue * pixelValue;
        } else {
            sum -= pixelValue;
            sumSq -= pixelValue * pixelValue;
        }
    }

    // fused NVF mask computation: sliding window mask calc + u = mask*w, optionally accumulating sum of squares for the embedding
    template <bool accumulate>
    float computeCustomMaskFused(const ArrayXXf& image, ArrayXXf& uOut) {
        float sumSqOut = 0.0f;
        const auto& w = randomMatrix.getGray();
        static constexpr double invPSquared = 1.0 / pSquared; // precompute the inverse to avoid division in the loop
        auto emitPixel = [&](const double winSum, const double winSumSq, const int i, const int j) -> float {
            const double mean = winSum * invPSquared;
            const double variance = (winSumSq * invPSquared) - (mean * mean);
            const double maskValue = variance / (1.0 + variance);
            const float m = std::clamp(static_cast<float>(maskValue), 0.0f, 1.0f);
            const float uVal = m * w(i, j);
            uOut(i, j) = uVal;
            if constexpr (accumulate)
                return uVal * uVal;
            else
                return 0.0f;
        };
        // process CENTER and BORDER regions in a single thread team
#pragma omp parallel reduction(+ : sumSqOut)
        {
            if (hasCenterRegion) {
#pragma omp for schedule(static) nowait
                for (int j = startCol; j < endCol; j++) {
                    double winSum = 0.0;
                    double winSumSq = 0.0;
                    for (int jj = -pad; jj <= pad; jj++)
                        for (int ii = -pad; ii <= pad; ii++)
                            computeCustomMaskSums<Op::ADD>(image(pad + ii, j + jj), winSum, winSumSq);
                    sumSqOut += emitPixel(winSum, winSumSq, pad, j);
                    // slide window down for remaining center rows in this column
                    for (int i = startRow + 1; i < endRow; i++) {
                        // remove top row and add new bottom row
                        for (int jj = -pad; jj <= pad; jj++)
                            computeCustomMaskSums<Op::SUB>(image(i - pad - 1, j + jj), winSum, winSumSq);
                        for (int jj = -pad; jj <= pad; jj++)
                            computeCustomMaskSums<Op::ADD>(image(i + pad, j + jj), winSum, winSumSq);
                        sumSqOut += emitPixel(winSum, winSumSq, i, j);
                    }
                }
            }

            // process BORDER regions
            auto processRect = [&](int rStart, int rEnd, int cStart, int cEnd) {
#pragma omp for schedule(static) collapse(2) nowait
                for (int j = cStart; j < cEnd; j++) {
                    for (int i = rStart; i < rEnd; i++) {
                        double winSum = 0.0;
                        double winSumSq = 0.0;
                        // for borders, we cannot slide, so we do the full O(p^2) sum
                        for (int jj = -pad; jj <= pad; jj++) {
                            for (int ii = -pad; ii <= pad; ii++) {
                                const float val = clampedValue(image, i + ii, j + jj, baseRows, baseCols);
                                computeCustomMaskSums<Op::ADD>(val, winSum, winSumSq);
                            }
                        }
                        sumSqOut += emitPixel(winSum, winSumSq, i, j);
                    }
                }
            };
            // feed the 4 border strips to the active thread team
            if (startRow > 0)
                processRect(0, startRow, 0, baseCols);
            if (endRow < baseRows)
                processRect(endRow, baseRows, 0, baseCols);
            if (startCol > 0 && hasCenterRegion)
                processRect(startRow, endRow, 0, startCol);
            if (endCol < baseCols && hasCenterRegion)
                processRect(startRow, endRow, endCol, baseCols);
        }
        return sumSqOut;
    }

    // helper method to process the border pixels (clamp if out of bounds) by using a supplied function
    // must be called from within an existing omp parallel region (uses "omp for" directly)
    template <typename Processor>
    void processBorder(const ArrayXXf& image, Processor&& processor) {
        const int threadId = omp_get_thread_num();
        LocalVector neighbors; // per thread

        auto processRect = [&](int rStart, int rEnd, int cStart, int cEnd) {
        // nowait: threads jump between borders immediately
        // collapse(2): handles thin strips efficiently
#pragma omp for schedule(static) collapse(2) nowait
            for (int j = cStart; j < cEnd; j++) {
                for (int i = rStart; i < rEnd; i++) {
                    // collect neighbors (clamped to avoid out of bounds)
                    int k = 0;
                    for (int dj = 0; dj < p; dj++) {
                        for (int di = 0; di < p; di++) {
                            if (di == center && dj == center)
                                continue;
                            neighbors(k++) = clampedValue(image, i + di - center, j + dj - center, baseRows, baseCols);
                        }
                    }
                    // execute the processing function
                    processor(i, j, neighbors, threadId);
                }
            }
        };
        // feed the 4 border strips
        if (startRow > 0)
            processRect(0, startRow, 0, baseCols);
        if (endRow < baseRows)
            processRect(endRow, baseRows, 0, baseCols);
        if (startCol > 0 && hasCenterRegion)
            processRect(startRow, endRow, 0, startCol);
        if (endCol < baseCols && hasCenterRegion)
            processRect(startRow, endRow, endCol, baseCols);
    }

    // compute the unscaled strengthened watermark u = mask * w and return its scale factor on success.
    // The scale is intentionally NOT applied here, it is fused later (in ImageEigenBuffer::processOutput).
    // for ME we fuse (abs(errorSequence)/maxAbs)*w + sumSq, skipping two full image passes
    std::optional<float> computeStrengthenedWatermark(const ArrayXXf& inputImage, MaskMethod maskType) {
        float sumSq = 0.0f;
        if (maskType == MaskMethod::NVF) {
            // fused NVF mask + u = mask*w + sumSq accumulation
            sumSq = computeCustomMaskFused<true>(inputImage, u);
        } else {
            // ME: skip mask creation entirely, populate errorSequence and its max abs, fuse (abs(e)*invMax)*w
            const auto maxAbsOpt = computePredictionErrorData<true>(inputImage);
            if (!maxAbsOpt || *maxAbsOpt <= 0.0f)
                return std::nullopt;
            const auto& w = randomMatrix.getGray();
            const float invMax = 1.0f / *maxAbsOpt;
            const float* ePtr = errorSequence.data();
#pragma omp parallel for schedule(static) reduction(+ : sumSq)
            for (int i = 0; i < errorSequence.size(); i++) {
                // mask is calculated inline here, helps calculate u directly
                const float uValue = std::abs(ePtr[i]) * invMax * w(i);
                u(i) = uValue;
                sumSq += uValue * uValue;
            }
        }
        if (sumSq <= 1e-3f) // for flat images/frames
            return std::nullopt;
        return strengthFactor / std::sqrt(sumSq / totalPixels);
    }

    // compute prediction coefficients and the first error sequence, with an optional maximum for ME normalization
    template <bool FindMax>
    std::optional<float> computePredictionErrorData(const ArrayXXf& image) {
        meMatrixData.setZero();
        const float* imgData = image.data();
        const auto& offsets = meMatrixData.offsets;
        // process CENTER and BORDER in one parallel team to eliminate a team launch
#pragma omp parallel
        {
            const int threadId = omp_get_thread_num();
            auto& RxLocal = meMatrixData.RxAll[threadId].mat;
            auto& rxLocal = meMatrixData.rxAll[threadId].mat;
            if (hasCenterRegion) {
                if constexpr (p == 3) {
                    // small localSize (8): dot-product loops, upper-triangle accumulation
#pragma omp for schedule(static) nowait
                    for (int j = startCol; j < endCol; j++) {
                        const int colOffset = j * baseRows;
                        const float* centerPtr = imgData + colOffset + startRow;
                        const Map<const VectorXf> centerBatch(centerPtr, stripHeight);
                        for (int u = 0; u < localSize; u++) {
                            const Map<const VectorXf> neighborBatch(centerPtr + offsets[u], stripHeight);
                            rxLocal(u) += neighborBatch.dot(centerBatch);
                            for (int v = 0; v <= u; v++)
                                RxLocal(v, u) += neighborBatch.dot(Map<const VectorXf>(centerPtr + offsets[v], stripHeight));
                        }
                    }
                } else {
                    // large localSize (24, 48 and 80): neighbor matrix + SSYRK is faster
                    // pre-allocated at construction (per thread), reused across frames and columns
                    auto& neighborMatrix = meMatrixData.neighborMatricesAll[threadId].mat;
                    Eigen::Matrix<float, localSize, localSize> columnRx;
                    LocalVector columnRxVector;
#pragma omp for schedule(static) nowait
                    for (int j = startCol; j < endCol; j++) {
                        const int colOffset = j * baseRows;
                        const float* centerPtr = imgData + colOffset + startRow;
                        const Map<const VectorXf> centerBatch(centerPtr, stripHeight);
                        for (int u = 0; u < localSize; u++)
                            neighborMatrix.col(u) = Map<const VectorXf>(centerPtr + offsets[u], stripHeight);
                        columnRx.setZero();
                        columnRx.template selfadjointView<Eigen::Upper>().rankUpdate(neighborMatrix.transpose());
                        RxLocal += columnRx.template cast<double>();
                        columnRxVector.noalias() = neighborMatrix.transpose() * centerBatch;
                        rxLocal += columnRxVector.template cast<double>();
                    }
                }
            }
            processBorder(image, [&](const int i, const int j, const LocalVector& neighbors, const int threadId) { meMatrixData.computePredictionErrorMatrices(neighbors, image(i, j), threadId); });
        }

        // solve system and coefficients
        if (!meMatrixData.computeCoefficients())
            return std::nullopt;
        // calculate ex(i,j) and, when requested, its maximum absolute value in the same pass
        return computeErrorSequence<FindMax>(image, errorSequence);
    }

    // evaluate one center column into a reusable destination, keeping each eight-neighbor expression vectorized
    void computeCenterErrorBatch(const float* imgData, const int colOffset, Map<VectorXf>& errorBatch) const {
        const auto& coefficients = meMatrixData.coefficients;
        const auto& offsets = meMatrixData.offsets;
        const Map<const VectorXf> imgBatch(imgData + colOffset, stripHeight);
        errorBatch.noalias() =
            imgBatch - (Map<const VectorXf>(imgData + colOffset + offsets[0], stripHeight) * coefficients(0) + Map<const VectorXf>(imgData + colOffset + offsets[1], stripHeight) * coefficients(1) +
                           Map<const VectorXf>(imgData + colOffset + offsets[2], stripHeight) * coefficients(2) + Map<const VectorXf>(imgData + colOffset + offsets[3], stripHeight) * coefficients(3) +
                           Map<const VectorXf>(imgData + colOffset + offsets[4], stripHeight) * coefficients(4) + Map<const VectorXf>(imgData + colOffset + offsets[5], stripHeight) * coefficients(5) +
                           Map<const VectorXf>(imgData + colOffset + offsets[6], stripHeight) * coefficients(6) + Map<const VectorXf>(imgData + colOffset + offsets[7], stripHeight) * coefficients(7));
        for (int k = 8; k < localSize; k += 8) {
            errorBatch.noalias() -= (Map<const VectorXf>(imgData + colOffset + offsets[k + 0], stripHeight) * coefficients(k + 0) +
                                     Map<const VectorXf>(imgData + colOffset + offsets[k + 1], stripHeight) * coefficients(k + 1) +
                                     Map<const VectorXf>(imgData + colOffset + offsets[k + 2], stripHeight) * coefficients(k + 2) +
                                     Map<const VectorXf>(imgData + colOffset + offsets[k + 3], stripHeight) * coefficients(k + 3) +
                                     Map<const VectorXf>(imgData + colOffset + offsets[k + 4], stripHeight) * coefficients(k + 4) +
                                     Map<const VectorXf>(imgData + colOffset + offsets[k + 5], stripHeight) * coefficients(k + 5) +
                                     Map<const VectorXf>(imgData + colOffset + offsets[k + 6], stripHeight) * coefficients(k + 6) +
                                     Map<const VectorXf>(imgData + colOffset + offsets[k + 7], stripHeight) * coefficients(k + 7));
        }
    }

    // compute the first prediction error sequence, skipping the maximum reduction when detection does not use it
    template <bool FindMax>
    float computeErrorSequence(const ArrayXXf& image, ArrayXXf& outputErrorSequence) {
        const auto& coefficients = meMatrixData.coefficients;
        float centerMax = 0.0f;
        const float* imgData = image.data();
        float* outData = outputErrorSequence.data();
        // process CENTER and BORDER in one parallel team to eliminate a team launch
#pragma omp parallel
        {
            if (hasCenterRegion) {
                if constexpr (FindMax) {
#pragma omp for schedule(static) reduction(max : centerMax) nowait
                    for (int j = startCol; j < endCol; j++) {
                        const int colOffset = (j * baseRows) + startRow;
                        Map<VectorXf> errorBatch(outData + colOffset, stripHeight);
                        computeCenterErrorBatch(imgData, colOffset, errorBatch);
                        centerMax = std::max(centerMax, errorBatch.cwiseAbs().maxCoeff());
                    }
                } else {
#pragma omp for schedule(static) nowait
                    for (int j = startCol; j < endCol; j++) {
                        const int colOffset = (j * baseRows) + startRow;
                        Map<VectorXf> errorBatch(outData + colOffset, stripHeight);
                        computeCenterErrorBatch(imgData, colOffset, errorBatch);
                    }
                }
            }
            processBorder(image, [&](const int i, const int j, const LocalVector& neighbors, const int) { outputErrorSequence(i, j) = image(i, j) - neighbors.dot(coefficients); });
        }
        if constexpr (FindMax) {
            float borderMax = 0.0f;
            if (startRow > 0)
                borderMax = std::max(borderMax, outputErrorSequence.topRows(startRow).abs().maxCoeff());
            if (endRow < baseRows)
                borderMax = std::max(borderMax, outputErrorSequence.bottomRows(baseRows - endRow).abs().maxCoeff());
            if (startCol > 0 && hasCenterRegion)
                borderMax = std::max(borderMax, outputErrorSequence.block(startRow, 0, stripHeight, startCol).abs().maxCoeff());
            if (endCol < baseCols && hasCenterRegion)
                borderMax = std::max(borderMax, outputErrorSequence.block(startRow, endCol, stripHeight, baseCols - endCol).abs().maxCoeff());
            return std::max(centerMax, borderMax);
        }
        return 0.0f;
    }

    // accumulate detection correlation while generating the second prediction error sequence
    std::array<float, 3> computeDetectionCorrelation(const ArrayXXf& image) {
        const auto& coefficients = meMatrixData.coefficients;
        const float* imgData = image.data();
        const float* errorData = errorSequence.data();
        float dot = 0.0f;
        float sqEz = 0.0f;
        float sqEu = 0.0f;
#pragma omp parallel reduction(+ : dot, sqEz, sqEu)
        {
            VectorXf columnError(stripHeight);
            if (hasCenterRegion) {
#pragma omp for schedule(static) nowait
                for (int j = startCol; j < endCol; j++) {
                    const int colOffset = (j * baseRows) + startRow;
                    Map<VectorXf> errorBatch(columnError.data(), stripHeight);
                    computeCenterErrorBatch(imgData, colOffset, errorBatch);
                    const Map<const VectorXf> sourceError(errorData + colOffset, stripHeight);
                    dot += sourceError.dot(errorBatch);
                    sqEz += sourceError.squaredNorm();
                    sqEu += errorBatch.squaredNorm();
                }
            }
            processBorder(image, [&](const int i, const int j, const LocalVector& neighbors, const int) {
                const float eu = image(i, j) - neighbors.dot(coefficients);
                const float ez = errorSequence(i, j);
                dot += ez * eu;
                sqEz += ez * ez;
                sqEu += eu * eu;
            });
        }
        return {dot, sqEz, sqEu};
    }
};
