#pragma once

#include "buffer.hpp"
#include "half_float.hpp"
#include "Eigen/Core"
#include "include/WatermarkTypes.hpp"
#include "PredictionErrorMatrixData.hpp"
#include "WatermarkBase.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <omp.h>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

/*!
 *  \brief  Watermark embedding and detection, Eigen (CPU) implementation.
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
    const int innerRows = endRow - startRow;
    const bool hasInnerRegion = (endRow > startRow) && (endCol > startCol);

    using LocalVector = Eigen::Matrix<float, localSize, 1>;
    using ArrayXXf = Eigen::ArrayXXf;
    using VectorXf = Eigen::VectorXf;
    template <typename T>
    using Map = Eigen::Map<T>;

  public:
    WatermarkEigen<p>(const int rows, const int cols, const std::string& watermarkPassword, const float psnr)
        : WatermarkBase(rows, cols, watermarkPassword, psnr, initializeRandomMatrix), errorSequence(rows, cols), u(rows, cols), predictionData(rows, cols) {}

    // RGB embedding: the watermark is computed from the luma and added to each channel of the 8-bit image, flat images are copied unchanged
    void makeWatermark(const ImageBuffer& inputGrayImage, const ImageOutputBuffer& inputImage, ImageOutputBuffer& output, const MaskMethod maskType) override {
        const auto scale = computeStrengthenedWatermark(inputGrayImage.getGray(), maskType);
        if (!output.matches(baseRows, baseCols, true))
            output = ImageOutputBuffer(EigenArrayU8RGB{Gray8(baseRows, baseCols), Gray8(baseRows, baseCols), Gray8(baseRows, baseCols)});
        const auto& rgbIn = inputImage.getRGB();
        auto& rgbOut = output.getRGB();
        if (!scale) {
            rgbOut = rgbIn;
            return;
        }
#pragma omp parallel for schedule(static)
        for (int col = 0; col < baseCols; col++)
            for (int channel = 0; channel < 3; channel++)
                watermarkColumn(rgbIn[channel].col(col).data(), u.col(col).data(), *scale, rgbOut[channel].col(col).data());
    }

    // grayscale embedding: the watermark is added to the luma itself. The row-major output is the transposed array, written in small tiles (to maximize cache hits)
    void makeWatermark(const ImageBuffer& inputGrayImage, ImageOutputBuffer& output, const MaskMethod maskType, const Layout outputLayout) override {
        const auto& gray = inputGrayImage.getGray();
        const auto scale = computeStrengthenedWatermark(gray, maskType);
        const float strength = scale.value_or(0.0f);
        const bool rowMajor = outputLayout == Layout::RowMajor;
        const int outRows = rowMajor ? baseCols : baseRows;
        const int outCols = rowMajor ? baseRows : baseCols;
        if (!output.matches(outRows, outCols, false))
            output = ImageOutputBuffer(Gray8(outRows, outCols));
        auto& out = output.getGray();
        if (!rowMajor) {
#pragma omp parallel for schedule(static)
            for (int col = 0; col < baseCols; col++) {
                if (scale)
                    watermarkColumn(gray.col(col).data(), u.col(col).data(), strength, out.col(col).data());
                else
                    out.col(col) = toPixels(gray.col(col));
            }
            return;
        }
        constexpr int tile = 64;
        const int rowTiles = (baseRows + tile - 1) / tile;
        const int colTiles = (baseCols + tile - 1) / tile;
#pragma omp parallel for collapse(2) schedule(static)
        for (int rowTile = 0; rowTile < rowTiles; rowTile++) {
            for (int colTile = 0; colTile < colTiles; colTile++) {
                const int row = rowTile * tile;
                const int col = colTile * tile;
                const int rows = std::min(tile, baseRows - row);
                const int cols = std::min(tile, baseCols - col);
                if (scale)
                    out.block(col, row, cols, rows) = toPixels(gray.block(row, col, rows, cols) + u.block(row, col, rows, cols) * strength).transpose();
                else
                    out.block(col, row, cols, rows) = toPixels(gray.block(row, col, rows, cols)).transpose();
            }
        }
    }

    // detection: correlation between the prediction error of the image and the prediction error of (mask * watermark)
    float detectWatermark(const ImageBuffer& inputImage, MaskMethod maskType) override {
        const auto& watermarkedBuffer = inputImage.getGray();
        if (maskType == MaskMethod::NVF) {
            if (!computePredictionErrorData<false>(watermarkedBuffer))
                return 0.0f;
            // NVF mask and u = mask * w in one pass
            computeNvfMaskAndU<false>(watermarkedBuffer, u);
        } else {
            // ME: u =|e| x w. The mask normally should be divided by max|e|: BUT the correlation does not change when u is scaled, the maximum is not needed!
            if (!computePredictionErrorData<false>(watermarkedBuffer))
                return 0.0f;
            const auto& w = randomMatrix.getGray();
#pragma omp parallel for schedule(static)
            for (int i = 0; i < u.size(); i++)
                u(i) = std::abs(errorSequence(i)) * w(i);
        }
        const auto [dot, sqEz, sqEu] = computeDetectionCorrelation(u);
        const float correlation = dot / (std::sqrt(sqEz) * std::sqrt(sqEu));
        return std::isfinite(correlation) ? correlation : 0.0f;
    }

  private:
    using Gray8 = Eigen::Array<uint8_t, Eigen::Dynamic, Eigen::Dynamic>;

    // prediction error of the image, and the strengthened watermark u (embedding) or mask * watermark (detection)
    ArrayXXf errorSequence, u;
    PredictionErrorMatrixData<p> predictionData;

    // watermarked output pixels: round (+ 0.5 then truncation) and clamp to [0, 255]
    template <typename Expr>
    static auto toPixels(const Expr& values) {
        return (values + 0.5f).cwiseMax(0.0f).cwiseMin(255.0f).template cast<uint8_t>();
    }

    // one watermarked column: output = input + strength * u. LOOP because the compiler vectorizes the 8-bit conversions here, Eigen's casts are NOT vectorized!
    // ALSO __restrict and the local row count are needed, else the 8-bit stores could overwrite anything and the compiler would not vectorize...
    template <typename T>
    void watermarkColumn(const T* __restrict input, const float* __restrict uColumn, const float strength, uint8_t* __restrict output) const {
        const int rows = baseRows;
        for (int row = 0; row < rows; row++) {
            const float us = uColumn[row] * strength;
            const float value = static_cast<float>(input[row]) + us;
            output[row] = static_cast<uint8_t>(std::min(std::max(value + 0.5f, 0.0f), 255.0f));
        }
    }

    // creates the watermark buffer: the half precision watermark values as floats
    static WatermarkBuffer initializeRandomMatrix(const std::span<const uint16_t> watermarkHalfBits, const int rows, const int cols) {
        ArrayXXf watermark(rows, cols);
        float* values = watermark.data();
#pragma omp parallel for schedule(static)
        for (int i = 0; i < rows * cols; i++)
            values[i] = HalfFloat::toFloat(watermarkHalfBits[i]);
        return WatermarkBuffer(std::move(watermark));
    }

    // pixel value, coords outside the image are clamped to the nearest edge pixel
    inline float clampedValue(const ArrayXXf& img, int r, int c, const int rows, const int cols) { return img(std::clamp(r, 0, rows - 1), std::clamp(c, 0, cols - 1)); }

    // adds (or removes) one pixel to the running window sums of the NVF mask
    template <Op OP>
    inline void updateWindowSums(const float pixelValue, double& sum, double& sumSq) {
        if constexpr (OP == Op::ADD) {
            sum += pixelValue;
            sumSq += pixelValue * pixelValue;
        } else {
            sum -= pixelValue;
            sumSq -= pixelValue * pixelValue;
        }
    }

    // NVF mask (sliding window variance) and u = mask * w in one pass, optionally returns sum(u^2) for the embedding
    template <bool accumulate>
    float computeNvfMaskAndU(const ArrayXXf& image, ArrayXXf& uOut) {
        float sumSqOut = 0.0f;
        const auto& w = randomMatrix.getGray();
        static constexpr double invPSquared = 1.0 / pSquared;
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
        // the inner region and the border regions in one parallel region
#pragma omp parallel reduction(+ : sumSqOut)
        {
            if (hasInnerRegion) {
#pragma omp for schedule(static) nowait
                for (int j = startCol; j < endCol; j++) {
                    double winSum = 0.0;
                    double winSumSq = 0.0;
                    for (int jj = -pad; jj <= pad; jj++)
                        for (int ii = -pad; ii <= pad; ii++)
                            updateWindowSums<Op::ADD>(image(pad + ii, j + jj), winSum, winSumSq);
                    sumSqOut += emitPixel(winSum, winSumSq, pad, j);
                    // slide the window down the column
                    for (int i = startRow + 1; i < endRow; i++) {
                        // remove the top row, add the new bottom row
                        for (int jj = -pad; jj <= pad; jj++)
                            updateWindowSums<Op::SUB>(image(i - pad - 1, j + jj), winSum, winSumSq);
                        for (int jj = -pad; jj <= pad; jj++)
                            updateWindowSums<Op::ADD>(image(i + pad, j + jj), winSum, winSumSq);
                        sumSqOut += emitPixel(winSum, winSumSq, i, j);
                    }
                }
            }

            // border regions
            auto processRect = [&](int rStart, int rEnd, int cStart, int cEnd) {
#pragma omp for schedule(static) collapse(2) nowait
                for (int j = cStart; j < cEnd; j++) {
                    for (int i = rStart; i < rEnd; i++) {
                        double winSum = 0.0;
                        double winSumSq = 0.0;
                        // border pixels sum the whole (clamped) window
                        for (int jj = -pad; jj <= pad; jj++) {
                            for (int ii = -pad; ii <= pad; ii++) {
                                const float val = clampedValue(image, i + ii, j + jj, baseRows, baseCols);
                                updateWindowSums<Op::ADD>(val, winSum, winSumSq);
                            }
                        }
                        sumSqOut += emitPixel(winSum, winSumSq, i, j);
                    }
                }
            };
            // the 4 border strips
            if (startRow > 0)
                processRect(0, startRow, 0, baseCols);
            if (endRow < baseRows)
                processRect(endRow, baseRows, 0, baseCols);
            if (startCol > 0 && hasInnerRegion)
                processRect(startRow, endRow, 0, startCol);
            if (endCol < baseCols && hasInnerRegion)
                processRect(startRow, endRow, endCol, baseCols);
        }
        return sumSqOut;
    }

    // calls processor(i, j, neighbors) for every border pixel, with the (clamped) window neighbors
    // must be called inside an existing omp parallel region (it uses "omp for")
    template <typename Processor>
    void processBorder(const ArrayXXf& image, Processor&& processor) {
        const int threadId = omp_get_thread_num();
        LocalVector neighbors; // per thread

        auto processRect = [&](int rStart, int rEnd, int cStart, int cEnd) {
        // nowait: threads move on to the next strip without waiting, collapse(2): thin strips are still split across threads
#pragma omp for schedule(static) collapse(2) nowait
            for (int j = cStart; j < cEnd; j++) {
                for (int i = rStart; i < rEnd; i++) {
                    // the window neighbors (clamped at the image edges)
                    int k = 0;
                    for (int dj = 0; dj < p; dj++) {
                        for (int di = 0; di < p; di++) {
                            if (di == center && dj == center)
                                continue;
                            neighbors(k++) = clampedValue(image, i + di - center, j + dj - center, baseRows, baseCols);
                        }
                    }
                    processor(i, j, neighbors, threadId);
                }
            }
        };
        // the 4 border strips
        if (startRow > 0)
            processRect(0, startRow, 0, baseCols);
        if (endRow < baseRows)
            processRect(endRow, baseRows, 0, baseCols);
        if (startCol > 0 && hasInnerRegion)
            processRect(startRow, endRow, 0, startCol);
        if (endCol < baseCols && hasInnerRegion)
            processRect(startRow, endRow, endCol, baseCols);
    }

    // computes u = mask * w (not yet scaled) and returns the scale factor, or nothing for flat images. The scale is applied later while
    // writing the output pixels
    std::optional<float> computeStrengthenedWatermark(const ArrayXXf& inputImage, MaskMethod maskType) {
        float sumSq = 0.0f;
        if (maskType == MaskMethod::NVF) {
            // NVF mask, u = mask * w and sum(u^2) in one pass
            sumSq = computeNvfMaskAndU<true>(inputImage, u);
        } else {
            // ME: the mask is |e| / max|e|, u and sum(u^2) are computed directly from the prediction error
            const auto maxAbsOpt = computePredictionErrorData<true>(inputImage);
            if (!maxAbsOpt || *maxAbsOpt <= 0.0f)
                return std::nullopt;
            const auto& w = randomMatrix.getGray();
            const float invMax = 1.0f / *maxAbsOpt;
            const float* ePtr = errorSequence.data();
#pragma omp parallel for schedule(static) reduction(+ : sumSq)
            for (int i = 0; i < errorSequence.size(); i++) {
                const float uValue = std::abs(ePtr[i]) * invMax * w(i);
                u(i) = uValue;
                sumSq += uValue * uValue;
            }
        }
        if (sumSq <= 1e-3f) // flat images / frames
            return std::nullopt;
        return strengthFactor / std::sqrt(sumSq / totalPixels);
    }

    // computes the prediction coefficients and the prediction error of the image, optionally with its maximum absolute value (ME embedding)
    template <bool FindMax>
    std::optional<float> computePredictionErrorData(const ArrayXXf& image) {
        if (!predictionData.computeCoefficients(image.data()))
            return std::nullopt;
        return computeErrorSequence<FindMax>(image, errorSequence);
    }

    // prediction error of the inner rows of one inner column, 8 neighbors per vectorized expression
    void computeInnerColumnError(const float* imgData, const int colOffset, Map<VectorXf>& columnError) const {
        const auto& coefficients = predictionData.coefficients;
        const auto& offsets = predictionData.neighborOffsets;
        const Map<const VectorXf> imgBatch(imgData + colOffset, innerRows);
        columnError.noalias() =
            imgBatch - (Map<const VectorXf>(imgData + colOffset + offsets[0], innerRows) * coefficients(0) + Map<const VectorXf>(imgData + colOffset + offsets[1], innerRows) * coefficients(1) +
                           Map<const VectorXf>(imgData + colOffset + offsets[2], innerRows) * coefficients(2) + Map<const VectorXf>(imgData + colOffset + offsets[3], innerRows) * coefficients(3) +
                           Map<const VectorXf>(imgData + colOffset + offsets[4], innerRows) * coefficients(4) + Map<const VectorXf>(imgData + colOffset + offsets[5], innerRows) * coefficients(5) +
                           Map<const VectorXf>(imgData + colOffset + offsets[6], innerRows) * coefficients(6) + Map<const VectorXf>(imgData + colOffset + offsets[7], innerRows) * coefficients(7));
        for (int k = 8; k < localSize; k += 8) {
            columnError.noalias() -= (Map<const VectorXf>(imgData + colOffset + offsets[k + 0], innerRows) * coefficients(k + 0) +
                                      Map<const VectorXf>(imgData + colOffset + offsets[k + 1], innerRows) * coefficients(k + 1) +
                                      Map<const VectorXf>(imgData + colOffset + offsets[k + 2], innerRows) * coefficients(k + 2) +
                                      Map<const VectorXf>(imgData + colOffset + offsets[k + 3], innerRows) * coefficients(k + 3) +
                                      Map<const VectorXf>(imgData + colOffset + offsets[k + 4], innerRows) * coefficients(k + 4) +
                                      Map<const VectorXf>(imgData + colOffset + offsets[k + 5], innerRows) * coefficients(k + 5) +
                                      Map<const VectorXf>(imgData + colOffset + offsets[k + 6], innerRows) * coefficients(k + 6) +
                                      Map<const VectorXf>(imgData + colOffset + offsets[k + 7], innerRows) * coefficients(k + 7));
        }
    }

    // prediction error of the image, FindMax returns its maximum abs value
    template <bool FindMax>
    float computeErrorSequence(const ArrayXXf& image, ArrayXXf& outputErrorSequence) {
        const auto& coefficients = predictionData.coefficients;
        float centerMax = 0.0f;
        const float* imgData = image.data();
        float* outData = outputErrorSequence.data();
        // the inner region and the border in one parallel region
#pragma omp parallel
        {
            if (hasInnerRegion) {
                if constexpr (FindMax) {
#pragma omp for schedule(static) reduction(max : centerMax) nowait
                    for (int j = startCol; j < endCol; j++) {
                        const int colOffset = (j * baseRows) + startRow;
                        Map<VectorXf> columnError(outData + colOffset, innerRows);
                        computeInnerColumnError(imgData, colOffset, columnError);
                        centerMax = std::max(centerMax, columnError.cwiseAbs().maxCoeff());
                    }
                } else {
#pragma omp for schedule(static) nowait
                    for (int j = startCol; j < endCol; j++) {
                        const int colOffset = (j * baseRows) + startRow;
                        Map<VectorXf> columnError(outData + colOffset, innerRows);
                        computeInnerColumnError(imgData, colOffset, columnError);
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
            if (startCol > 0 && hasInnerRegion)
                borderMax = std::max(borderMax, outputErrorSequence.block(startRow, 0, innerRows, startCol).abs().maxCoeff());
            if (endCol < baseCols && hasInnerRegion)
                borderMax = std::max(borderMax, outputErrorSequence.block(startRow, endCol, innerRows, baseCols - endCol).abs().maxCoeff());
            return std::max(centerMax, borderMax);
        }
        return 0.0f;
    }

    // correlation sums of the detection: computes the prediction error of "image" (mask * watermark) and correlates it with the image's
    // prediction error, returns {dot, sum(ez^2), sum(eu^2)}
    std::array<float, 3> computeDetectionCorrelation(const ArrayXXf& image) {
        const auto& coefficients = predictionData.coefficients;
        const float* imgData = image.data();
        const float* errorData = errorSequence.data();
        float dot = 0.0f;
        float sqEz = 0.0f;
        float sqEu = 0.0f;
#pragma omp parallel reduction(+ : dot, sqEz, sqEu)
        {
            VectorXf columnBuffer(innerRows);
            if (hasInnerRegion) {
#pragma omp for schedule(static) nowait
                for (int j = startCol; j < endCol; j++) {
                    const int colOffset = (j * baseRows) + startRow;
                    Map<VectorXf> columnError(columnBuffer.data(), innerRows);
                    computeInnerColumnError(imgData, colOffset, columnError);
                    const Map<const VectorXf> sourceError(errorData + colOffset, innerRows);
                    dot += sourceError.dot(columnError);
                    sqEz += sourceError.squaredNorm();
                    sqEu += columnError.squaredNorm();
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
