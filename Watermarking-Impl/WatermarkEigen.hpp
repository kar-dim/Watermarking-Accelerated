#pragma once

#include "buffer.hpp"
#include "Eigen/Core"
#include "PredictionErrorMatrixData.hpp"
#include "WatermarkBase.hpp"
#include <algorithm>
#include <cmath>
#include <omp.h>
#include <string>

/*!
 *  \brief  Functions for watermark computation and detection, Eigen implementation.
 *  \author Dimitris Karatzas
 */
template <int p> class WatermarkEigen final : public WatermarkBase {
  private:
    enum class Op { ADD, SUB };
    static constexpr int pSquared = p * p;
    static constexpr int pad = p / 2;
    static constexpr int localSize = pSquared - 1;
    static constexpr int startRow = pad, startCol = pad, center = pad;
    const int endRow = baseRows - pad;
    const int endCol = baseCols - pad;
    const int stripHeight = endRow - startRow;
    const bool hasCenterRegion = (endRow > startRow) && (endCol > startCol);
    using LocalVector = Eigen::Matrix<float, localSize, 1>;
    using ArrayXXf = Eigen::ArrayXXf;
    using VectorXf = Eigen::VectorXf;
    template <typename T> using Map = Eigen::Map<T>;

  public:
    WatermarkEigen<p>(const unsigned int rows, const unsigned int cols, const std::string& randomMatrixPath, const float psnr)
        : WatermarkBase(rows, cols, randomMatrixPath, psnr), mask(rows, cols), errorSequence(rows, cols), filteredEstimation(rows, cols), u(rows, cols), uStrengthened(rows, cols),
          meMatrixData(omp_get_max_threads(), rows) {}

    // main watermark embedding method
    void makeWatermark(const ImageBuffer& inputGrayImage, const ImageBuffer& inputImage, ImageOutputBuffer& output, float& watermarkStrength, const MASK_TYPE maskType) override {
        // compute the strengthened watermark, if it fails assign input to output and return
        if (!computeStrengthenedWatermark(inputGrayImage.getGray(), watermarkStrength, maskType)) {
            watermarkStrength = 0.0f;
            inputImage.assignTo(output);
            return;
        }
        // embed the watermark into the input image
        inputImage.applyWatermark(uStrengthened, output);
    }

    // main watermark detection method
    float detectWatermark(const ImageBuffer& inputImage, MASK_TYPE maskType) override {
        const auto& watermarkedBuffer = inputImage.getGray();
        if (maskType == NVF) {
            if (!computePredictionErrorData<maskCalcNotRequired>(watermarkedBuffer))
                return 0.0f;
            computeCustomMask(watermarkedBuffer);
        } else {
            if (!computePredictionErrorData<maskCalcRequired>(watermarkedBuffer))
                return 0.0f;
        }
        u = mask * randomMatrix.getGray();
        computeErrorSequence(u, filteredEstimation);
        // optimized and fused correlation calculation using Eigen and OpenMP
        float globalDot = 0.0;
        float globalSqEz = 0.0;
        float globalSqEu = 0.0;
        const float* ezPtr = errorSequence.data();
        const float* euPtr = filteredEstimation.data();
        const auto totalPixels = errorSequence.size();

#pragma omp parallel reduction(+ : globalDot, globalSqEz, globalSqEu)
        {
            const int numThreads = omp_get_num_threads();
            const int tid = omp_get_thread_num();
            const auto chunkSize = totalPixels / numThreads;
            const auto start = tid * chunkSize;
            const auto actualSize = (tid == numThreads - 1) ? (totalPixels - start) : chunkSize;
            if (actualSize > 0) {
                Eigen::Map<const Eigen::VectorXf> ezVec(ezPtr + start, actualSize);
                Eigen::Map<const Eigen::VectorXf> euVec(euPtr + start, actualSize);
                globalDot += ezVec.dot(euVec);
                globalSqEz += ezVec.squaredNorm();
                globalSqEu += euVec.squaredNorm();
            }
        }
        const float correlation = globalDot / (std::sqrt(globalSqEz) * std::sqrt(globalSqEu));
        return std::isfinite(correlation) ? correlation : 0.0f;
    }

  private:
    ArrayXXf mask, errorSequence, filteredEstimation, u, uStrengthened;
    PredictionErrorMatrixData<p> meMatrixData;

    // helper method to clamp the pixel value to the image boundaries if out of bounds
    inline float clampedValue(const ArrayXXf& img, int r, int c, const int rows, const int cols) { return img(std::clamp(r, 0, rows - 1), std::clamp(c, 0, cols - 1)); }

    // helper method for custom mask sums accumulation
    template <Op OP> inline void computeCustomMaskSums(const float pixelValue, double& sum, double& sumSq) {
        if constexpr (OP == Op::ADD) {
            sum += pixelValue;
            sumSq += pixelValue * pixelValue;
        } else {
            sum -= pixelValue;
            sumSq -= pixelValue * pixelValue;
        }
    }

    // helper method for custom mask calculation per pixel
    inline void computeCustomMaskPixel(const double sum, const double sumSq, const int i, const int j) {
        const double mean = sum / pSquared;
        const double variance = (sumSq / pSquared) - (mean * mean);
        const double maskValue = variance / (1.0 + variance);
        mask(i, j) = std::clamp(static_cast<float>(maskValue), 0.0f, 1.0f);
    }

    // helper method for calling lambda border handlers for both custom and prediction error masks
    template <typename ProcessBorderFunc>
    inline void computeMaskBorders(const int startRow, const int endRow, const int startCol, const int endCol, bool hasCenterRegion, const ProcessBorderFunc& processBorder) {
        if (startRow > 0)
            processBorder(0, startRow, 0, baseCols);
        if (endRow < baseRows)
            processBorder(endRow, baseRows, 0, baseCols);
        if (startCol > 0 && hasCenterRegion)
            processBorder(startRow, endRow, 0, startCol);
        if (endCol < baseCols && hasCenterRegion)
            processBorder(startRow, endRow, endCol, baseCols);
    }

    // main method to compute the custom mask
    void computeCustomMask(const ArrayXXf& image) {
        // process CENTER region
        if (hasCenterRegion) {
#pragma omp parallel for
            for (int j = startCol; j < endCol; j++) {
                double sum = 0.0, sumSq = 0.0;
                for (int jj = -pad; jj <= pad; jj++)
                    for (int ii = -pad; ii <= pad; ii++)
                        computeCustomMaskSums<Op::ADD>(image(pad + ii, j + jj), sum, sumSq);
                computeCustomMaskPixel(sum, sumSq, pad, j);
                // slide window down for remaining center rows in this column
                for (int i = startRow + 1; i < endRow; i++) {
                    // remove top row and add new bottom row
                    for (int jj = -pad; jj <= pad; jj++)
                        computeCustomMaskSums<Op::SUB>(image(i - pad - 1, j + jj), sum, sumSq);
                    for (int jj = -pad; jj <= pad; jj++)
                        computeCustomMaskSums<Op::ADD>(image(i + pad, j + jj), sum, sumSq);
                    computeCustomMaskPixel(sum, sumSq, i, j);
                }
            }
        }

        // process BORDER regions
        auto processBorder = [&](const int startRow, const int endRow, int const startCol, const int endCol) {
#pragma omp parallel for collapse(2) schedule(dynamic, 8)
            for (int j = startCol; j < endCol; j++)
                for (int i = startRow; i < endRow; i++) {
                    double sum = 0.0, sumSq = 0.0;
                    for (int jj = -pad; jj <= pad; jj++)
                        for (int ii = -pad; ii <= pad; ii++)
                            computeCustomMaskSums<Op::ADD>(clampedValue(image, i + ii, j + jj, baseRows, baseCols), sum, sumSq);
                    computeCustomMaskPixel(sum, sumSq, i, j);
                }
        };
        computeMaskBorders(startRow, endRow, startCol, endCol, hasCenterRegion, processBorder);
    }

    // helper method to process the border pixels (clamp if out of bounds)
    // by using a supplied function (used in prediction error correlation and error sequence calculation)
    template <typename BorderFunc> inline void processPredictionErrorBorder(const ArrayXXf& image, const int startRow, const int endRow, const int startCol, const int endCol, BorderFunc&& func) {
#pragma omp parallel
        {
            const int threadId = omp_get_thread_num();
            LocalVector neighbors;
#pragma omp for collapse(2)
            for (int j = startCol; j < endCol; j++)
                for (int i = startRow; i < endRow; i++) {
                    int k = 0;
                    for (int dj = 0; dj < p; dj++)
                        for (int di = 0; di < p; di++) {
                            if (di == center && dj == center)
                                continue;
                            neighbors(k++) = clampedValue(image, i + di - center, j + dj - center, baseRows, baseCols);
                        }
                    func(i, j, neighbors, threadId);
                }
        }
    }

    // compute the strengthened watermark, calculated by multiplying the mask with the strengthened watermark (random matrix)
    bool computeStrengthenedWatermark(const ArrayXXf& inputImage, float& watermarkStrength, MASK_TYPE maskType) {
        if (maskType == NVF)
            computeCustomMask(inputImage);
        else {
            if (!computePredictionErrorData<maskCalcRequired>(inputImage))
                return false;
        }
        const auto& w = randomMatrix.getGray();

        // optimized calculation of the strengthened watermark
        float sumSq = 0.0f;
#pragma omp parallel for reduction(+ : sumSq)
        for (auto i = 0; i < mask.size(); i++) {
            float u = mask(i) * w(i);
            uStrengthened(i) = u;
            sumSq += u * u;
        }
        if (sumSq <= 1e-3f) // for flat images/frames
            return false;

        watermarkStrength = strengthFactor / std::sqrt(sumSq / (baseRows * baseCols));
        uStrengthened *= watermarkStrength;
        return true;
    }

    // compute Prediction error data (coefficients, error sequence), and if needed, prediction error mask
    template <bool maskNeeded> bool computePredictionErrorData(const ArrayXXf& image) {
        meMatrixData.setZero();
        // process CENTER region
        if (hasCenterRegion) {
            const float* imgData = image.data();
            const auto& offsets = meMatrixData.offsets;
#pragma omp parallel
            {
                const int threadId = omp_get_thread_num();
                auto& RxVec = meMatrixData.RxVec_all[threadId].mat;
                auto& rx = meMatrixData.rx_all[threadId].mat;
#pragma omp for
                for (int j = startCol; j < endCol; j++) {
                    const int colOffset = j * baseRows;
                    const float* centerPtr = imgData + colOffset + startRow;
                    const Map<const VectorXf> centerBatch(centerPtr, stripHeight);

                    int k = 0;
                    // rx(u) = sum(center * neighbor_u)
                    // Rx(u, v) = sum(neighbor_u * neighbor_v)
                    for (int u = 0; u < localSize; u++) {
                        const float* neighborPtr = centerPtr + offsets[u];
                        const Map<const VectorXf> neighborBatch(neighborPtr, stripHeight);
                        rx(u) += neighborBatch.dot(centerBatch);
                        for (int v = 0; v <= u; v++, k++) {
                            const float* ptrV = centerPtr + offsets[v];
                            const Map<const VectorXf> mapV(ptrV, stripHeight);
                            RxVec(k) += neighborBatch.dot(mapV);
                        }
                    }
                }
            }
        }
        // process BORDER regions
        auto processBorder = [&](const int startRow, const int endRow, const int startCol, const int endCol) {
            processPredictionErrorBorder(image, startRow, endRow, startCol, endCol,
                                         [&](int i, int j, const LocalVector& neighbors, const int threadId) { meMatrixData.computePredictionErrorMatrices(neighbors, image(i, j), threadId); });
        };
        computeMaskBorders(startRow, endRow, startCol, endCol, hasCenterRegion, processBorder);

        // solve system and coefficients
        if (!meMatrixData.computeCoefficients())
            return false;

        // calculate ex(i,j)
        computeErrorSequence(image, errorSequence);
        if constexpr (maskNeeded) {
            const auto errorSequenceAbs = errorSequence.abs();
            mask = errorSequenceAbs / errorSequenceAbs.maxCoeff();
        }
        return true;
    }

    // computes the prediction error sequence of the input image
    void computeErrorSequence(const ArrayXXf& image, ArrayXXf& outputErrorSequence) {
        const auto& coefficients = meMatrixData.coefficients;
        const auto& offsets = meMatrixData.offsets;
        // process CENTER region
        if (hasCenterRegion) {
            const float* imgData = image.data();
            float* outData = outputErrorSequence.data();
#pragma omp parallel for
            for (int j = startCol; j < endCol; j++) {
                const int colOffset = (j * baseRows) + startRow;
                const Map<const VectorXf> imgBatch(imgData + colOffset, stripHeight);
                Map<VectorXf> errorBatch(outData + colOffset, stripHeight);
                errorBatch = imgBatch; // initialize with image values
                // compute prediction error
                for (int k = 0; k < localSize; k++) {
                    const float* neighborPtr = imgData + colOffset + offsets[k];
                    const Map<const VectorXf> neighborBatch(neighborPtr, stripHeight);
                    errorBatch.noalias() -= neighborBatch * coefficients(k);
                }
            }
        }
        // process BORDER regions
        auto processBorder = [&](const int startRow, const int endRow, const int startCol, const int endCol) {
            processPredictionErrorBorder(image, startRow, endRow, startCol, endCol,
                                         [&](int i, int j, const LocalVector& neighbors, const int) { outputErrorSequence(i, j) = image(i, j) - neighbors.dot(coefficients); });
        };
        computeMaskBorders(startRow, endRow, startCol, endCol, hasCenterRegion, processBorder);
    }
};