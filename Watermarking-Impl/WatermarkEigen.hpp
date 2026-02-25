#pragma once

#include "buffer.hpp"
#include "Eigen/Core"
#include "include/WatermarkTypes.hpp"
#include "PredictionErrorMatrixData.hpp"
#include "WatermarkBase.hpp"
#include <cmath>
#include <fstream>
#include <omp.h>
#include <string>

/*!
 *  \brief  Functions for watermark computation and detection, Eigen implementation.
 *  \author Dimitris Karatzas
 */
template <int p>
class WatermarkEigen final : public WatermarkBase {
  private:
    enum class Op { ADD, SUB };

    static constexpr bool maskCalcRequired = true;
    static constexpr bool maskCalcNotRequired = false;
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
    WatermarkEigen<p>(const unsigned int rows, const unsigned int cols, const std::string& randomMatrixPath, const float psnr)
        : WatermarkBase(rows, cols, randomMatrixPath, psnr, initializeRandomMatrix), mask(rows, cols), errorSequence(rows, cols), filteredEstimation(rows, cols), u(rows, cols),
          uStrengthened(rows, cols), meMatrixData(omp_get_max_threads(), rows) {}

    // main watermark embedding method
    void makeWatermark(const ImageBuffer& inputGrayImage, const ImageBuffer& inputImage, ImageOutputBuffer& output, const MaskMethod maskType) override {
        // compute the strengthened watermark, if it fails assign input to output and return
        if (!computeStrengthenedWatermark(inputGrayImage.getGray(), maskType)) {
            inputImage.assignTo(output);
            return;
        }
        // embed the watermark into the input image
        inputImage.applyWatermark(uStrengthened, output);
    }

    // main watermark detection method
    float detectWatermark(const ImageBuffer& inputImage, MaskMethod maskType) override {
        const auto& watermarkedBuffer = inputImage.getGray();
        if (maskType == MaskMethod::NVF) {
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
                Map<const VectorXf> ezVec(ezPtr + start, actualSize);
                Map<const VectorXf> euVec(euPtr + start, actualSize);
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

    // initialize the watermark random matrix into an Eigen buffer
    static ImageBuffer initializeRandomMatrix(std::ifstream& stream, const size_t totalBytes, const unsigned int rows, const unsigned int cols) {
        Eigen::ArrayXXf watermark(cols, rows);
        stream.read(reinterpret_cast<char*>(watermark.data()), totalBytes);
        return ImageBuffer(watermark.transpose());
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

    // helper method for custom mask calculation per pixel
    inline void computeCustomMaskPixel(const double sum, const double sumSq, const int i, const int j) {
        const double mean = sum / pSquared;
        const double variance = (sumSq / pSquared) - (mean * mean);
        const double maskValue = variance / (1.0 + variance);
        mask(i, j) = std::clamp(static_cast<float>(maskValue), 0.0f, 1.0f);
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
#pragma omp parallel
        {
            auto processRect = [&](int rStart, int rEnd, int cStart, int cEnd) {
#pragma omp for collapse(2) nowait
                for (int j = cStart; j < cEnd; j++) {
                    for (int i = rStart; i < rEnd; i++) {
                        double sum = 0.0, sumSq = 0.0;
                        // for borders, we cannot slide, so we do the full O(p^2) sum
                        for (int jj = -pad; jj <= pad; jj++) {
                            for (int ii = -pad; ii <= pad; ii++) {
                                const float val = clampedValue(image, i + ii, j + jj, baseRows, baseCols);
                                computeCustomMaskSums<Op::ADD>(val, sum, sumSq);
                            }
                        }
                        computeCustomMaskPixel(sum, sumSq, i, j);
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
    }

    // helper method to process the border pixels (clamp if out of bounds)
    // by using a supplied function
    template <typename Processor>
    void processBorder(const ArrayXXf& image, Processor&& processor) {
#pragma omp parallel
        {
            const int threadId = omp_get_thread_num();
            LocalVector neighbors; // per thread

            auto processRect = [&](int rStart, int rEnd, int cStart, int cEnd) {
            // nowait: threads jump between borders immediately
            // collapse(2): handles thin strips efficiently
#pragma omp for collapse(2) nowait
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
    }

    // compute the strengthened watermark, calculated by multiplying the mask with the strengthened watermark (random matrix)
    bool computeStrengthenedWatermark(const ArrayXXf& inputImage, MaskMethod maskType) {
        if (maskType == MaskMethod::NVF)
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

        const float watermarkStrength = strengthFactor / std::sqrt(sumSq / (baseRows * baseCols));
        uStrengthened *= watermarkStrength;
        return true;
    }

    // compute Prediction error data (coefficients, error sequence), and if needed, prediction error mask
    template <bool maskNeeded>
    bool computePredictionErrorData(const ArrayXXf& image) {
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
        // border regions parallel handling
        processBorder(image, [&](const int i, const int j, const LocalVector& neighbors, const int threadId) { meMatrixData.computePredictionErrorMatrices(neighbors, image(i, j), threadId); });

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
            // calculate prediction error for center region using Eigen maps and OpenMP
            // optimized to calculate 8 neighbors at a time to fully utilize vectorization
            // and eigen lazy evaluation with big expression trees
#pragma omp parallel for
            for (int j = startCol; j < endCol; j++) {
                const int colOffset = (j * baseRows) + startRow;
                const Map<const VectorXf> imgBatch(imgData + colOffset, stripHeight);
                Map<VectorXf> errorBatch(outData + colOffset, stripHeight);
                // first block: initialization and calculation of 8 neighbors
                // E = I - (c0*N0 + c1*N1... + c7*N7)
                errorBatch.noalias() =
                    imgBatch -
                    (Map<const VectorXf>(imgData + colOffset + offsets[0], stripHeight) * coefficients(0) + Map<const VectorXf>(imgData + colOffset + offsets[1], stripHeight) * coefficients(1) +
                     Map<const VectorXf>(imgData + colOffset + offsets[2], stripHeight) * coefficients(2) + Map<const VectorXf>(imgData + colOffset + offsets[3], stripHeight) * coefficients(3) +
                     Map<const VectorXf>(imgData + colOffset + offsets[4], stripHeight) * coefficients(4) + Map<const VectorXf>(imgData + colOffset + offsets[5], stripHeight) * coefficients(5) +
                     Map<const VectorXf>(imgData + colOffset + offsets[6], stripHeight) * coefficients(6) + Map<const VectorXf>(imgData + colOffset + offsets[7], stripHeight) * coefficients(7));
                // calculate remaining blocks (indices 8 to localSize)
                // for p=3 this won't even run (compiler will optimize it out entirely)
                // E = E - (c8*N8 + ...)
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
        }
        // process BORDER regions
        processBorder(image, [&](const int i, const int j, const LocalVector& neighbors, const int) { outputErrorSequence(i, j) = image(i, j) - neighbors.dot(coefficients); });
    }
};