#pragma once

#include "buffer.hpp"
#include "Eigen/Core"
#include "PredictionErrorMatrixData.hpp"
#include "simd.hpp"
#include "WatermarkBase.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <omp.h>
#include <optional>
#include <utility>
#include <vector>

/*!
 *  \brief  Watermark embedding and detection, Eigen CPU implementation
 *  \author Dimitris Karatzas
 */
template <int p>
class WatermarkEigen final : public WatermarkBase {
  private:
    static constexpr int pSquared = p * p;
    static constexpr int pad = p / 2;
    static constexpr int centerIndex = pSquared / 2;
    // per thread columns: p of the prediction error, p of the masked watermark and one result column
    static constexpr int workspaceCols = (2 * p) + 1;
    // neighbor columns per pass over an error column, their coefficients STAY in registers
    static constexpr int passColumns = p == 3 ? 3 : 1;
    // sums per vector (chain of FMAs)
    static constexpr int chains = 2;

    const int endRow = baseRows - pad;

    using ArrayXXf = Eigen::ArrayXXf;

  public:
    WatermarkEigen<p>(const int rows, const int cols, WatermarkBuffer watermark, const float psnr) : WatermarkBase(rows, cols, std::move(watermark), psnr), u(rows, cols), predictionData(rows, cols) {
        prepareWorkspaces();
    }

    // RGB embedding: compute watermark from luma, flat images copied unchanged
    void makeWatermark(const ImageBuffer& inputGrayImage, const ImageOutputBuffer& inputImage, ImageOutputBuffer& output) override {
        const auto scale = computeStrengthenedWatermark(inputGrayImage.getGray());
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

    // grayscale embedding: watermark added to luma, row-major output transposed in small tiles
    void makeWatermark(const ImageBuffer& inputGrayImage, ImageOutputBuffer& output, const Layout outputLayout) override {
        const auto& gray = inputGrayImage.getGray();
        const auto scale = computeStrengthenedWatermark(gray);
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

    // detection: correlation between image prediction error and watermark error
    float detectWatermark(const ImageBuffer& inputImage) override {
        const auto& image = inputImage.getGray();
        if (!predictionData.computeCoefficients(image.data()))
            return 0.0f;
        prepareWorkspaces();
        const auto& w = randomMatrix.getGray();
        const auto imageColumn = [&](const int col) { return image.data() + (static_cast<size_t>(std::clamp(col, 0, baseCols - 1)) * baseRows); };
        double dot = 0.0;
        double sqError = 0.0;
        double sqMaskedError = 0.0;
#pragma omp parallel reduction(+ : dot, sqError, sqMaskedError)
        {
            // each thread keeps only the last p columns of e and of |e| * w (the correlation is scale-invariant, no max|e| division)
            ArrayXXf& columns = workspaces[omp_get_thread_num()];
            const int threads = omp_get_num_threads();
            const int first = static_cast<int>((static_cast<int64_t>(baseCols) * omp_get_thread_num()) / threads);
            const int last = static_cast<int>((static_cast<int64_t>(baseCols) * (omp_get_thread_num() + 1)) / threads);
            const auto maskedColumn = [&](const int col) { return columns.col(p + (std::clamp(col, 0, baseCols - 1) % p)).data(); };
            auto maskedError = columns.col(2 * p);
            int next = std::max(first - pad, 0);
            for (int col = first; col < last; col++) {
                for (; next <= std::min(col + pad, baseCols - 1); next++) {
                    auto error = columns.col(next % p);
                    errorColumn(imageColumn, next, error.data());
                    columns.col(p + (next % p)) = error.abs() * w.col(next);
                }
                errorColumn(maskedColumn, col, maskedError.data());
                const auto error = columns.col(col % p);
                dot += (error * maskedError).sum();
                sqError += error.square().sum();
                sqMaskedError += maskedError.square().sum();
            }
        }
        const float correlation = static_cast<float>(dot / (std::sqrt(sqError) * std::sqrt(sqMaskedError)));
        return std::isfinite(correlation) ? correlation : 0.0f;
    }

  private:
    using Gray8 = Eigen::Array<uint8_t, Eigen::Dynamic, Eigen::Dynamic>;

    // unscaled watermark |e| * w
    ArrayXXf u;
    PredictionErrorMatrixData<p> predictionData;
    std::vector<ArrayXXf> workspaces;

    // the thread count can grow after construction
    void prepareWorkspaces() {
        while (workspaces.size() < static_cast<size_t>(omp_get_max_threads()))
            workspaces.emplace_back(baseRows, workspaceCols);
    }

    // watermarked output pixels: round and clamp to [0, 255]
    template <typename Expr>
    static auto toPixels(const Expr& values) {
        return (values + 0.5f).cwiseMax(0.0f).cwiseMin(255.0f).template cast<uint8_t>();
    }

    // one watermarked column: output = input + strength * u
    // __restrict and local row count prevent store aliasing for auto-vectorization
    template <typename T>
    void watermarkColumn(const T* __restrict input, const float* __restrict uColumn, const float strength, uint8_t* __restrict output) const {
        const int rows = baseRows;
        for (int row = 0; row < rows; row++) {
            const float us = uColumn[row] * strength;
            const float value = static_cast<float>(input[row]) + us;
            output[row] = static_cast<uint8_t>(std::min(std::max(value + 0.5f, 0.0f), 255.0f));
        }
    }

    // prediction error of one column: e = x - sum(c_k * x_k), fused multiply-subtract per neighbor
    // columnAt returns a column with its index clamped at the image edges
    template <typename ColumnAt>
    void errorColumn(const ColumnAt& columnAt, const int col, float* __restrict error) const {
        using namespace Simd;
        const auto& coefficients = predictionData.coefficients;
        std::array<const float*, p> columns;
        for (int dj = 0; dj < p; dj++)
            columns[dj] = columnAt(col + dj - pad);
        const auto isNeighbor = [](const int dj, const int di) { return dj != pad || di != pad; };
        const auto coefficient = [&](const int dj, const int di) { return coefficients(((dj * p) + di) - (((dj * p) + di) > centerIndex)); };
        // one pass of a row: the neighbors of "passColumns" columns leave "sums", neighbor rows come from rowAt
        // row offset di goes to sum di % chains in the vector and in the scalar code
        const auto scalarPass = [&](const int first, const float start, const auto& rowAt) {
            std::array<float, chains> sums{start};
            for (int offset = 0; offset < passColumns; offset++)
                for (int di = 0; di < p; di++)
                    if (isNeighbor(first + offset, di))
                        sums[di % chains] = std::fma(-coefficient(first + offset, di), columns[first + offset][rowAt(di)], sums[di % chains]);
            for (int chain = 1; chain < chains; chain++)
                sums[0] += sums[chain];
            return sums[0];
        };

        constexpr int quads = windowQuads(p);
        // the last load of a neighbor column reads this many rows past its last window
        const int vectorEnd = endRow - ((4 * (quads - 1)) - (p - 1));
        for (int first = 0; first < p; first += passColumns) {
            // locals: the stores to "error" could alias the member coefficients
            Vec c[passColumns * p];
            for (int offset = 0; offset < passColumns; offset++)
                for (int di = 0; di < p; di++)
                    c[(offset * p) + di] = isNeighbor(first + offset, di) ? broadcast(coefficient(first + offset, di)) : zero();
            const float* source = first == 0 ? columns[pad] : error;
            int row = pad;
            for (; row + lanes <= vectorEnd; row += lanes) {
                Vec sums[chains];
                sums[0] = load(source + row);
                for (int chain = 1; chain < chains; chain++)
                    sums[chain] = zero();
                // constant offset and di: the window shifts resolve at compile time and the sums stay in registers
#pragma clang loop unroll(full)
                for (int offset = 0; offset < passColumns; offset++) {
                    Vec neighbors[quads];
                    for (int q = 0; q < quads; q++)
                        neighbors[q] = load(columns[first + offset] + row - pad + (4 * q));
#pragma clang loop unroll(full)
                    for (int di = 0; di < p; di++)
                        if (isNeighbor(first + offset, di))
                            sums[di % chains] = fnmadd(c[(offset * p) + di], window(neighbors, di), sums[di % chains]);
                }
                for (int chain = 1; chain < chains; chain++)
                    sums[0] = add(sums[0], sums[chain]);
                store(error + row, sums[0]);
            }
#if defined(__AVX512F__)
            // Neighbor loads extend beyond the output rows, even in the narrow cleanup
            if (row + Avx2::lanes <= vectorEnd) {
                Avx2::Vec sums[chains];
                sums[0] = Avx2::load(source + row);
                for (int chain = 1; chain < chains; chain++)
                    sums[chain] = Avx2::zero();
#pragma clang loop unroll(full)
                for (int offset = 0; offset < passColumns; offset++) {
                    Avx2::Vec neighbors[quads];
                    for (int q = 0; q < quads; q++)
                        neighbors[q] = Avx2::load(columns[first + offset] + row - pad + (4 * q));
#pragma clang loop unroll(full)
                    for (int di = 0; di < p; di++)
                        if (isNeighbor(first + offset, di))
                            sums[di % chains] = Avx2::fnmadd(low(c[(offset * p) + di]), window(neighbors, di), sums[di % chains]);
                }
                for (int chain = 1; chain < chains; chain++)
                    sums[0] = Avx2::add(sums[0], sums[chain]);
                Avx2::store(error + row, sums[0]);
                row += Avx2::lanes;
            }
#endif
            for (; row < endRow; row++)
                error[row] = scalarPass(first, source[row], [&](const int di) { return row + di - pad; });
        }
        // the top and bottom rows read clamped neighbors
        const auto clampedRow = [&](const int row) {
            for (int first = 0; first < p; first += passColumns)
                error[row] = scalarPass(first, first == 0 ? columns[pad][row] : error[row], [&](const int di) { return std::clamp(row + di - pad, 0, baseRows - 1); });
        };
        for (int row = 0; row < pad; row++) {
            clampedRow(row);
            clampedRow(endRow + row);
        }
    }

    // unscaled watermark u = |e| * w and the scale that reaches the PSNR, empty for flat images
    std::optional<float> computeStrengthenedWatermark(const ArrayXXf& inputImage) {
        if (!predictionData.computeCoefficients(inputImage.data()))
            return std::nullopt;
        prepareWorkspaces();
        const auto& w = randomMatrix.getGray();
        const auto imageColumn = [&](const int col) { return inputImage.data() + (static_cast<size_t>(std::clamp(col, 0, baseCols - 1)) * baseRows); };
        float maxAbs = 0.0f;
        double sumSq = 0.0;
#pragma omp parallel reduction(max : maxAbs) reduction(+ : sumSq)
        {
            // the error column stays in cache between its passes, the full size error is never stored
            auto error = workspaces[omp_get_thread_num()].col(0);
#pragma omp for schedule(static)
            for (int col = 0; col < baseCols; col++) {
                errorColumn(imageColumn, col, error.data());
                maxAbs = std::max(maxAbs, error.abs().maxCoeff());
                u.col(col) = error.abs() * w.col(col);
                sumSq += u.col(col).square().sum();
            }
        }
        // flat image: sum(u^2) with u normalized by max|e|
        if (!(maxAbs > 0.0f) || sumSq <= 1e-3 * maxAbs * maxAbs)
            return std::nullopt;
        return static_cast<float>(strengthFactor / std::sqrt(sumSq / totalPixels));
    }
};
