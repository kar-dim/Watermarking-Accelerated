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
#include <cstring>
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
    WatermarkEigen<p>(const int rows, const int cols, WatermarkBuffer watermark, const float psnr) : WatermarkBase(rows, cols, std::move(watermark), psnr), predictionData(rows, cols) {
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
#pragma omp parallel for schedule(static)
            for (int col = 0; col < baseCols; col++)
                for (int channel = 0; channel < 3; channel++)
                    std::memcpy(rgbOut[channel].col(col).data(), rgbIn[channel].col(col).data(), static_cast<size_t>(baseRows));
            return;
        }
#pragma omp parallel for schedule(static)
        for (int col = 0; col < baseCols; col++)
            for (int channel = 0; channel < 3; channel++)
                watermarkColumn(rgbIn[channel].col(col).data(), u.col(col).data(), *scale, rgbOut[channel].col(col).data());
    }

    // grayscale embedding: watermark added to luma, row-major output transposed in 16x16 SIMD tiles
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
#pragma omp parallel for schedule(static)
        for (int row = 0; row < baseRows; row += 16) {
            if (row + 16 <= baseRows) {
                int col = 0;
                for (; col + 16 <= baseCols; col += 16) {
                    __m128i blockCols[16];
                    for (int c = 0; c < 16; ++c)
                        blockCols[c] = watermark16Rows(gray.col(col + c).data() + row, scale ? u.col(col + c).data() + row : nullptr, strength);
                    Simd::transposeU8_16x16(blockCols);
                    for (int r = 0; r < 16; ++r)
                        _mm_storeu_si128(reinterpret_cast<__m128i*>(out.data() + (static_cast<size_t>(row + r) * baseCols) + col), blockCols[Simd::kTransposeRegister[r]]);
                }
                for (; col < baseCols; ++col) {
                    const float* gCol = gray.col(col).data() + row;
                    const float* uCol = scale ? u.col(col).data() + row : nullptr;
                    for (int r = 0; r < 16; ++r)
                        out.data()[(static_cast<size_t>(row + r) * baseCols) + col] = toPixelScalar(uCol ? gCol[r] + (uCol[r] * strength) : gCol[r]);
                }
            } else {
                for (int col = 0; col < baseCols; ++col) {
                    const float* gCol = gray.col(col).data() + row;
                    const float* uCol = scale ? u.col(col).data() + row : nullptr;
                    for (int r = 0; r < baseRows - row; ++r)
                        out.data()[(static_cast<size_t>(row + r) * baseCols) + col] = toPixelScalar(uCol ? gCol[r] + (uCol[r] * strength) : gCol[r]);
                }
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
            float* const maskedErrorPtr = columns.col(2 * p).data();
            int next = std::max(first - pad, 0);
            for (int col = first; col < last; col++) {
                for (; next <= std::min(col + pad, baseCols - 1); next++) {
                    float* const errPtr = columns.col(next % p).data();
                    float* const maskOutPtr = columns.col(p + (next % p)).data();
                    const float* const wPtr = w.col(next).data();
                    errorColumn(imageColumn, next, errPtr);
                    int r = 0;
                    for (; r + Simd::lanes <= baseRows; r += Simd::lanes)
                        Simd::store(maskOutPtr + r, Simd::mul(Simd::abs(Simd::load(errPtr + r)), Simd::load(wPtr + r)));
                    for (; r < baseRows; ++r)
                        maskOutPtr[r] = std::abs(errPtr[r]) * wPtr[r];
                }
                errorColumn(maskedColumn, col, maskedErrorPtr);
                const float* const errPtr = columns.col(col % p).data();
                Simd::Vec dotAcc = Simd::zero();
                Simd::Vec sqEAcc = Simd::zero();
                Simd::Vec sqMAcc = Simd::zero();
                int r = 0;
                for (; r + Simd::lanes <= baseRows; r += Simd::lanes) {
                    const Simd::Vec ev = Simd::load(errPtr + r);
                    const Simd::Vec mv = Simd::load(maskedErrorPtr + r);
                    dotAcc = Simd::fmadd(ev, mv, dotAcc);
                    sqEAcc = Simd::fmadd(ev, ev, sqEAcc);
                    sqMAcc = Simd::fmadd(mv, mv, sqMAcc);
                }
                double colDot = Simd::sum(dotAcc);
                double colSqE = Simd::sum(sqEAcc);
                double colSqM = Simd::sum(sqMAcc);
                for (; r < baseRows; ++r) {
                    const double ev = errPtr[r];
                    const double mv = maskedErrorPtr[r];
                    colDot += ev * mv;
                    colSqE += ev * ev;
                    colSqM += mv * mv;
                }
                dot += colDot;
                sqError += colSqE;
                sqMaskedError += colSqM;
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

    static uint8_t toPixelScalar(const float value) { return static_cast<uint8_t>(std::min(std::max(value + 0.5f, 0.0f), 255.0f)); }

    // truncation matches scalar float->uint8 cast after [0, 255] clamp
    static __m128i watermark16Rows(const float* __restrict grayCol, const float* __restrict uCol, const float strength) {
        const __m256 half = _mm256_set1_ps(0.5f);
        const __m256 zero = _mm256_setzero_ps();
        const __m256 limit = _mm256_set1_ps(255.0f);
        const __m256 vStrength = _mm256_set1_ps(strength);
        const auto clamp8 = [&](const int offset) {
            const __m256 base = _mm256_loadu_ps(grayCol + offset);
            const __m256 val = uCol ? _mm256_add_ps(base, _mm256_mul_ps(_mm256_loadu_ps(uCol + offset), vStrength)) : base;
            return _mm256_cvttps_epi32(_mm256_min_ps(_mm256_max_ps(_mm256_add_ps(val, half), zero), limit));
        };
        const __m256i u16 = _mm256_permute4x64_epi64(_mm256_packus_epi32(clamp8(0), clamp8(8)), 0b11'01'10'00);
        return _mm_packus_epi16(_mm256_castsi256_si128(u16), _mm256_extracti128_si256(u16, 1));
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

    // unscaled watermark u = |e| * w and scale that reaches PSNR, empty for flat images
    std::optional<float> computeStrengthenedWatermark(const ArrayXXf& inputImage) {
        if (!predictionData.computeCoefficients(inputImage.data()))
            return std::nullopt;
        if (u.size() == 0)
            u.resize(baseRows, baseCols);
        prepareWorkspaces();
        const auto& w = randomMatrix.getGray();
        const auto imageColumn = [&](const int col) { return inputImage.data() + (static_cast<size_t>(std::clamp(col, 0, baseCols - 1)) * baseRows); };
        float maxAbs = 0.0f;
        double sumSq = 0.0;
#pragma omp parallel reduction(max : maxAbs) reduction(+ : sumSq)
        {
            // error column stays in cache between passes, full size error never stored
            float* const errPtr = workspaces[omp_get_thread_num()].col(0).data();
#pragma omp for schedule(static)
            for (int col = 0; col < baseCols; col++) {
                errorColumn(imageColumn, col, errPtr);
                const float* const wPtr = w.col(col).data();
                float* const uPtr = u.col(col).data();
                Simd::Vec maxVec = Simd::zero();
                Simd::Vec sqVec = Simd::zero();
                int r = 0;
                for (; r + Simd::lanes <= baseRows; r += Simd::lanes) {
                    const Simd::Vec eAbs = Simd::abs(Simd::load(errPtr + r));
                    maxVec = Simd::max(maxVec, eAbs);
                    const Simd::Vec uv = Simd::mul(eAbs, Simd::load(wPtr + r));
                    Simd::store(uPtr + r, uv);
                    sqVec = Simd::fmadd(uv, uv, sqVec);
                }
                float colMax = Simd::reduceMax(maxVec);
                double colSq = Simd::sum(sqVec);
                for (; r < baseRows; ++r) {
                    const float eAbs = std::abs(errPtr[r]);
                    colMax = std::max(colMax, eAbs);
                    const float uv = eAbs * wPtr[r];
                    uPtr[r] = uv;
                    colSq += static_cast<double>(uv) * uv;
                }
                maxAbs = std::max(maxAbs, colMax);
                sumSq += colSq;
            }
        }
        // flat image: sum(u^2) with u normalized by max|e|
        if (!(maxAbs > 0.0f) || sumSq <= 1e-3 * maxAbs * maxAbs)
            return std::nullopt;
        return static_cast<float>(strengthFactor / std::sqrt(sumSq / totalPixels));
    }
};
