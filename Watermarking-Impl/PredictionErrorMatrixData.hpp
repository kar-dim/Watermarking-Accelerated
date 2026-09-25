#pragma once
#include "Eigen/Cholesky"
#include "Eigen/Core"
#include <algorithm>
#include <array>
#include <omp.h>
#include <vector>

/*!
 *  \brief  Builds and solves the ME prediction system (Rx * coefficients = rx) using Eigen.
 *
 *  Computes autocorrelation sums by spatial shift rather than looping over all window pairs per pixel.
 *  Sums unique shifts over the inner image, then adds border rows, border columns, and corners.
 *  \author Dimitris Karatzas
 */
template <int p>
class PredictionErrorMatrixData {
  private:
    static constexpr int localSize = (p * p) - 1;
    static constexpr int windowSize = p * p;
    static constexpr int pad = p / 2;
    static constexpr int windowCenter = windowSize / 2;
    // Keep only unique shift directions (dc > 0 or dc == 0 with dr >= 0)
    static constexpr int maxShift = p - 1;
    static constexpr int shiftSpan = (2 * maxShift) + 1;
    static constexpr int numShifts = ((shiftSpan * shiftSpan) + 1) / 2;
    // Border rows and columns outside the inner image area
    static constexpr int borderSize = 4 * pad;

    using LocalVector = Eigen::Matrix<float, localSize, 1>;
    // Double precision prevents rounding errors on large images
    using AccumVector = Eigen::Matrix<double, localSize, 1>;
    using AccumMatrix = Eigen::Matrix<double, localSize, localSize>;
    using WindowMatrix = Eigen::Matrix<double, windowSize, windowSize>;

    // Per-thread shift sums for inner rows and border rows
    struct alignas(64) ShiftSums {
        std::array<double, numShifts> inner;
        std::array<double, numShifts * borderSize> borderRows;
    };

    const int rows, cols;
    std::vector<ShiftSums> threadSums;
    // Border columns are processed once, so threads write them directly
    std::vector<double> borderCols; // Inner row sums per shift and border column
    std::vector<double> corners;    // Corner sums per shift, border row, and border column
    WindowMatrix windowSums;

    static constexpr int shiftIndex(const int dr, const int dc) { return dc == 0 ? dr : (maxShift + 1) + ((dc - 1) * shiftSpan) + (dr + maxShift); }

    // Vector register width and maximum shifts processed together in one pass
#if defined(__AVX512F__)
    static constexpr int rowLanes = 16;
    static constexpr int maxShiftsPerPass = 24;
#else
    static constexpr int rowLanes = 8;
    static constexpr int maxShiftsPerPass = 12;
#endif

    // Computes dot products for NUM consecutive row shifts using vector instructions
    // Inner rows run vectorized, while boundary rows read clamped neighbor values
    template <int NUM>
    void innerDots(const float* __restrict colA, const float* __restrict colB, const int dr0, float* __restrict dots) const {
        constexpr int lanes = rowLanes;
        using Lanes = Eigen::Array<float, lanes, 1>;
        const int rStart = pad;
        const int rEnd = rows - pad;
        const int mainLo = std::clamp(-dr0, rStart, rEnd);
        const int mainHi = std::clamp(rows - (dr0 + NUM - 1), mainLo, rEnd);
        std::array<Lanes, NUM> acc;
        for (auto& lane : acc)
            lane.setZero();
        int r = mainLo;
        for (; r + lanes <= mainHi; r += lanes) {
            const Lanes a = Lanes::Map(colA + r);
            for (int k = 0; k < NUM; k++)
                acc[k] += a * Lanes::Map(colB + r + dr0 + k);
        }
        for (int k = 0; k < NUM; k++)
            dots[k] = acc[k].sum();
        const auto edgeRows = [&](const int lo, const int hi) {
            constexpr int block = 16;
            std::array<float, block + NUM - 1> partners;
            for (int blockLo = lo; blockLo < hi; blockLo += block) {
                const int count = std::min(block, hi - blockLo);
                for (int i = 0; i < count + NUM - 1; i++)
                    partners[i] = colB[std::clamp(blockLo + dr0 + i, 0, rows - 1)];
                for (int i = 0; i < count; i++)
                    for (int k = 0; k < NUM; k++)
                        dots[k] += colA[blockLo + i] * partners[i + k];
            }
        };
        edgeRows(rStart, mainLo);
        // Remainder rows and bottom edge rows
        edgeRows(r, rEnd);
    }

    // Splits COUNT row shifts into balanced passes that fit in vector registers
    template <int COUNT>
    void innerDotsRun(const float* colA, const float* colB, const int dr0, float* dots) const {
        constexpr int passes = (COUNT + maxShiftsPerPass - 1) / maxShiftsPerPass;
        constexpr int first = (COUNT + passes - 1) / passes;
        innerDots<first>(colA, colB, dr0, dots);
        if constexpr (COUNT > first)
            innerDotsRun<COUNT - first>(colA, colB, dr0 + first, dots + first);
    }

    // Maps a border row index to the unclamped image row coordinate
    int borderRowToRow(const int border) const { return border < 2 * pad ? border - pad : rows - (3 * pad) + border; }

    // Gathers clamped neighbor values needed by border row calculations
    static constexpr int borderPartnerCount = (2 * pad) + (2 * maxShift);
    static constexpr int borderPartnerIndex(const int border) { return (border < 2 * pad ? border : border + (2 * maxShift)) + maxShift; }
    std::array<float, 2 * borderPartnerCount> borderPartners(const float* colB) const {
        std::array<float, 2 * borderPartnerCount> partners;
        for (int i = 0; i < borderPartnerCount; i++) {
            partners[i] = colB[std::clamp(i - pad - maxShift, 0, rows - 1)];
            partners[borderPartnerCount + i] = colB[std::clamp(rows - pad - maxShift + i, 0, rows - 1)];
        }
        return partners;
    }

    // Computes all shift sums for a single column
    void sumColumn(const float* image, const int c, ShiftSums& sums) {
        const bool innerColumn = c >= pad && c < cols - pad;
        const int borderCol = c < pad ? c + pad : (2 * pad) + c - (cols - pad);
        const float* colA = image + (static_cast<size_t>(std::clamp(c, 0, cols - 1)) * rows);
        std::array<float, borderSize> borderA;
        for (int border = 0; border < borderSize; border++)
            borderA[border] = colA[std::clamp(borderRowToRow(border), 0, rows - 1)];

        for (int dc = 0; dc <= maxShift; dc++) {
            const float* colB = image + (static_cast<size_t>(std::clamp(c + dc, 0, cols - 1)) * rows);
            const int drFirst = dc == 0 ? 0 : -maxShift;
            std::array<float, shiftSpan> dots;
            if (dc == 0)
                innerDotsRun<maxShift + 1>(colA, colB, 0, dots.data());
            else
                innerDotsRun<shiftSpan>(colA, colB, -maxShift, dots.data());
            const auto partners = borderPartners(colB);
            for (int dr = drFirst; dr <= maxShift; dr++) {
                const int shift = shiftIndex(dr, dc);
                const float dot = dots[dr - drFirst];
                if (innerColumn) {
                    sums.inner[shift] += dot;
                    for (int border = 0; border < borderSize; border++)
                        sums.borderRows[(shift * borderSize) + border] += borderA[border] * partners[borderPartnerIndex(border) + dr];
                } else {
                    borderCols[(shift * borderSize) + borderCol] = dot;
                    for (int border = 0; border < borderSize; border++)
                        corners[(((shift * borderSize) + border) * borderSize) + borderCol] = borderA[border] * partners[borderPartnerIndex(border) + dr];
                }
            }
        }
    }

    // Combines inner sum, border rows, border columns, and corners for a window position
    double windowSum(const int shift, const int baseRow, const int baseCol, const ShiftSums& total) const {
        // Border indices covered by this window are contiguous
        const int rowLo = baseRow + pad, rowHi = baseRow + (3 * pad);
        const int colLo = baseCol + pad, colHi = baseCol + (3 * pad);
        double value = total.inner[shift];
        for (int border = rowLo; border < rowHi; border++) {
            value += total.borderRows[(shift * borderSize) + border];
            for (int borderC = colLo; borderC < colHi; borderC++)
                value += corners[(((shift * borderSize) + border) * borderSize) + borderC];
        }
        for (int borderC = colLo; borderC < colHi; borderC++)
            value += borderCols[(shift * borderSize) + borderC];
        return value;
    }

  public:
    LocalVector coefficients;
    AccumVector rx;
    AccumMatrix Rx;
    // Column-major memory offsets for neighboring pixels around the center
    std::vector<int> neighborOffsets;

  public:
    // Allocates buffers for the given image dimensions
    PredictionErrorMatrixData(const int rows, const int cols)
        : rows(rows), cols(cols), threadSums(omp_get_max_threads()), borderCols(numShifts * borderSize), corners(numShifts * borderSize * borderSize) {
        neighborOffsets.reserve(localSize);
        for (int dj = 0; dj < p; dj++) {
            for (int di = 0; di < p; di++) {
                if (di == pad && dj == pad)
                    continue;
                neighborOffsets.push_back((dj - pad) * rows + (di - pad));
            }
        }
    }

    // Computes Rx and rx from shift sums and solves the system with Cholesky
    bool computeCoefficients(const float* image) {
        // Resize thread buffers if thread count changed, then reset sums to zero
        threadSums.resize(std::max<size_t>(threadSums.size(), omp_get_max_threads()));
        for (auto& sums : threadSums) {
            sums.inner.fill(0.0);
            sums.borderRows.fill(0.0);
        }
#pragma omp parallel
        {
            auto& sums = threadSums[omp_get_thread_num()];
#pragma omp for schedule(static)
            for (int c = -pad; c < cols + pad; c++)
                sumColumn(image, c, sums);
        }
        ShiftSums total{};
        for (const auto& sums : threadSums) {
            for (int i = 0; i < numShifts; i++)
                total.inner[i] += sums.inner[i];
            for (int i = 0; i < numShifts * borderSize; i++)
                total.borderRows[i] += sums.borderRows[i];
        }
        // Fill the symmetric matrix of window pair sums
        for (int wa = 0; wa < windowSize; wa++) {
            for (int wb = wa; wb < windowSize; wb++) {
                int baseRow = (wa % p) - pad, baseCol = (wa / p) - pad;
                int dr = (wb % p) - (wa % p), dc = (wb / p) - (wa / p);
                // Swap direction when needed to match the stored positive half-shift
                if (dc < 0 || (dc == 0 && dr < 0)) {
                    baseRow += dr;
                    baseCol += dc;
                    dr = -dr;
                    dc = -dc;
                }
                windowSums(wa, wb) = windowSums(wb, wa) = windowSum(shiftIndex(dr, dc), baseRow, baseCol, total);
            }
        }
        // Build Rx for neighbor pairs and rx for neighbor-to-center pairs
        for (int k = 0; k < localSize; k++) {
            const int wk = k + (k >= windowCenter);
            rx(k) = windowSums(wk, windowCenter);
            for (int l = 0; l < localSize; l++)
                Rx(k, l) = windowSums(wk, l + (l >= windowCenter));
        }
        if (!Rx.allFinite() || !rx.allFinite())
            return false;
        // Solve Rx * coefficients = rx using upper Cholesky decomposition
        Eigen::LLT<AccumMatrix, Eigen::Upper> llt(Rx);
        if (llt.info() != Eigen::Success)
            return false;
        coefficients = llt.solve(rx).template cast<float>();
        return true;
    }
};
