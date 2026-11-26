#pragma once

/*!
 *  \brief  Shared ME autocorrelation shift layout constants and index helpers
 *  \author Dimitris Karatzas
 */

#if defined(__CUDACC__)
#define WM_ME_HD __host__ __device__
#else
#define WM_ME_HD
#endif

template <int p>
struct MeShiftLayout {
    static constexpr int pad = p / 2;
    static constexpr int windowSize = p * p;
    static constexpr int windowCenter = windowSize / 2;
    static constexpr int localSize = windowSize - 1;
    static constexpr int maxShift = p - 1;
    static constexpr int shiftSpan = (2 * maxShift) + 1;
    // Unique shift directions where dc > 0 or dc == 0 with dr >= 0
    static constexpr int numShifts = ((shiftSpan * shiftSpan) + 1) / 2;
    static constexpr int borderSize = 4 * pad;
    static constexpr int borderRowsOffset = numShifts;
    static constexpr int borderColsOffset = numShifts + (numShifts * borderSize);
    static constexpr int shiftSumsSize = numShifts * (1 + (2 * borderSize));
    // Scale products by 1 / 255^2 to keep fixed point sums in uint64 range
    static constexpr float productScale = 1.0f / (255.0f * 255.0f);
    static constexpr int solverSystemSize = ((localSize * (localSize + 1)) / 2) + localSize;

    WM_ME_HD static constexpr int shiftIndex(const int dr, const int dc) { return dc == 0 ? dr : (maxShift + 1) + ((dc - 1) * shiftSpan) + (dr + maxShift); }
    WM_ME_HD static constexpr int firstShift(const int dc) { return dc == 0 ? 0 : shiftIndex(-maxShift, dc); }
    static constexpr int shiftGroups = p >= 9 ? 3 : (p >= 7 ? 2 : 1);
    static constexpr int interiorCols = p >= 7 ? 2 : 1;
    static constexpr int interiorRun = 8;
    WM_ME_HD static constexpr int groupFirstDc(const int group) { return (group * (maxShift + 1)) / shiftGroups; }
    WM_ME_HD static constexpr int shiftEnd(const int dc) { return dc > maxShift ? numShifts : firstShift(dc); }
    static constexpr bool copyBorderRows = p >= 7;
    static constexpr int borderCopyRows = 6 * pad;
#if defined(__CUDACC__)
    __device__ static int borderToCoord(const int border, const int size) { return border < 2 * pad ? border - pad : size - (3 * pad) + border; }
    __device__ static int borderCopyRow(const int row, const int height) { return row < borderCopyRows / 2 ? row : row - height + borderCopyRows; }
#endif
};

constexpr int meShiftGroups(const int p) { return p >= 9 ? 3 : (p >= 7 ? 2 : 1); }
constexpr int meFirstShift(const int p, const int dc) { return dc == 0 ? 0 : p + ((dc - 1) * ((2 * p) - 1)); }
