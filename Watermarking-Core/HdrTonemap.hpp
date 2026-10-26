#pragma once
#include <algorithm>

/*!
 *  \brief  HDR to SDR tonemapping parameters shared by video pipeline and GPU kernels
 *  \author Dimitris Karatzas
 */
namespace video_utils {
// Mobius tonemap coeffs K * (x + a) / (x + b) depend only on HDR peak luminance
// video constant computed once and reused across frames
struct MobiusParams {
    float a, b, k;

    // build coefficients from HDR peak in 100 nit units
    static MobiusParams fromHdrPeak(const float hdrPeak) {
        constexpr float j = 0.3f;
        const float a = -j * j * (hdrPeak - 1.0f) / (j * j - 2.0f * j + hdrPeak);
        const float b = (j * j - 2.0f * j * hdrPeak + hdrPeak) / std::max(hdrPeak - 1.0f, 1e-6f);
        const float k = (b * b + 2.0f * b * j + j * j) / (b - a);
        return {a, b, k};
    }
};

// position of first 4:2:0 chroma sample in luma pixels
// (0, 0.5) left (MPEG-2, unspecified), (0, 0) top-left (HDR10), (0.5, 0.5) center
struct ChromaSiting {
    float x, y;
};
} // namespace video_utils
