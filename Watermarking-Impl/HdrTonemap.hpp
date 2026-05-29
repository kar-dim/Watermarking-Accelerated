#pragma once
#include <algorithm>

/*!
 *  \brief  HDR -> SDR tonemapping parameters shared by the video pipeline and the CUDA kernels
 *  \author Dimitris Karatzas
 */
namespace video_utils {
// Mobius tonemap coeffs for the FFmpeg vf_tonemap.c formula K * (x + a) / (x + b): a, b, K depend ONLY on the HDR peak luminance
// (video constant, read once from MaxCLL side data), so they are computed a single time and reused for every frame/kernel
struct MobiusParams {
    float a, b, k;

    // build the coefficients from the HDR peak (npl=100 units)
    static MobiusParams fromHdrPeak(const float hdrPeak) {
        constexpr float j = 0.3f;
        const float a = -j * j * (hdrPeak - 1.0f) / (j * j - 2.0f * j + hdrPeak);
        const float b = (j * j - 2.0f * j * hdrPeak + hdrPeak) / std::max(hdrPeak - 1.0f, 1e-6f);
        const float k = (b * b + 2.0f * b * j + j * j) / (b - a);
        return {a, b, k};
    }
};
} // namespace video_utils
