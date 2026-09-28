#pragma once
#include <bit>
#include <cstdint>

/*!
 *  \brief  IEEE 754 half precision (binary16) conversions on the host
 *  \author Dimitris Karatzas
 */
namespace HalfFloat {
// float -> half bits, round to nearest even (the rounding of the GPU conversions), overflow to infinity, NaN stays NaN
inline uint16_t fromFloat(const float value) {
    const uint32_t bits = std::bit_cast<uint32_t>(value);
    const uint32_t sign = (bits >> 16) & 0x8000u;
    const uint32_t absBits = bits & 0x7FFFFFFFu;
    if (absBits >= 0x47800000u)
        return static_cast<uint16_t>(sign | (absBits > 0x7F800000u ? 0x7E00u : 0x7C00u));
    if (absBits < 0x38800000u)
        return static_cast<uint16_t>(sign | (std::bit_cast<uint32_t>(std::bit_cast<float>(absBits) + 0.5f) - 0x3F000000u));
    const uint32_t odd = (absBits >> 13) & 1u;
    return static_cast<uint16_t>(sign | ((absBits - 0x38000000u + 0xFFFu + odd) >> 13));
}

// half bits -> float (exact)
inline float toFloat(const uint16_t half) {
    const uint32_t sign = static_cast<uint32_t>(half & 0x8000u) << 16;
    const uint32_t absBits = half & 0x7FFFu;
    if (absBits < 0x0400u)
        return std::bit_cast<float>(sign | std::bit_cast<uint32_t>(static_cast<float>(absBits) * 0x1.0p-24f));
    if (absBits >= 0x7C00u)
        return std::bit_cast<float>(sign | 0x7F800000u | ((absBits & 0x3FFu) << 13));
    return std::bit_cast<float>(sign | ((absBits << 13) + 0x38000000u));
}
} // namespace HalfFloat
