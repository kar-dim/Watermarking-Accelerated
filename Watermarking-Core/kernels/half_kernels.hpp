#pragma once
#include <string>

/*!
 *  \brief  OpenCL kernel source for deterministic IEEE 754 half-precision float conversions
 *  \author Dimitris Karatzas
 */
// OpenCL float to half rounding, matching HalfFloat::fromFloat and __float2half_rn
// native storage conversion on Intel, integer fallback on other drivers
inline const std::string halfKernels = R"CLC(
#if defined(WM_NATIVE_HALF_STORAGE)
static inline __attribute__((always_inline)) ushort toHalfBits(const float value) {
    const uint bits = as_uint(value);
    if ((bits & 0x7FFFFFFFu) > 0x7F800000u)
        return (ushort)(((bits >> 16) & 0x8000u) | 0x7E00u);
    ushort result;
    vstore_half_rte(value, 0, (__private half*)&result);
    return result;
}
#else
// float to half bits, round to nearest even, overflow to infinity
// integer fallback avoids driver tie-rounding bugs in vstore_half_rte
static inline __attribute__((always_inline)) ushort toHalfBits(const float value) {
    const uint bits = as_uint(value);
    const uint sign = (bits >> 16) & 0x8000u;
    const uint absBits = bits & 0x7FFFFFFFu;
    if (absBits >= 0x47800000u)
        return (ushort)(sign | (absBits > 0x7F800000u ? 0x7E00u : 0x7C00u));
    // subnormal half: the float addition rounds the value to a multiple of 2^-24 (the mantissa bits of 0.5f + value)
    if (absBits < 0x38800000u)
        return (ushort)(sign | (as_uint(as_float(absBits) + 0.5f) - 0x3F000000u));
    const uint odd = (absBits >> 13) & 1u;
    return (ushort)(sign | ((absBits - 0x38000000u + 0xFFFu + odd) >> 13));
}
#endif
)CLC";
