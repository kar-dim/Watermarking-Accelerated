#pragma once
#include <string>

// OpenCL float -> half rounding, the same bits as HalfFloat::fromFloat on the host and __float2half_rn in CUDA
// (half -> float with vload_half is exact and stays as is), because some drivers are bugged on the vstore_half_rte
inline const std::string halfKernels = R"CLC(
// float -> half bits, round to nearest even (HalfFloat::fromFloat of the host), overflow to infinity.
// NOTE: Integer code on purpose, some drivers like AMD are bugged and rounded exact ties AWAY from zero in vstore_half_rte (depending on the surrounding code)
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
)CLC";
