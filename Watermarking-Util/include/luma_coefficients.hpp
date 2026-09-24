#pragma once

namespace CommonUtils {
// ITU-R BT.601 RGB-to-grayscale coefficients shared by the CPU and GPU backends.
inline constexpr float kLumaR = 0.299f;
inline constexpr float kLumaG = 0.587f;
inline constexpr float kLumaB = 0.114f;
} // namespace CommonUtils
