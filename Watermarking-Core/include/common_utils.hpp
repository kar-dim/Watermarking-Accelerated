#pragma once
#include <numbers>
#include <stdexcept>
#include <string>

/*!
 *  \brief  Common color conversion constants, console styling, and assertion helpers
 *  \author Dimitris Karatzas
 */
namespace CommonUtils {
// ITU-R BT.601 RGB-to-grayscale coefficients shared by the CPU and GPU backends.
inline constexpr float kLumaR = 0.299f;
inline constexpr float kLumaG = 0.587f;
inline constexpr float kLumaB = 0.114f;

// 2 * pi for the Box-Muller transform (0x1.921fb6p+2f in single precision)
inline constexpr float kTwoPi = 2.0f * std::numbers::pi_v<float>;

// check if an error condition is true and throw with a specified error message
inline void checkError(const bool isError, const std::string& errorMsg) {
    if (isError)
        throw std::runtime_error(errorMsg);
}

// printing to console
inline std::string info(const std::string& str) { return "\033[38;5;208m" + str + "\033[0m"; }
inline std::string err(const std::string& str) { return "\033[91m" + str + "\033[0m"; }
inline std::string success(const std::string& str) { return "\033[92m" + str + "\033[0m"; }

// get the first line of the error message for cleaner output
inline std::string cleanError(const std::string& fullError) { return fullError.substr(0, fullError.find('\n')); }
} // namespace CommonUtils
