#pragma once

#include <chrono>
#include <filesystem>
#include <format>
#include <stdexcept>
#include <string>
#include <vector>

/*!
 *  \brief  Helper general utilities
 *  \author Dimitris Karatzas
 */
namespace CommonUtils {

std::string addSuffixBeforeExtension(const std::string& file, const std::string& suffix);
std::vector<std::filesystem::path> getValidImageFiles(const std::filesystem::path& inputDir);
double calculateCV(const std::vector<double>& times);

// printing to console functions
inline std::string info(const std::string& str) { return "\033[38;5;208m" + str + "\033[0m"; }
inline std::string err(const std::string& str) { return "\033[91m" + str + "\033[0m"; }
inline std::string success(const std::string& str) { return "\033[92m" + str + "\033[0m"; }

// check if an error condition is true and throw with a specified error message
inline void checkError(const bool isError, const std::string& errorMsg) {
    if (isError)
        throw std::runtime_error(errorMsg);
}

// get the first line of the error message for cleaner output
inline std::string cleanError(const std::string& fullError) { return fullError.substr(0, fullError.find('\n')); }

inline std::string formatExecutionTime(const bool showFps, const double seconds) { return showFps ? std::format("FPS: {:.2f} FPS", 1.0 / seconds) : std::format("{:.6f} seconds", seconds); }

// measures execution time for a passed function
template <typename F>
double executionTime(F&& func, int loops = 1, const bool warmup = true) {
    if (warmup)
        func(); // warmup one time
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < loops; i++)
        func();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end - start).count();
}
} // namespace CommonUtils