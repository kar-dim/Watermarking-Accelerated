#pragma once

#include <chrono>
#include <format>
#include <stdexcept>
#include <string>

/*!
 *  \brief  Helper general utilities
 *  \author Dimitris Karatzas
 */
namespace CommonUtils {

inline void checkError(const bool isError, const std::string& errorMsg) {
    if (isError)
        throw std::runtime_error(errorMsg);
}
inline std::string formatExecutionTime(const bool showFps, const double seconds) { return showFps ? std::format("FPS: {:.2f} FPS", 1.0 / seconds) : std::format("{:.6f} seconds", seconds); }
inline std::string addSuffixBeforeExtension(const std::string& file, const std::string& suffix) {
    const auto dot = file.find_last_of('.');
    checkError(dot == std::string::npos || dot == file.size() - 1, "Filename has no valid extension: " + file);
    return file.substr(0, dot) + suffix + file.substr(dot);
}

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