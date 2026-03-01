#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <format>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

/*!
 *  \brief  Helper general utilities
 *  \author Dimitris Karatzas
 */
namespace CommonUtils {

inline void checkError(const bool isError, const std::string& errorMsg) {
    if (isError)
        throw std::runtime_error(errorMsg);
}

// get the first line of the error message for cleaner output
inline std::string cleanError(const std::string& fullError) { return fullError.substr(0, fullError.find('\n')); }

inline std::string formatExecutionTime(const bool showFps, const double seconds) { return showFps ? std::format("FPS: {:.2f} FPS", 1.0 / seconds) : std::format("{:.6f} seconds", seconds); }

inline std::string addSuffixBeforeExtension(const std::string& file, const std::string& suffix) {
    const auto dot = file.find_last_of('.');
    checkError(dot == std::string::npos || dot == file.size() - 1, "Filename has no valid extension: " + file);
    return file.substr(0, dot) + suffix + file.substr(dot);
}

// get valid image files from a directory, based on their extensions, and return them in a sorted vector
inline std::vector<std::filesystem::path> getValidImageFiles(const std::filesystem::path& inputDir) {
    static constexpr std::array<std::string_view, 7> validExts{".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"};
    std::vector<std::filesystem::path> validFiles;
    if (!std::filesystem::exists(inputDir) || !std::filesystem::is_directory(inputDir))
        return validFiles;
    for (const auto& entry : std::filesystem::directory_iterator(inputDir)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (std::find(validExts.begin(), validExts.end(), ext) != validExts.end())
                validFiles.push_back(entry.path());
        }
    }
    std::sort(validFiles.begin(), validFiles.end());
    return validFiles;
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