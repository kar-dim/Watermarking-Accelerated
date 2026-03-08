#include "include/common_utils.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <numeric>
#include <string>
#include <string_view>
#include <vector>

using std::string;
using std::string_view;

namespace CommonUtils {

string addSuffixBeforeExtension(const string& file, const string& suffix) {
    const auto dot = file.find_last_of('.');
    checkError(dot == string::npos || dot == file.size() - 1, "Filename has no valid extension: " + file);
    return file.substr(0, dot) + suffix + file.substr(dot);
}

// retrieve a list of valid image file paths based on their extension (case insensitive)
std::vector<std::filesystem::path> getValidImageFiles(const std::filesystem::path& inputDir) {
    static constexpr std::array<string_view, 7> validExts{".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"};

    std::vector<std::filesystem::path> validFiles;
    if (!std::filesystem::exists(inputDir) || !std::filesystem::is_directory(inputDir))
        return validFiles;
    for (const auto& entry : std::filesystem::directory_iterator(inputDir)) {
        if (!entry.is_regular_file())
            continue;
        string ext = entry.path().extension().string();
        std::ranges::transform(ext, ext.begin(), [](unsigned char c) { return std::tolower(c); });
        if (std::ranges::find(validExts, ext) != validExts.end())
            validFiles.push_back(entry.path());
    }
    std::sort(validFiles.begin(), validFiles.end());
    return validFiles;
}

// calculate coefficient of variation for a sequence of execution times
double calculateCV(const std::vector<double>& times) {
    if (times.empty())
        return 0.0;

    const double mean = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    const double sqSum = std::inner_product(times.begin(), times.end(), times.begin(), 0.0);
    const double variance = sqSum / times.size() - mean * mean;
    const double stdev = variance > 0.0 ? std::sqrt(variance) : 0.0;
    return mean > 0.0 ? stdev / mean : 0.0;
}

} // namespace CommonUtils