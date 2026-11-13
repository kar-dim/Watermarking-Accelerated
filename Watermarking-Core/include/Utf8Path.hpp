#pragma once

#include <filesystem>
#include <string>
#include <string_view>

/*!
 * \brief UTF-8 filesystem path conversion shared by the core and CLI
 * \author Dimitris Karatzas
 */

namespace WatermarkCore {
// char8_t selects UTF-8 conversion instead of the Windows code page
inline std::filesystem::path pathFromUtf8(const std::string_view path) { return std::filesystem::path(std::u8string(path.begin(), path.end())); }
} // namespace WatermarkCore
