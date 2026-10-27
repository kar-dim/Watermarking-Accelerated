#pragma once

#include "AvUtil.hpp"
#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <system_error>
#include <windows.h>

/*!
 *  \brief  RAII helper for atomic file output creation, replacement, and cleanup
 *  \author Dimitris Karatzas
 */

// preserve destination while encoding into temporary file beside it
class OutputFile {
    std::filesystem::path temporary;
    std::filesystem::path destination;

  public:
    OutputFile() = default;
    OutputFile(const OutputFile&) = delete;
    OutputFile& operator=(const OutputFile&) = delete;
    ~OutputFile() { discard(); }

    // reserve unique file on same filesystem for atomic replacement
    void prepare(const std::string& outputPath, const bool keepExtension = false) {
        if (!temporary.empty())
            throw std::logic_error("An output file is already in progress");
        destination = std::filesystem::absolute(video_utils::pathFromUtf8(outputPath));
        static std::atomic<uint64_t> sequence{0};
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        for (int attempt = 0; attempt < 256; ++attempt) {
            auto candidate = destination.parent_path() / (L".watermark-" + std::to_wstring(GetCurrentProcessId()) + L"-" + std::to_wstring(stamp) + L"-" + std::to_wstring(sequence.fetch_add(1)) +
                                                             L".tmp" + (keepExtension ? destination.extension().native() : L""));
            // CREATE_NEW avoids claiming another encoder temporary file
            const HANDLE file = CreateFileW(candidate.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
            if (file != INVALID_HANDLE_VALUE) {
                temporary = std::move(candidate);
                CloseHandle(file);
                return;
            }
            const DWORD error = GetLastError();
            if (error != ERROR_FILE_EXISTS && error != ERROR_ALREADY_EXISTS)
                throw std::system_error(error, std::system_category(), "Could not create temporary output file");
        }
        throw std::runtime_error("Could not reserve a unique temporary output file");
    }

    // FFmpeg expects temporary filename in UTF-8
    std::string path() const {
        const auto utf8 = temporary.u8string();
        return std::string(utf8.begin(), utf8.end());
    }

    // publish completed file after encoder and muxer close it
    void commit(const bool replaceExisting = true) {
        if (temporary.empty())
            throw std::logic_error("No output file to finalize");
        const DWORD flags = MOVEFILE_WRITE_THROUGH | (replaceExisting ? MOVEFILE_REPLACE_EXISTING : 0);
        if (!MoveFileExW(temporary.c_str(), destination.c_str(), flags))
            throw std::system_error(GetLastError(), std::system_category(), replaceExisting ? "Could not replace output file" : "Could not publish output file");
        temporary.clear();
    }

    // failure cleanup removes only temporary file
    void discard() noexcept {
        if (!temporary.empty()) {
            std::error_code ignored;
            std::filesystem::remove(temporary, ignored);
            temporary.clear();
        }
    }
};
