#pragma once

/*!
 *  \brief  libav error reporting, frame timestamp helpers, and utility functions
 *  \author Dimitris Karatzas
 */

#include "video_defines.hpp"
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>

extern "C" {
#include "libavcodec/avcodec.h"
#include "libavcodec/codec.h"
#include "libavutil/avutil.h"
#include "libavutil/dict.h"
#include "libavutil/error.h"
#include "libavutil/frame.h"
#include "libavutil/pixfmt.h"
}

namespace video_utils {

using MediaLog = std::function<void(const std::string&)>;

// human readable libav error string
inline std::string avErrorText(const int avRet) {
    char buf[AV_ERROR_MAX_STRING_SIZE] = {0};
    av_strerror(avRet, buf, sizeof(buf));
    return buf;
}

// throws runtime_error containing formatted FFmpeg error description
inline void checkAv(const int avRet, const std::string& what) {
    if (avRet < 0)
        throw std::runtime_error(what + ": " + avErrorText(avRet));
}

// retrieve PTS for a decoded frame (falls back to best_effort_timestamp if pts is missing)
inline int64_t framePts(const AVFrame* f) { return (f->pts != AV_NOPTS_VALUE) ? f->pts : f->best_effort_timestamp; }

inline bool enforceMonotonicDts(AVPacket* packet, int64_t& lastWrittenDts) {
    if (packet->dts == AV_NOPTS_VALUE)
        return false;
    bool repaired = false;
    if (packet->pts != AV_NOPTS_VALUE && packet->dts > packet->pts) {
        packet->dts = packet->pts;
        repaired = true;
    }
    if (lastWrittenDts != AV_NOPTS_VALUE && packet->dts <= lastWrittenDts) {
        if (lastWrittenDts == std::numeric_limits<int64_t>::max())
            throw std::runtime_error("Encoded video DTS exceeds the supported timestamp range");
        const int64_t earliest = lastWrittenDts + 1;
        if (packet->pts != AV_NOPTS_VALUE && packet->pts < earliest)
            packet->pts = earliest;
        packet->dts = earliest;
        repaired = true;
    }
    lastWrittenDts = packet->dts;
    return repaired;
}

class PacketDurations {
  public:
    void remember(const AVFrame* frame) {
        if (frame->pts != AV_NOPTS_VALUE && frame->duration > 0)
            byPts_.insert_or_assign(frame->pts, frame->duration);
        if (byPts_.size() > 4096)
            byPts_.erase(byPts_.begin());
    }
    void restore(AVPacket* packet) {
        const auto found = byPts_.find(packet->pts);
        if (found == byPts_.end())
            return;
        if (packet->duration <= 0)
            packet->duration = found->second;
        byPts_.erase(found);
    }

  private:
    std::map<int64_t, int64_t> byPts_;
};

// RAII wrapper for AVDictionary*
class OptionDict {
  public:
    explicit OptionDict(AVDictionary* dict = nullptr) noexcept : m_dict(dict) {}
    ~OptionDict() { av_dict_free(&m_dict); }
    OptionDict(const OptionDict&) = delete;
    OptionDict& operator=(const OptionDict&) = delete;
    OptionDict(OptionDict&& other) noexcept : m_dict(other.m_dict) { other.m_dict = nullptr; }
    OptionDict& operator=(OptionDict&& other) noexcept {
        if (this != &other) {
            av_dict_free(&m_dict);
            m_dict = other.m_dict;
            other.m_dict = nullptr;
        }
        return *this;
    }
    AVDictionary** ptr() noexcept { return &m_dict; }
    const AVDictionary* get() const noexcept { return m_dict; }
    AVDictionary* release() noexcept {
        AVDictionary* d = m_dict;
        m_dict = nullptr;
        return d;
    }

  private:
    AVDictionary* m_dict = nullptr;
};

// checks if both paths point to the same physical file on disk to prevent overwriting
inline bool sameFileOnDisk(const std::filesystem::path& left, const std::filesystem::path& right) {
    std::error_code error;
    return std::filesystem::equivalent(left, right, error) && !error;
}

// checks if encoder supports the requested pixel format
inline bool encoderSupportsPixFmt(const AVCodec* encoder, const AVPixelFormat want) {
    const void* configs = nullptr;
    if (avcodec_get_supported_config(nullptr, encoder, AV_CODEC_CONFIG_PIX_FORMAT, 0, &configs, nullptr) < 0 || !configs)
        return true;
    for (const auto* fmt = static_cast<const AVPixelFormat*>(configs); *fmt != AV_PIX_FMT_NONE; ++fmt)
        if (*fmt == want)
            return true;
    return false;
}

} // namespace video_utils
