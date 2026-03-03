#pragma once

#include "buffer.hpp"
#include "HostMemory.hpp"
#include "include/WatermarkCore.hpp"
#include "video_defines.hpp"
#include "WatermarkBase.hpp"
#include <cstdint>
#include <memory>

extern "C" {
#include "libavformat/avformat.h"
#include "libavfilter/avfilter.h"
}

namespace WatermarkCore {
/*!
 * \brief  The unified internal session for video processing,
 *         this is hidden from the public API but shared internally
 */
struct VideoSession {
    VideoSettings settings;
    // ffmpeg contexts and stream info
    video_utils::AVFormatContextPtr inputFormatCtx;
    video_utils::AVCodecContextPtr inputDecoderCtx;
    const AVStream* videoStream = nullptr;
    int videoStreamIndex = -1;
    bool useHwDecoder = false;
    // watermarking related buffers and objects
    std::unique_ptr<WatermarkBase> watermarkObj;
    std::unique_ptr<HostMemory<uint8_t>> hostFrame;
    // optional filter graph context for 10-bit to 8-bit conversion and HDR to SDR tonemapping, initialized only when needed
    video_utils::AVFilterGraphPtr filterGraph;
    AVFilterContext* buffersrcCtx = nullptr;
    AVFilterContext* buffersinkCtx = nullptr;
    // image buffers for processing the frames, reused for each frame to save memory allocations
    ImageBuffer inputFrame;
    ImageOutputBuffer watermarkedFrame;
    Gray8Buffer grayFrame;
    // convenient getters for video properties
    inline int width() const { return videoStream->codecpar->width; }
    inline int height() const { return videoStream->codecpar->height; }
};

} // namespace WatermarkCore