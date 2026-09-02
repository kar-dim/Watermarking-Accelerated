#pragma once

#include "AuxiliaryMux.hpp"
#include "buffer.hpp"
#include "HdrTonemap.hpp"
#include "HostMemory.hpp"
#include "include/WatermarkCore.hpp"
#include "video_defines.hpp"
#include "WatermarkBase.hpp"
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

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
    // HDR metadata cached from the decoder
    bool isHdr = false;
    video_utils::MobiusParams mobius = video_utils::MobiusParams::fromHdrPeak(10.0f); // default 1000 nits (used by HDR CUDA kernels)
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
    // output encoding (embed mode only, initialized in embedVideo, null for detect)
    video_utils::AVOutputFormatContextPtr outputFormatCtx;
    video_utils::AVCodecContextPtr outputEncoderCtx;
    video_utils::AuxiliaryMux auxMux; // handles audio remux and subtitle transcoding
    int outputVideoStreamIndex = -1;
    // convenient getter for video properties
    inline std::pair<int, int> videoDims() const { return {videoStream->codecpar->height, videoStream->codecpar->width}; }
};

} // namespace WatermarkCore