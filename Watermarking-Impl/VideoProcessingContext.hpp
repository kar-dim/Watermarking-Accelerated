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

extern "C" {
#include "libavformat/avformat.h"
#include "libavfilter/avfilter.h"
#include "libavutil/pixfmt.h"
#include "libavutil/rational.h"
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
    // the pixel format every decoded frame must have (the software format of a hardware frame)
    AVPixelFormat decodedFormat = AV_PIX_FMT_NONE;
    // guessed stream frame rate and one frame at that rate, used when a frame has no duration (NVDEC frames never have one)
    AVRational frameRate{0, 1};
    int64_t nominalFrameDuration = 0;
    // pts given to the next frame that has none
    int64_t nextFramePts = 0;
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
    // output encoding (embed mode only, initialized in embedVideo, null for detect)
    video_utils::AVOutputFormatContextPtr outputFormatCtx;
    video_utils::AVCodecContextPtr outputEncoderCtx;
    video_utils::AuxiliaryMux auxMux; // handles audio remux and subtitle transcoding
    int outputVideoStreamIndex = -1;
    video_utils::PacketDurations frameDurations;
    int64_t lastWrittenVideoDts = AV_NOPTS_VALUE;
    bool reportedDtsRepair = false;
    // the newest encoded packet is held back, so the last one in decode order can be stretched to the presentation end (MP4/MOV)
    video_utils::AVPacketPtr heldPacket;
    bool holdingPacket = false;
    bool stretchLastPacket = false;
    int64_t presentationEnd = AV_NOPTS_VALUE;
    int64_t firstReorderDelay = 0;
    uint64_t encodedPackets = 0;
    // true once the output file is opened, a failed embed deletes only a file it created
    bool outputFileCreated = false;
    // convenient getter for video properties
    inline std::pair<int, int> videoDims() const { return {videoStream->codecpar->height, videoStream->codecpar->width}; }
};

} // namespace WatermarkCore
