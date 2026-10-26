#pragma once

#include "AuxiliaryMux.hpp"
#include "buffer.hpp"
#include "HdrTonemap.hpp"
#include "OutputFile.hpp"
#include "include/WatermarkCore.hpp"
#include "video_defines.hpp"
#include "WatermarkBase.hpp"
#include <cstdint>
#include <memory>
#include <utility>

#if defined(_USE_OPENCL_)
#include "OclVideoInterop.hpp"
#include "OclHdrTonemap.hpp"
#endif

extern "C" {
#include "libavformat/avformat.h"
#include "libavfilter/avfilter.h"
#include "libavutil/pixfmt.h"
#include "libavutil/rational.h"
}

namespace WatermarkCore {
/*!
 *  \brief  The unified internal session for video processing,
 *          this is hidden from the public API but shared internally
 *  \author Dimitris Karatzas
 */
struct VideoSession {
    VideoSettings settings;
    video_utils::AVFormatContextPtr inputFormatCtx;
    video_utils::AVCodecContextPtr inputDecoderCtx;
    const AVStream* videoStream = nullptr;
    int videoStreamIndex = -1;
    bool useHwDecoder = false;
    // decoded frames held before processing, decoder surfaces must cover them
    size_t decodeLookahead = 0;
#if defined(_USE_OPENCL_)
    // QSV direct mappings retain source frame without D3D11 frames descriptor
    video_utils::AVBufferRefPtr openclMappedFrames;
    std::unique_ptr<OclVideoFrame> openclVideoFrame;
    std::unique_ptr<OclHdrTonemap> openclHdrTonemap;
    std::unique_ptr<OclVideoOutput> openclVideoOutput;
#endif
    // software format every decoded frame must have
    AVPixelFormat decodedFormat = AV_PIX_FMT_NONE;
    // 8-bit planar YUV format of encoded frames, none in detection
    AVPixelFormat outputFormat = AV_PIX_FMT_NONE;
    // guessed stream frame rate used when frame has no duration
    AVRational frameRate{0, 1};
    int64_t nominalFrameDuration = 0;
    // pts for next frame without timestamp
    int64_t nextFramePts = 0;
    // HDR input tonemapped to SDR
    bool isHdr = false;
    std::unique_ptr<WatermarkBase> watermarkObj;
    // reused software encoder frame buffers, pinned on GPU builds for direct readback
    video_utils::AVBufferPoolPtr encoderFramePool;
    // optional filter graph for 10-bit conversion and tonemapping
    video_utils::AVFilterGraphPtr filterGraph;
    AVFilterContext* buffersrcCtx = nullptr;
    AVFilterContext* buffersinkCtx = nullptr;
    // frame processing buffers reused per frame
    ImageBuffer inputFrame;
    ImageOutputBuffer watermarkedFrame;
    // output encoding in embed mode
    OutputFile outputFile; // destroyed after FFmpeg contexts close handles
    video_utils::AVOutputFormatContextPtr outputFormatCtx;
    video_utils::AVCodecContextPtr outputEncoderCtx;
    video_utils::AuxiliaryMux auxMux; // audio remux and subtitle transcode
    int outputVideoStreamIndex = -1;
    video_utils::PacketDurations frameDurations;
    int64_t lastWrittenVideoDts = AV_NOPTS_VALUE;
    bool reportedDtsRepair = false;
    // newest encoded packet is held back to stretch last packet in decode order to presentation end
    video_utils::AVPacketPtr heldPacket;
    bool holdingPacket = false;
    bool stretchLastPacket = false;
    int64_t presentationEnd = AV_NOPTS_VALUE;
    int64_t firstReorderDelay = 0;
    uint64_t encodedPackets = 0;
    inline std::pair<int, int> videoDims() const { return {videoStream->codecpar->height, videoStream->codecpar->width}; }
};

} // namespace WatermarkCore
