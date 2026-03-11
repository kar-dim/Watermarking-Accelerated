#pragma once

#include "buffer.hpp"
#include "include/WatermarkTypes.hpp"
#include "video_defines.hpp"
#include "VideoProcessingContext.hpp"
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <span>
#include <stdexcept>
#include <string>

extern "C" {
#include "libavcodec/avcodec.h"
#include "libavcodec/packet.h"
#include "libavformat/avformat.h"
#include "libavcodec/codec_par.h"
#include "libavutil/pixfmt.h"
#include "libavutil/error.h"
#include "libavutil/frame.h"
#include "libavutil/rational.h"
}

/*!
 *  \brief  Utility functions for video, including decoding, frames processing and writing.
 *  \author Dimitris Karatzas
 */
namespace video_utils {
// 10-bit and HDR helpers
inline bool is10bit(const AVCodecContext* codecCtx, const AVStream* st) {
    const bool is10bitCtx = codecCtx->pix_fmt == AV_PIX_FMT_YUV420P10LE || codecCtx->pix_fmt == AV_PIX_FMT_YUV420P16LE;
    return is10bitCtx || (st->codecpar->format == AV_PIX_FMT_YUV420P10LE || st->codecpar->format == AV_PIX_FMT_YUV420P16LE || st->codecpar->bits_per_raw_sample == 10);
}
// PQ HDR10 or HLG HDR
inline bool isHDR(const AVCodecContext* codecCtx) { return codecCtx->color_trc == AVCOL_TRC_SMPTE2084 || codecCtx->color_trc == AVCOL_TRC_ARIB_STD_B67; }

#if defined(_USE_CUDA_)
AVCodecContextPtr openDecoderHWAccel(const AVCodecParameters* inputCodecParams, const std::string& userHwDecoder, bool& useHwDecoder);
void embedWatermarkHWAccel(WatermarkCore::VideoSession* s, int& framesCount, const AVFrame* frame, FILE* ffmpegPipe);
void detectWatermarkHWAccel(WatermarkCore::VideoSession* s, int& framesCount, const AVFrame* frame);
#endif
std::string getFilterGraphString(const WatermarkCore::VideoSession* s);
bool initFilterGraph(WatermarkCore::VideoSession* s);

AVCodecContextPtr openDecoder(const AVCodecParameters* inputCodecParams, const std::string& userHwDecoder, bool& useHwDecoder);
AVCodecContextPtr openSoftwareDecoder(const AVCodecParameters* inputCodecParams);
bool checkPixelFormatSupport(const std::span<const AVPixelFormat> supportedFormats, const AVPixelFormat format);
std::string getFrameRate(const AVStream* st);
AVRational getTimeBase(const AVStream* st);
inline std::string getPixFmt(const AVStream* st) { return st->codecpar->color_range == AVCOL_RANGE_JPEG ? "-pix_fmt yuvj420p " : "-pix_fmt yuv420p "; }
inline std::string getColorRange(const AVStream* st) { return st->codecpar->color_range == AVCOL_RANGE_JPEG ? "-color_range:v:0 pc " : "-color_range:v:0 tv "; }
std::string getStreamRotation(const AVStream* st);
int findVideoStream(const AVFormatContext* inputFormatCtx);
void embedWatermark(WatermarkCore::VideoSession* s, int& framesCount, const AVFrame* frame, FILE* ffmpegPipe);
void detectWatermark(WatermarkCore::VideoSession* s, int& framesCount, const AVFrame* frame);
void embedAndWriteFrame(WatermarkCore::VideoSession* s, const ImageBuffer& buffer, const int elements, FILE* ffmpegPipe);
void processAndWriteYPlane(const bool embedWatermark, const AVFrame* frame, WatermarkCore::VideoSession* s, FILE* ffmpegPipe);
void writeChromaPlanes(const AVFrame* frame, WatermarkCore::VideoSession* s, FILE* ffmpegPipe);
int videoDispatcher(WatermarkCore::VideoSession* s, VideoMode op, const bool needsFiltert = false, FILE* ffmpegPipe = nullptr);
void filterFrame(AVFramePtr& frame, AVFramePtr& filteredFrame, const WatermarkCore::VideoSession* s);
void loadInputFrame(WatermarkCore::VideoSession* s, const uint8_t* hostPtr);

// main frames loop logic for video watermark embedding and detection
template <bool needsFilter, typename Func>
int processFrames(const WatermarkCore::VideoSession* s, Func&& processFrame) {
    const AVPacketPtr packet(av_packet_alloc());
    AVFramePtr frame(av_frame_alloc());
    AVFramePtr filteredFrame(nullptr);
    if constexpr (needsFilter)
        filteredFrame.reset(av_frame_alloc());
    int framesCount = 0;

    // read video frames loop
    while (av_read_frame(s->inputFormatCtx.get(), packet.get()) >= 0) {
        if (packet->stream_index != s->videoStreamIndex || avcodec_send_packet(s->inputDecoderCtx.get(), packet.get()) < 0) {
            av_packet_unref(packet.get());
            continue;
        }
        while (true) {
            const int ret = avcodec_receive_frame(s->inputDecoderCtx.get(), frame.get());
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
                break;
            if (ret < 0) {
                char errbuf[256];
                av_strerror(ret, errbuf, sizeof(errbuf));
                av_packet_unref(packet.get());
                throw std::runtime_error(std::string("FFmpeg decoding error: ") + errbuf);
            }
            // optionally filter frame (10-bit to 8-bit conversion, HDR to SDR tonemapping)
            if constexpr (needsFilter)
                filterFrame(frame, filteredFrame, s);
            std::forward<Func>(processFrame)(frame.get(), framesCount);
        }
        av_packet_unref(packet.get());
    }
    // ensure all remaining frames are flushed
    avcodec_send_packet(s->inputDecoderCtx.get(), nullptr);
    while (avcodec_receive_frame(s->inputDecoderCtx.get(), frame.get()) == 0) {
        if constexpr (needsFilter)
            filterFrame(frame, filteredFrame, s);
        std::forward<Func>(processFrame)(frame.get(), framesCount);
    }
    return framesCount;
}
} // namespace video_utils