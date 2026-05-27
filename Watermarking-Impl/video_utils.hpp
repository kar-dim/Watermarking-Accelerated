#pragma once

#include "buffer.hpp"
#include "include/WatermarkTypes.hpp"
#include "video_defines.hpp"
#include "VideoProcessingContext.hpp"
#include <cstdint>
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
void embedWatermarkHWAccel(WatermarkCore::VideoSession* s, int& framesCount, const AVFrame* frame);
void detectWatermarkHWAccel(WatermarkCore::VideoSession* s, int& framesCount, const AVFrame* frame);
#endif
std::string getFilterGraphString(const WatermarkCore::VideoSession* s);
bool initFilterGraph(WatermarkCore::VideoSession* s);

AVCodecContextPtr openDecoder(const AVCodecParameters* inputCodecParams, const std::string& userHwDecoder, bool& useHwDecoder);
AVCodecContextPtr openSoftwareDecoder(const AVCodecParameters* inputCodecParams);
bool checkPixelFormatSupport(const std::span<const AVPixelFormat> supportedFormats, const AVPixelFormat format);
int findVideoStream(const AVFormatContext* inputFormatCtx);
void embedWatermark(WatermarkCore::VideoSession* s, int& framesCount, const AVFrame* frame);
void detectWatermark(WatermarkCore::VideoSession* s, int& framesCount, const AVFrame* frame);
void embedAndFillYPlane(WatermarkCore::VideoSession* s, const ImageBuffer& buffer);
void fillYPlane(const bool doEmbed, const AVFrame* frame, WatermarkCore::VideoSession* s);
void fillChromaPlanes(const AVFrame* frame, WatermarkCore::VideoSession* s);
int videoDispatcher(WatermarkCore::VideoSession* s, VideoMode op, const bool needsFilter = false);
void filterFrame(AVFramePtr& frame, AVFramePtr& filteredFrame, const WatermarkCore::VideoSession* s);
void loadInputFrame(WatermarkCore::VideoSession* s, const uint8_t* hostPtr);
// output encoder lifecycle
void initOutputEncoder(WatermarkCore::VideoSession* s);
void encodeFrame(WatermarkCore::VideoSession* s, int64_t pts);
void flushAndFinalize(WatermarkCore::VideoSession* s);

// main frames loop logic for video watermark embedding and detection.
// when s->outputFormatCtx is set (embed mode), non-video packets (audio, subtitles)
// are remuxed directly to the output, otherwise they are discarded (detect mode)
template <bool needsFilter, typename Func>
int processFrames(WatermarkCore::VideoSession* s, Func&& processFrame) {
    const AVPacketPtr packet(av_packet_alloc());
    AVFramePtr frame(av_frame_alloc());
    AVFramePtr filteredFrame(nullptr);
    if constexpr (needsFilter)
        filteredFrame.reset(av_frame_alloc());
    int framesCount = 0;

    while (av_read_frame(s->inputFormatCtx.get(), packet.get()) >= 0) {
        if (packet->stream_index == s->videoStreamIndex) {
            if (avcodec_send_packet(s->inputDecoderCtx.get(), packet.get()) >= 0) {
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
            }
        } else if (s->outputFormatCtx) {
            // remux non video streams (audio, subtitles) directly to output
            const int inIdx = packet->stream_index;
            if (inIdx < static_cast<int>(s->inputToOutputStreamMap.size())) {
                const int outIdx = s->inputToOutputStreamMap[inIdx];
                if (outIdx >= 0) {
                    av_packet_rescale_ts(packet.get(),
                        s->inputFormatCtx->streams[inIdx]->time_base,
                        s->outputFormatCtx->streams[outIdx]->time_base);
                    packet->stream_index = outIdx;
                    av_interleaved_write_frame(s->outputFormatCtx.get(), packet.get());
                }
            }
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