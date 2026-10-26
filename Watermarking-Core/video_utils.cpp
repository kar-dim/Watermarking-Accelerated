#include "AuxiliaryMux.hpp"
#include "AvUtil.hpp"
#include "buffer.hpp"
#include "common_utils.hpp"
#include "EncodeOptions.hpp"
#include "HostMemory.hpp"
#include "include/WatermarkCore.hpp"
#include "video_defines.hpp"
#include "video_utils.hpp"
#include "VideoProcessingContext.hpp"
#include "WatermarkBase.hpp"
#include <algorithm>
#include <cerrno>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <exception>
#include <format>
#include <functional>
#include <iostream>
#include <iterator>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>
#include <variant>

#if defined(_USE_CUDA_)
#include "cuda_utils.hpp"
#include "CudaArray.hpp"
#include "CudaCheck.hpp"
#include "CudaStreamManager.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
extern "C" {
#include "libavcodec/codec_id.h"
#include "libavutil/buffer.h"
#include "libavutil/hwcontext_cuda.h"
}
#elif defined(_USE_OPENCL_)
#include "OclQueueManager.hpp"
#include "OclVideoDevice.hpp"
#include "OclArray.hpp"
#include "opencl_utils.hpp"
#include <d3d11.h>
extern "C" {
#include "libavutil/hwcontext_d3d11va.h"
}
#endif

extern "C" {
#include "libavcodec/avcodec.h"
#include "libavcodec/codec.h"
#include "libavcodec/codec_par.h"
#include "libavcodec/packet.h"
#include "libavfilter/avfilter.h"
#include "libavfilter/buffersink.h"
#include "libavfilter/buffersrc.h"
#include "libavformat/avformat.h"
#include "libavformat/avio.h"
#include "libavutil/avutil.h"
#include "libavutil/dict.h"
#include "libavutil/error.h"
#include "libavutil/frame.h"
#include "libavutil/mastering_display_metadata.h"
#include "libavutil/mem.h"
#include "libavutil/pixdesc.h"
#include "libavutil/pixfmt.h"
#include "libavutil/rational.h"
#include "libavutil/hwcontext.h"
#include "libavutil/imgutils.h"
#include "libavcodec/defs.h"
#include "HdrTonemap.hpp"
}

#if defined(_USE_EIGEN_)
using namespace Eigen;
#endif

/*!
 *  \brief  Video decode, encode, filter graph, HW acceleration, and frame processing pipelines
 *  \author Dimitris Karatzas
 */

using namespace CommonUtils;
using namespace WatermarkCore;
using std::cout;
using std::string;

namespace {
#if defined(_USE_CUDA_)
constexpr AVPixelFormat supportedHwFormats[] = {
    AV_PIX_FMT_NV12, AV_PIX_FMT_P010LE, AV_PIX_FMT_P016LE, AV_PIX_FMT_NV16, AV_PIX_FMT_P210LE, AV_PIX_FMT_P216LE, AV_PIX_FMT_YUV444P, AV_PIX_FMT_YUV444P10MSBLE, AV_PIX_FMT_YUV444P16LE};
#endif
} // namespace

namespace video_utils {
namespace {

#if defined(_USE_CUDA_)
bool isPixelFormatSupported(const std::span<const AVPixelFormat> formats, const AVPixelFormat format) {
    return std::ranges::any_of(formats, [&](auto f) { return f == format; });
}
#endif

// queue decoupling decode and watermark main thread from encode background thread
struct EncodeQueue {
    using Item = std::variant<AVFramePtr, AVPacketPtr>;
    static constexpr int MAX_DEPTH = 8;
    std::queue<Item> q;
    std::mutex mtx;
    std::condition_variable cv;
    bool closed = false;
    bool aborted = false;

    [[nodiscard]] bool push(Item item) {
        std::unique_lock lk(mtx);
        cv.wait(lk, [&] { return static_cast<int>(q.size()) < MAX_DEPTH || aborted; });
        if (aborted)
            return false;
        q.push(std::move(item));
        cv.notify_all();
        return true;
    }

    void close() {
        std::unique_lock lk(mtx);
        closed = true;
        cv.notify_all();
    }

    std::optional<Item> pop() {
        std::unique_lock lk(mtx);
        cv.wait(lk, [&] { return !q.empty() || closed || aborted; });
        if (q.empty())
            return std::nullopt;
        Item item = std::move(q.front());
        q.pop();
        cv.notify_all();
        return item;
    }

    void abort() {
        std::unique_lock lk(mtx);
        aborted = true;
        cv.notify_all();
    }
};

std::string pixelFormatName(const AVPixelFormat format) {
    const char* name = av_get_pix_fmt_name(format);
    return name != nullptr ? name : "?";
}

std::string resolutionChangeMessage(const int expectedWidth, const int expectedHeight, const int width, const int height) {
    return std::format("Video dimensions changed during decode (expected {}x{}, got {}x{}). Split or re-encode the source before watermarking.", expectedWidth, expectedHeight, width, height);
}

std::string pixelFormatChangeMessage(const AVPixelFormat expected, const AVPixelFormat actual) {
    return std::format("Video pixel format changed during decode (expected {}, got {}). Split or re-encode the source before watermarking.", pixelFormatName(expected), pixelFormatName(actual));
}

// full range formats have same layout, parsers never report them
AVPixelFormat withoutFullRangeAlias(const AVPixelFormat format) {
    switch (format) {
    case AV_PIX_FMT_YUVJ420P: return AV_PIX_FMT_YUV420P;
    case AV_PIX_FMT_YUVJ422P: return AV_PIX_FMT_YUV422P;
    case AV_PIX_FMT_YUVJ444P: return AV_PIX_FMT_YUV444P;
    case AV_PIX_FMT_YUVJ440P: return AV_PIX_FMT_YUV440P;
    case AV_PIX_FMT_YUVJ411P: return AV_PIX_FMT_YUV411P;
    default: return format;
    }
}

// layout of decoded frame, software format for hardware frame
AVPixelFormat decodedPixelFormat(const AVFrame* frame) {
    if (frame->hw_frames_ctx == nullptr)
        return static_cast<AVPixelFormat>(frame->format);
    return reinterpret_cast<const AVHWFramesContext*>(frame->hw_frames_ctx->data)->sw_format;
}

// 8-bit planar YUV format with chroma resolution of input, 4:2:0 for grayscale
AVPixelFormat planar8Bit(const AVPixelFormat format) {
    const auto* descriptor = av_pix_fmt_desc_get(format);
    if (!descriptor || descriptor->nb_components < 3 || (descriptor->flags & (AV_PIX_FMT_FLAG_RGB | AV_PIX_FMT_FLAG_PAL)))
        return AV_PIX_FMT_YUV420P;
    for (const auto planar : {AV_PIX_FMT_YUV420P, AV_PIX_FMT_YUV422P, AV_PIX_FMT_YUV444P, AV_PIX_FMT_YUV440P, AV_PIX_FMT_YUV411P, AV_PIX_FMT_YUV410P}) {
        const auto* candidate = av_pix_fmt_desc_get(planar);
        if (candidate->log2_chroma_w == descriptor->log2_chroma_w && candidate->log2_chroma_h == descriptor->log2_chroma_h)
            return planar;
    }
    return AV_PIX_FMT_YUV420P;
}

bool keepsChroma(const AVPixelFormat target, const AVPixelFormat source) {
    const auto* kept = av_pix_fmt_desc_get(target);
    const auto* original = av_pix_fmt_desc_get(source);
    return kept && original && kept->log2_chroma_w <= original->log2_chroma_w && kept->log2_chroma_h <= original->log2_chroma_h;
}

// luma is plane 0 with one byte per sample
bool hasPlanarLuma8(const AVPixelFormat format) {
    const auto* descriptor = av_pix_fmt_desc_get(format);
    if (!descriptor || (descriptor->flags & (AV_PIX_FMT_FLAG_RGB | AV_PIX_FMT_FLAG_PAL | AV_PIX_FMT_FLAG_BITSTREAM | AV_PIX_FMT_FLAG_HWACCEL)))
        return false;
    const AVComponentDescriptor& luma = descriptor->comp[0];
    return luma.plane == 0 && luma.step == 1 && luma.offset == 0 && luma.shift == 0 && luma.depth == 8;
}

// 8-bit CUDA surface layout that NVDEC frames convert to for planar output
AVPixelFormat cudaSurfaceFormat(const AVPixelFormat planar) {
    switch (planar) {
    case AV_PIX_FMT_YUV422P: return AV_PIX_FMT_NV16;
    case AV_PIX_FMT_YUV444P: return AV_PIX_FMT_YUV444P;
    default: return AV_PIX_FMT_NV12;
    }
}

#if defined(_USE_OPENCL_)
AVPixelFormat openclSurfaceFormat(const AVPixelFormat planar) {
    switch (planar) {
    case AV_PIX_FMT_YUV422P: return AV_PIX_FMT_YUYV422;
    case AV_PIX_FMT_YUV444P: return AV_PIX_FMT_VUYX;
    default: return AV_PIX_FMT_NV12;
    }
}
#endif

// planes of 8-bit planar YUV frame packed without row padding
struct PlanarLayout {
    int width = 0;
    int height = 0;
    int chromaWidth = 0;
    int chromaHeight = 0;
    size_t lumaBytes() const { return static_cast<size_t>(width) * height; }
    size_t chromaBytes() const { return static_cast<size_t>(chromaWidth) * chromaHeight; }
    size_t bytes() const { return lumaBytes() + 2 * chromaBytes(); }
    size_t offset(const int plane) const { return plane == 0 ? 0 : lumaBytes() + (plane - 1) * chromaBytes(); }
};

PlanarLayout outputLayout(const VideoSession* s) {
    const auto* descriptor = av_pix_fmt_desc_get(s->outputFormat);
    checkError(!descriptor, "The video output format is not negotiated");
    const auto [height, width] = s->videoDims();
    return {width, height, AV_CEIL_RSHIFT(width, descriptor->log2_chroma_w), AV_CEIL_RSHIFT(height, descriptor->log2_chroma_h)};
}

// bitstream parser detects mid-stream size or format changes hidden by hardware decoders
class StreamShapeWatch {
  public:
    explicit StreamShapeWatch(const AVCodecParameters* params)
        : declaredWidth_(params->width), declaredHeight_(params->height), declaredFormat_(static_cast<AVPixelFormat>(params->format)), parser_(av_parser_init(params->codec_id)) {
        if (!parser_)
            return;
        parserContext_.reset(avcodec_alloc_context3(nullptr));
        if (!parserContext_ || avcodec_parameters_to_context(parserContext_.get(), params) < 0) {
            parser_.reset();
            return;
        }
        parser_->flags |= PARSER_FLAG_COMPLETE_FRAMES;
    }

    std::string inspect(const AVPacket* packet) { return parse(packet->data, packet->size, packet->pts, packet->dts, packet->pos); }
    std::string flush() { return parse(nullptr, 0, AV_NOPTS_VALUE, AV_NOPTS_VALUE, -1); }

  private:
    struct ParserDeleter {
        void operator()(AVCodecParserContext* parser) const noexcept { av_parser_close(parser); }
    };

    std::string parse(const uint8_t* data, int size, const int64_t pts, const int64_t dts, const int64_t pos) {
        if (!parser_)
            return {};
        do {
            uint8_t* parsedData = nullptr;
            int parsedSize = 0;
            const int consumed = av_parser_parse2(parser_.get(), parserContext_.get(), &parsedData, &parsedSize, data, size, pts, dts, pos);
            if (consumed < 0)
                return {};
            data += consumed;
            size -= consumed;
            if (std::string change = shapeChange(); !change.empty())
                return change;
            if (consumed == 0) // parser wants more than this packet holds
                break;
        } while (size > 0);
        return {};
    }

    std::string shapeChange() const {
        if (parser_->width > 0 && parser_->height > 0 && (parser_->width != declaredWidth_ || parser_->height != declaredHeight_))
            return resolutionChangeMessage(declaredWidth_, declaredHeight_, parser_->width, parser_->height);
        const auto parsedFormat = static_cast<AVPixelFormat>(parser_->format);
        if (parsedFormat != AV_PIX_FMT_NONE && declaredFormat_ != AV_PIX_FMT_NONE && withoutFullRangeAlias(parsedFormat) != withoutFullRangeAlias(declaredFormat_))
            return pixelFormatChangeMessage(declaredFormat_, parsedFormat);
        return {};
    }

    int declaredWidth_ = 0;
    int declaredHeight_ = 0;
    AVPixelFormat declaredFormat_ = AV_PIX_FMT_NONE;
    std::unique_ptr<AVCodecParserContext, ParserDeleter> parser_;
    AVCodecContextPtr parserContext_;
};

// pts and duration of decoded frame, falls back to nominal frame rate
FrameTiming frameTiming(VideoSession* s, const AVFrame* frame) {
    const int64_t duration = frame->duration > 0 ? frame->duration : s->nominalFrameDuration;
    const int64_t sourcePts = framePts(frame);
    const int64_t pts = sourcePts != AV_NOPTS_VALUE ? sourcePts : s->nextFramePts;
    s->nextFramePts = pts + std::max<int64_t>(1, duration);
    return {pts, duration};
}

// mov family times sample by next decode timestamp, only last packet stores duration
bool durationsFollowDecodeOrder(const AVOutputFormat* format) {
    constexpr std::string_view movFamily[] = {"mov", "mp4", "3gp", "3g2", "psp", "ipod", "ismv", "f4v"};
    const std::string_view name = format->name != nullptr ? format->name : "";
    return std::ranges::find(movFamily, name) != std::end(movFamily);
}

// forward item to encode thread, bailing out if encode thread failed
void pushToEncoder(EncodeQueue& queue, EncodeQueue::Item item) { checkError(!queue.push(std::move(item)), "Encode thread stopped, aborting the decode loop"); }

AVCodecContextPtr openSoftwareDecoder(const AVCodecParameters* inputCodecParams, AVRational pktTimebase) {
    const AVCodec* inputDecoder = avcodec_find_decoder(inputCodecParams->codec_id);
    if (!inputDecoder)
        return nullptr;
    AVCodecContextPtr ctx(avcodec_alloc_context3(inputDecoder));
    if (!ctx)
        return nullptr;
    if (avcodec_parameters_to_context(ctx.get(), inputCodecParams) < 0)
        return nullptr;
    ctx->thread_count = 0;
    ctx->pkt_timebase = pktTimebase; // stream timebase silences invalid pkt_timebase warnings
    if (avcodec_open2(ctx.get(), inputDecoder, nullptr) < 0)
        return nullptr;
    const auto* descriptor = av_pix_fmt_desc_get(ctx->pix_fmt);
    checkError(!descriptor || (descriptor->flags & (AV_PIX_FMT_FLAG_HWACCEL | AV_PIX_FMT_FLAG_RGB | AV_PIX_FMT_FLAG_PAL | AV_PIX_FMT_FLAG_BITSTREAM)) || descriptor->comp[0].depth > 10,
        "Video input requires a YUV or grayscale format with at most 10 bits per sample");
    return ctx;
}

// HDR peak luminance in units of 100 nits, static HDR10 metadata not present on NVDEC
// clamp to at least 100 nits to keep Mobius coefficients finite
float getHdrPeak(const VideoSession* s) {
    const AVCodecParameters* params = s->videoStream->codecpar;
    const auto sideData = [&](const AVPacketSideDataType type, const size_t size) -> const uint8_t* {
        const auto* data = av_packet_side_data_get(params->coded_side_data, params->nb_coded_side_data, type);
        return data && data->size >= size ? data->data : nullptr;
    };
    float peak = 10.0f;
    if (const auto* light = reinterpret_cast<const AVContentLightMetadata*>(sideData(AV_PKT_DATA_CONTENT_LIGHT_LEVEL, sizeof(AVContentLightMetadata))); light && light->MaxCLL > 0)
        peak = static_cast<float>(light->MaxCLL) / 100.0f;
    else if (const auto* display = reinterpret_cast<const AVMasteringDisplayMetadata*>(sideData(AV_PKT_DATA_MASTERING_DISPLAY_METADATA, sizeof(AVMasteringDisplayMetadata)));
        display && display->has_luminance != 0 && display->max_luminance.num > 0 && display->max_luminance.den > 0)
        peak = static_cast<float>(av_q2d(display->max_luminance) / 100.0);
    return std::max(1.0f, peak);
}

#if defined(_USE_GPU_)
bool configureFilterGraph(VideoSession* session, const AVCodecContext* decoder, const AVFrame* firstFrame);

MobiusParams hdrMobius(const VideoSession* s) { return MobiusParams::fromHdrPeak(getHdrPeak(s)); }

// probe through separate demuxer, rejected hardware path leaves input untouched
AVFramePtr firstHardwareFrame(AVCodecContext* decoder, const VideoSession* session) {
    AVFormatContext* rawInput = nullptr;
    const int opened = avformat_open_input(&rawInput, session->settings.videoFile.c_str(), nullptr, nullptr);
    AVFormatContextPtr input(rawInput);
    if (opened < 0 || avformat_find_stream_info(input.get(), nullptr) < 0)
        return nullptr;
    AVPacketPtr packet(av_packet_alloc());
    AVFramePtr frame(av_frame_alloc());
    if (!packet || !frame)
        return nullptr;
    bool draining = false;
    while (true) {
        const int received = avcodec_receive_frame(decoder, frame.get());
        if (received == 0)
            return frame;
        if (received != AVERROR(EAGAIN) || draining)
            return nullptr;
        int read;
        do {
            av_packet_unref(packet.get());
            read = av_read_frame(input.get(), packet.get());
        } while (read >= 0 && packet->stream_index != session->videoStreamIndex);
        if (read < 0 && read != AVERROR_EOF)
            return nullptr;
        draining = read == AVERROR_EOF;
        if (avcodec_send_packet(decoder, draining ? nullptr : packet.get()) < 0)
            return nullptr;
    }
}

bool gpuTonemapUnsupported(const AVCodecParameters* codecParams) {
    const bool hdr = codecParams->color_trc == AVCOL_TRC_SMPTE2084 || codecParams->color_trc == AVCOL_TRC_ARIB_STD_B67;
    const bool bt2020 = codecParams->color_space == AVCOL_SPC_BT2020_NCL || codecParams->color_space == AVCOL_SPC_UNSPECIFIED;
    const bool bt2020Primaries = codecParams->color_primaries == AVCOL_PRI_BT2020 || codecParams->color_primaries == AVCOL_PRI_UNSPECIFIED;
    return hdr && (codecParams->color_trc != AVCOL_TRC_SMPTE2084 || !bt2020 || !bt2020Primaries || codecParams->color_range == AVCOL_RANGE_JPEG);
}

// CPU-decoded BT.2020 PQ 4:2:0 10-bit uploaded and tonemapped on GPU
bool gpuTonemapsCpuFrames(const VideoSession* s, const AVCodecContext* decoder) {
    const auto* params = s->videoStream->codecpar;
    return isHDR(decoder) && !gpuTonemapUnsupported(params) && decoder->pix_fmt == AV_PIX_FMT_YUV420P10LE && decoder->color_range != AVCOL_RANGE_JPEG && (params->width & 1) == 0 &&
           (params->height & 1) == 0;
}

// 4:2:0 chroma siting relative to luma, unspecified defaults to left
ChromaSiting chromaSiting(const AVCodecContext* decoder, const AVFrame* frame) {
    switch (frame->chroma_location != AVCHROMA_LOC_UNSPECIFIED ? frame->chroma_location : decoder->chroma_sample_location) {
    case AVCHROMA_LOC_CENTER: return {0.5f, 0.5f};
    case AVCHROMA_LOC_TOPLEFT: return {0.0f, 0.0f};
    case AVCHROMA_LOC_TOP: return {0.5f, 0.0f};
    case AVCHROMA_LOC_BOTTOMLEFT: return {0.0f, 1.0f};
    case AVCHROMA_LOC_BOTTOM: return {0.5f, 1.0f};
    default: return {0.0f, 0.5f};
    }
}

// NVDEC and QSV accept these H.264 profiles yet return incorrect pixels
bool hardwareMisdecodesH264Profile(const AVCodecParameters* codecParams) {
    if (codecParams->codec_id != AV_CODEC_ID_H264)
        return false;
    const int profile = codecParams->profile & ~AV_PROFILE_H264_INTRA;
    return profile == AV_PROFILE_H264_HIGH_444 || profile == AV_PROFILE_H264_HIGH_444_PREDICTIVE || profile == AV_PROFILE_H264_CAVLC_444;
}

#endif

#if defined(_USE_CUDA_)
bool nvdecUnsupportedFormat(const AVCodecParameters* codecParams) {
    const auto* format = av_pix_fmt_desc_get(static_cast<AVPixelFormat>(codecParams->format));
    if (!format)
        return false; // let decoder inspect bitstream when demuxer has no format yet
    const bool supportedChroma = (format->log2_chroma_w == 1 && format->log2_chroma_h <= 1) || (format->log2_chroma_w == 0 && format->log2_chroma_h == 0);
    const bool hdr = codecParams->color_trc == AVCOL_TRC_SMPTE2084 || codecParams->color_trc == AVCOL_TRC_ARIB_STD_B67;
    return (format->flags & (AV_PIX_FMT_FLAG_RGB | AV_PIX_FMT_FLAG_HWACCEL | AV_PIX_FMT_FLAG_PAL | AV_PIX_FMT_FLAG_BITSTREAM)) || format->nb_components < 3 || !supportedChroma ||
           format->comp[0].depth > 10 || (hdr && (format->log2_chroma_w != 1 || format->log2_chroma_h != 1));
}

bool probeCudaConversion(AVCodecContext* decoder, VideoSession* session) {
    auto frame = firstHardwareFrame(decoder, session);
    if (!frame || frame->format != AV_PIX_FMT_CUDA || !frame->hw_frames_ctx || frame->width != decoder->width || frame->height != decoder->height)
        return false;
    if (isHDR(decoder) || frame->color_trc == AVCOL_TRC_SMPTE2084 || frame->color_trc == AVCOL_TRC_ARIB_STD_B67)
        return false; // SDR probe selected, newly discovered HDR needs software path
    const auto nativeFormat = decodedPixelFormat(frame.get());
    if (!isPixelFormatSupported(supportedHwFormats, nativeFormat))
        return false;
    // encoder picks output format later, check conversions to 8 bits and 4:2:0
    const AVPixelFormat targets[] = {planar8Bit(nativeFormat), AV_PIX_FMT_YUV420P};
    for (const auto target : std::span(targets, targets[0] == AV_PIX_FMT_YUV420P ? 1 : 2)) {
        // separate graph, production graph starts with no probe timestamps
        VideoSession probe;
        probe.videoStream = session->videoStream;
        probe.useHwDecoder = true;
        probe.decodedFormat = nativeFormat;
        probe.outputFormat = target;
        if (!configureFilterGraph(&probe, decoder, frame.get()))
            continue;
        AVFramePtr converted(av_frame_alloc());
        if (!converted || av_buffersrc_add_frame_flags(probe.buffersrcCtx, frame.get(), AV_BUFFERSRC_FLAG_KEEP_REF) < 0 || av_buffersink_get_frame(probe.buffersinkCtx, converted.get()) < 0)
            return false;
        if (converted->format != AV_PIX_FMT_CUDA || decodedPixelFormat(converted.get()) != cudaSurfaceFormat(target))
            return false;
    }
    CUDA_CHECK(cudaStreamSynchronize(CudaStreamManager::getInstance().getComputeStream()));
    session->decodedFormat = nativeFormat;
    avcodec_flush_buffers(decoder);
    return true;
}

// FFmpeg CUDA device on selected GPU primary context and compute stream
AVBufferRefPtr createCudaDevice() {
    AVBufferRefPtr device(av_hwdevice_ctx_alloc(AV_HWDEVICE_TYPE_CUDA));
    if (!device)
        return nullptr;
    auto* hwCtx = reinterpret_cast<AVHWDeviceContext*>(device->data);
    auto* cudaCtx = reinterpret_cast<AVCUDADeviceContext*>(hwCtx->hwctx);
    int runtimeDevice = 0;
    CUDA_CHECK(cudaGetDevice(&runtimeDevice));
    CUdevice cuDev = 0;
    if (cuDeviceGet(&cuDev, runtimeDevice) != CUDA_SUCCESS)
        return nullptr;
    CUcontext primaryCtx = nullptr;
    if (cuDevicePrimaryCtxRetain(&primaryCtx, cuDev) != CUDA_SUCCESS)
        return nullptr;
    cudaCtx->cuda_ctx = primaryCtx;
    cudaCtx->stream = CudaStreamManager::getInstance().getComputeStream();
    hwCtx->user_opaque = reinterpret_cast<void*>(static_cast<intptr_t>(cuDev));
    hwCtx->free = [](AVHWDeviceContext* c) {
        const CUdevice d = static_cast<CUdevice>(reinterpret_cast<intptr_t>(c->user_opaque));
        cuDevicePrimaryCtxRelease(d);
    };
    if (av_hwdevice_ctx_init(device.get()) < 0 || cuCtxSetCurrent(primaryCtx) != CUDA_SUCCESS)
        return nullptr;
    return device;
}

const char* cuvidNameFor(const AVCodecID codecId) {
    switch (codecId) {
    case AV_CODEC_ID_H264: return "h264_cuvid";
    case AV_CODEC_ID_HEVC: return "hevc_cuvid";
    case AV_CODEC_ID_AV1: return "av1_cuvid";
    case AV_CODEC_ID_VP9: return "vp9_cuvid";
    case AV_CODEC_ID_VP8: return "vp8_cuvid";
    case AV_CODEC_ID_MPEG1VIDEO: return "mpeg1_cuvid";
    case AV_CODEC_ID_MPEG2VIDEO: return "mpeg2_cuvid";
    case AV_CODEC_ID_MPEG4: return "mpeg4_cuvid";
    case AV_CODEC_ID_VC1: return "vc1_cuvid";
    default: return nullptr;
    }
}

// open CUDA decoder, fall back to software on failure
AVCodecContextPtr openDecoderHWAccel(VideoSession* session) {
    const auto* inputCodecParams = session->videoStream->codecpar;
    const auto pktTimebase = session->videoStream->time_base;
    session->useHwDecoder = false;
    const char* decoderName = cuvidNameFor(inputCodecParams->codec_id);
    if (!decoderName) {
        cout << info(std::format("NVDEC cannot decode codec '{}', falling back to software decoder (CPU).\n", avcodec_get_name(inputCodecParams->codec_id)));
        return openSoftwareDecoder(inputCodecParams, pktTimebase);
    }
    if (hardwareMisdecodesH264Profile(inputCodecParams)) {
        const char* profileName = avcodec_profile_name(inputCodecParams->codec_id, inputCodecParams->profile);
        cout << info(std::format("NVDEC decodes H.264 profile '{}' incorrectly, falling back to software decoder (CPU).\n", profileName ? profileName : "?"));
        return openSoftwareDecoder(inputCodecParams, pktTimebase);
    }
    if (nvdecUnsupportedFormat(inputCodecParams)) {
        cout << info("NVDEC cannot use this source chroma format or bit depth; falling back to software decoder (CPU).\n");
        return openSoftwareDecoder(inputCodecParams, pktTimebase);
    }
    if (gpuTonemapUnsupported(inputCodecParams)) {
        const char* transfer = av_color_transfer_name(inputCodecParams->color_trc);
        const char* matrix = av_color_space_name(inputCodecParams->color_space);
        cout << info(std::format("NVDEC HDR tonemapping supports limited range BT.2020 PQ only (input: {}, {}, {} range), falling back to software decoder (CPU).\n", transfer ? transfer : "?",
            matrix ? matrix : "?", inputCodecParams->color_range == AVCOL_RANGE_JPEG ? "full" : "limited"));
        return openSoftwareDecoder(inputCodecParams, pktTimebase);
    }
    const AVCodec* inputDecoder = avcodec_find_decoder_by_name(decoderName);
    if (!inputDecoder)
        return openSoftwareDecoder(inputCodecParams, pktTimebase);
    const AVBufferRefPtr hw_device_ctx = createCudaDevice();
    if (!hw_device_ctx)
        return openSoftwareDecoder(inputCodecParams, pktTimebase);
    const auto tryOpen = [&](const bool deinterlace) -> AVCodecContextPtr {
        AVCodecContextPtr ctx(avcodec_alloc_context3(inputDecoder));
        if (!ctx || avcodec_parameters_to_context(ctx.get(), inputCodecParams) < 0)
            return nullptr;
        ctx->hw_device_ctx = av_buffer_ref(hw_device_ctx.get());
        if (!ctx->hw_device_ctx)
            return nullptr;
        ctx->get_format = [](AVCodecContext*, const AVPixelFormat* formats) {
            for (; *formats != AV_PIX_FMT_NONE; ++formats)
                if (*formats == AV_PIX_FMT_CUDA)
                    return AV_PIX_FMT_CUDA;
            return AV_PIX_FMT_NONE;
        };
        ctx->pkt_timebase = pktTimebase;
        AVDictionary* decOpts = nullptr;
        if (deinterlace) {
            av_dict_set(&decOpts, "deint", "adaptive", 0);
            av_dict_set(&decOpts, "drop_second_field", "1", 0);
        }
        const int status = avcodec_open2(ctx.get(), inputDecoder, &decOpts);
        av_dict_free(&decOpts);
        if (status < 0)
            return nullptr;
        return ctx;
    };
    // failed open can leave private decoder state, retry in fresh context
    auto ctx = tryOpen(true);
    if (!ctx)
        ctx = tryOpen(false);
    if (!ctx) {
        cout << info(std::format("NVDEC decoder '{}' could not open input, falling back to software decoder (CPU).\n", decoderName));
        return openSoftwareDecoder(inputCodecParams, pktTimebase);
    }
    if (!isPixelFormatSupported(supportedHwFormats, ctx->sw_pix_fmt)) {
        cout << info(std::format("NVDEC output format '{}' unsupported, falling back to software decoder (CPU).\n", av_get_pix_fmt_name(ctx->sw_pix_fmt) ? av_get_pix_fmt_name(ctx->sw_pix_fmt) : "?"));
        return openSoftwareDecoder(inputCodecParams, pktTimebase);
    }
    if (isHDR(ctx.get()) && ctx->sw_pix_fmt != AV_PIX_FMT_P010LE && ctx->sw_pix_fmt != AV_PIX_FMT_P016LE) {
        cout << info("This HDR surface requires software tone mapping, falling back to software decoder (CPU).\n");
        return openSoftwareDecoder(inputCodecParams, pktTimebase);
    }
    // NV12 consumed directly, verify decode and GPU conversion for other surfaces
    if (!isHDR(ctx.get()) && ctx->sw_pix_fmt != AV_PIX_FMT_NV12) {
        try {
            if (!probeCudaConversion(ctx.get(), session)) {
                cout << info("NVDEC could not decode and convert the first frame, falling back to software decoder (CPU).\n");
                return openSoftwareDecoder(inputCodecParams, pktTimebase);
            }
        } catch (const std::exception& error) {
            cout << info(std::string("NVDEC conversion setup failed: ") + error.what() + ", falling back to software decoder (CPU).\n");
            return openSoftwareDecoder(inputCodecParams, pktTimebase);
        }
    } else {
        session->decodedFormat = ctx->sw_pix_fmt;
    }
    session->useHwDecoder = true;
    cout << info(std::format("Using NVDEC decoding ({}).\n", pixelFormatName(session->decodedFormat)));
    return ctx;
}
#endif

#if defined(_USE_OPENCL_)
// D3D11VA returns frame on submission, holding frames lets decode run ahead
constexpr int d3d11DecodeLookahead = 2;

const AVCodec* findOpenCLVideoDecoder(const AVCodecID codecId, const AVHWDeviceType deviceType) {
    const auto supportsDevice = [deviceType](const AVCodec* codec) {
        if (!codec)
            return false;
        for (int index = 0; const auto* config = avcodec_get_hw_config(codec, index); ++index)
            if (config->device_type == deviceType && (config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX))
                return true;
        return false;
    };
    const auto* preferred = avcodec_find_decoder(codecId);
    if (supportsDevice(preferred))
        return preferred;
    void* iterator = nullptr;
    while (const auto* candidate = av_codec_iterate(&iterator))
        if (candidate->id == codecId && av_codec_is_decoder(candidate) && supportsDevice(candidate))
            return candidate;
    return nullptr;
}

// mapping retains complete QSV frame, not only COM texture reference
void mapOpenCLVideoFrame(AVFramePtr& frame, VideoSession* session) {
    if (frame->format != AV_PIX_FMT_QSV)
        return;
    AVFramePtr mapped(av_frame_alloc());
    checkError(!mapped, "Failed to allocate a mapped QSV frame");
    mapped->format = AV_PIX_FMT_D3D11;
    checkAv(av_hwframe_map(mapped.get(), frame.get(), AV_HWFRAME_MAP_READ | AV_HWFRAME_MAP_DIRECT), "Map decoded QSV frame to D3D11");
    checkAv(av_frame_copy_props(mapped.get(), frame.get()), "Copy QSV decoder frame properties");
    if (!mapped->hw_frames_ctx) {
        checkError(!frame->hw_frames_ctx, "QSV frame has no surface descriptor");
        const auto* sourceFrames = reinterpret_cast<const AVHWFramesContext*>(frame->hw_frames_ctx->data);
        D3D11_TEXTURE2D_DESC texture{};
        reinterpret_cast<ID3D11Texture2D*>(mapped->data[0])->GetDesc(&texture);
        if (!session->openclMappedFrames) {
            AVBufferRefPtr device(OclQueueManager::getInstance().getVideoInterop()->createDevice());
            checkError(!device, "Failed to reference the QSV mapping's D3D11 device");
            session->openclMappedFrames.reset(av_hwframe_ctx_alloc(device.get()));
            checkError(!session->openclMappedFrames, "Failed to allocate QSV mapping metadata");
            auto* frames = reinterpret_cast<AVHWFramesContext*>(session->openclMappedFrames->data);
            frames->format = AV_PIX_FMT_D3D11;
            frames->sw_format = sourceFrames->sw_format;
            frames->width = static_cast<int>(texture.Width);
            frames->height = static_cast<int>(texture.Height);
            frames->initial_pool_size = 0;
            checkAv(av_hwframe_ctx_init(session->openclMappedFrames.get()), "Initialize QSV mapping metadata");
        }
        const auto* frames = reinterpret_cast<const AVHWFramesContext*>(session->openclMappedFrames->data);
        checkError(frames->sw_format != sourceFrames->sw_format || frames->width != static_cast<int>(texture.Width) || frames->height != static_cast<int>(texture.Height),
            "QSV decoder surface layout changed");
        mapped->hw_frames_ctx = av_buffer_ref(session->openclMappedFrames.get());
        checkError(!mapped->hw_frames_ctx, "Failed to retain QSV mapping metadata");
    }
    frame = std::move(mapped);
}

bool probeOpenCLDecoder(AVCodecContext* decoder, VideoSession* session) {
    auto frame = firstHardwareFrame(decoder, session);
    if (!frame)
        return false;
    mapOpenCLVideoFrame(frame, session);
    if (frame->format != AV_PIX_FMT_D3D11 || !frame->hw_frames_ctx)
        return false;
    session->decodedFormat = decodedPixelFormat(frame.get());
    session->isHdr = isHDR(decoder) || frame->color_trc == AVCOL_TRC_SMPTE2084 || frame->color_trc == AVCOL_TRC_ARIB_STD_B67;
    if (session->isHdr) {
        const auto transfer = frame->color_trc == AVCOL_TRC_UNSPECIFIED ? decoder->color_trc : frame->color_trc;
        const auto matrix = frame->colorspace == AVCOL_SPC_UNSPECIFIED ? decoder->colorspace : frame->colorspace;
        const auto range = frame->color_range == AVCOL_RANGE_UNSPECIFIED ? decoder->color_range : frame->color_range;
        const auto primaries = frame->color_primaries == AVCOL_PRI_UNSPECIFIED ? decoder->color_primaries : frame->color_primaries;
        if (session->decodedFormat != AV_PIX_FMT_P010 || transfer != AVCOL_TRC_SMPTE2084 || (matrix != AVCOL_SPC_BT2020_NCL && matrix != AVCOL_SPC_UNSPECIFIED) || range == AVCOL_RANGE_JPEG)
            return false;
        if (primaries != AVCOL_PRI_BT2020 && primaries != AVCOL_PRI_UNSPECIFIED)
            return false;
        session->openclHdrTonemap = std::make_unique<OclHdrTonemap>(frame->width, frame->height, OclQueueManager::getInstance().getQueueRaw());
    }
    auto& manager = OclQueueManager::getInstance();
    session->openclVideoFrame = manager.getVideoInterop()->createFrame(frame.get(), manager.getQueueRaw(), session->isHdr);
    const auto& copied = session->openclVideoFrame->copyFrom(frame.get());
    if (session->isHdr)
        session->openclHdrTonemap->convert(copied, hdrMobius(session), chromaSiting(decoder, frame.get()));
    manager.getQueue().finish();
    av_frame_unref(frame.get());
    avcodec_flush_buffers(decoder);
    return true;
}

AVCodecContextPtr openDecoderOpenCL(VideoSession* session) {
    const auto* parameters = session->videoStream->codecpar;
    const auto resetProbe = [&] {
        session->useHwDecoder = false;
        session->decodeLookahead = 0;
        session->filterGraph.reset();
        session->buffersrcCtx = session->buffersinkCtx = nullptr;
        session->openclVideoFrame.reset();
        session->openclMappedFrames.reset();
        session->openclHdrTonemap.reset();
        session->isHdr = false;
    };
    const auto fallback = [&](const std::string& reason) {
        resetProbe();
        cout << info("Hardware decoding " + reason + ", falling back to software decoder (CPU).\n");
        return openSoftwareDecoder(parameters, session->videoStream->time_base);
    };
    if (hardwareMisdecodesH264Profile(parameters))
        return fallback("cannot correctly decode this H.264 High 4:4:4 profile");
    if (gpuTonemapUnsupported(parameters))
        return fallback("HDR tonemapping supports limited-range BT.2020 PQ only");
    const auto* descriptor = av_pix_fmt_desc_get(static_cast<AVPixelFormat>(parameters->format));
    if (descriptor && ((descriptor->flags & AV_PIX_FMT_FLAG_RGB) || descriptor->comp[0].depth > 10))
        return fallback("cannot use this source format");
    auto& manager = OclQueueManager::getInstance();
    auto contextLock = manager.lockContext();
    const auto interop = manager.getVideoInterop();
    if (!interop)
        return fallback("cannot share buffers with the selected OpenCL GPU");
    std::string reason = "does not support this codec/profile on the selected GPU";
    for (const auto type : {AV_HWDEVICE_TYPE_D3D11VA, AV_HWDEVICE_TYPE_QSV}) {
        // QSV can expose native layouts omitted by FFmpeg D3D11 decoder
        if (type == AV_HWDEVICE_TYPE_QSV && manager.getDevice().getInfo<CL_DEVICE_VENDOR_ID>() != 0x8086)
            continue;
        const auto* codec = findOpenCLVideoDecoder(parameters->codec_id, type);
        if (!codec)
            continue;
        resetProbe();
        try {
            AVCodecContextPtr decoder(avcodec_alloc_context3(codec));
            checkError(!decoder, "Could not allocate the hardware decoder");
            checkAv(avcodec_parameters_to_context(decoder.get(), parameters), "Copy hardware decoder parameters");
            AVBufferRefPtr d3d(interop->createDevice());
            checkError(!d3d, "Could not initialize the hardware device");
            if (type == AV_HWDEVICE_TYPE_QSV)
                checkAv(av_hwdevice_ctx_create_derived(&decoder->hw_device_ctx, type, d3d.get(), 0), "Derive QSV decoder from selected D3D11 adapter");
            else
                decoder->hw_device_ctx = av_buffer_ref(d3d.get());
            decoder->get_format = [](AVCodecContext* context, const AVPixelFormat* formats) {
                const auto* device = reinterpret_cast<const AVHWDeviceContext*>(context->hw_device_ctx->data);
                const auto wanted = device->type == AV_HWDEVICE_TYPE_QSV ? AV_PIX_FMT_QSV : AV_PIX_FMT_D3D11;
                for (; *formats != AV_PIX_FMT_NONE; ++formats)
                    if (*formats == wanted)
                        return wanted;
                return AV_PIX_FMT_NONE;
            };
            decoder->pkt_timebase = session->videoStream->time_base;
            decoder->extra_hw_frames = d3d11DecodeLookahead;
            checkAv(avcodec_open2(decoder.get(), codec, nullptr), "Open hardware decoder");
            session->useHwDecoder = true;
            if (!probeOpenCLDecoder(decoder.get(), session)) {
                reason = "could not decode and copy the first frame on this GPU";
                continue;
            }
            if (type == AV_HWDEVICE_TYPE_QSV) {
                // QSV can retain delayed surfaces across flush, do not reuse probe decoder
                AVBufferRefPtr device(av_buffer_ref(decoder->hw_device_ctx));
                checkError(!device, "Failed to retain the probed QSV device");
                decoder.reset();
                session->openclVideoFrame.reset();
                session->openclMappedFrames.reset();
                decoder.reset(avcodec_alloc_context3(codec));
                checkError(!decoder, "Failed to reopen the QSV decoder after preflight");
                checkAv(avcodec_parameters_to_context(decoder.get(), parameters), "Copy QSV decoder parameters after preflight");
                decoder->hw_device_ctx = device.release();
                decoder->get_format = [](AVCodecContext*, const AVPixelFormat* formats) {
                    for (; *formats != AV_PIX_FMT_NONE; ++formats)
                        if (*formats == AV_PIX_FMT_QSV)
                            return AV_PIX_FMT_QSV;
                    return AV_PIX_FMT_NONE;
                };
                decoder->pkt_timebase = session->videoStream->time_base;
                decoder->extra_hw_frames = d3d11DecodeLookahead;
                checkAv(avcodec_open2(decoder.get(), codec, nullptr), "Reopen QSV decoder after preflight");
            }
            session->decodeLookahead = d3d11DecodeLookahead;
            cout << info(std::format("Using {} decoding ({}) with {}.\n", type == AV_HWDEVICE_TYPE_QSV ? "QSV" : "D3D11VA", pixelFormatName(session->decodedFormat),
                session->isHdr ? "OpenCL Mobius HDR tonemapping" : "GPU conversion and copies into OpenCL"));
            return decoder;
        } catch (const cl::Error& error) { reason = std::string(error.what()) + " (" + std::to_string(error.err()) + ")"; } catch (const std::exception& error) {
            reason = error.what();
        }
    }
    return fallback(reason);
}
#endif

// filter graph for depth reduction, chroma conversion or CPU tonemapping when needed
string getFilterGraphString(const VideoSession* s, const AVCodecContext* decoder) {
#if defined(_USE_OPENCL_)
    if (s->useHwDecoder)
        return "";
#endif
    if (!isHDR(decoder)) {
        // detection reads only the luma plane
        if (s->outputFormat == AV_PIX_FMT_NONE && hasPlanarLuma8(s->decodedFormat))
            return "";
        const AVPixelFormat target = s->outputFormat != AV_PIX_FMT_NONE ? s->outputFormat : planar8Bit(s->decodedFormat);
        if (s->useHwDecoder) {
            const AVPixelFormat surface = cudaSurfaceFormat(target);
            return s->decodedFormat != surface ? "scale_cuda=format=" + pixelFormatName(surface) : "";
        }
        return withoutFullRangeAlias(decoder->pix_fmt) == target ? "" : "format=" + pixelFormatName(target);
    }
    // HDR converted by the GPU kernels (hardware-decoded, or CPU-decoded PQ uploaded): no filter graph needed
    if (s->useHwDecoder || s->isHdr)
        return "";
    // SW decoder + HDR: zscale needs explicit input primaries/matrix
    const char* primaries = av_color_primaries_name(decoder->color_primaries);
    const char* matrix = av_color_space_name(decoder->colorspace);
    // fallback to safe HDR10 defaults if any field is unspecified
    if (!primaries || decoder->color_primaries == AVCOL_PRI_UNSPECIFIED)
        primaries = "bt2020";
    if (!matrix || decoder->colorspace == AVCOL_SPC_UNSPECIFIED)
        matrix = "bt2020nc";
    // tonemap uses matrix luma, tag BT.2020 matrix before tonemapping to avoid summing channels
    // The peak is explicit: zscale drops the frame metadata
    return std::format("zscale=primaries={}:transfer=linear:matrix={}:npl=100,setparams=colorspace={},tonemap=mobius:peak={},setparams=colorspace=gbr,"
                       "zscale=transfer=bt709:primaries=bt709:matrix=bt709,format=yuv420p",
        primaries, matrix, matrix, getHdrPeak(s));
}

void filterFrame(AVFramePtr& frame, AVFramePtr& filteredFrame, const VideoSession* s) {
    av_frame_unref(filteredFrame.get());
    int ret = av_buffersrc_add_frame_flags(s->buffersrcCtx, frame.get(), AV_BUFFERSRC_FLAG_KEEP_REF);
    checkError(ret < 0, "Failed to add frame to filter graph");
    ret = av_buffersink_get_frame(s->buffersinkCtx, filteredFrame.get());
    checkError(ret == AVERROR(EAGAIN) || ret == AVERROR_EOF, "Filter graph buffered the frame. 1-to-1 filtering strictly required!");
    checkError(ret < 0, "Failed to get filtered frame: " + std::to_string(ret));
    av_frame_unref(frame.get());
    av_frame_move_ref(frame.get(), filteredFrame.get());
}

#if defined(_USE_GPU_)
// one GPU buffer for the whole session because a new buffer per frame would re-capture the CUDA graphs every frame
ImageBuffer& lumaBuffer(VideoSession* s) {
    if (s->inputFrame.empty()) {
        const auto [height, width] = s->videoDims();
#if defined(_USE_CUDA_)
        s->inputFrame = CudaArray<float>(height, width, CudaStreamManager::getInstance().getComputeStream());
#else
        s->inputFrame = OclArray<float>(height, width, OclQueueManager::getInstance().getQueueRaw());
#endif
    }
    return s->inputFrame;
}
#endif

// one copy for a plane without row padding on either side (large copies are measurably faster than one per row)
void copyPlane(uint8_t* dst, const int dstPitch, const uint8_t* src, const int srcPitch, const int width, const int height) {
    if (dstPitch == width && srcPitch == width)
        std::memcpy(dst, src, static_cast<size_t>(width) * height);
    else
        av_image_copy_plane(dst, dstPitch, src, srcPitch, width, height);
}

#if defined(_USE_OPENCL_)
// packed plane into pitched host plane, single read then spread rows in place
// No rectangle reads: Intel's driver CRASHES reading rectangles of D3D11-shared buffers (a CPU copy from the GPU address)
void readPlane(cl::CommandQueue& queue, const cl::Buffer& source, const size_t offset, const int width, const int height, uint8_t* dst, const int pitch, const bool blocking) {
    queue.enqueueReadBuffer(source, blocking || pitch != width, offset, static_cast<size_t>(width) * height, dst);
    // backwards unpack ensures destination row is free before overwrite
    if (pitch != width)
        for (int row = height - 1; row > 0; --row)
            std::memmove(dst + static_cast<size_t>(row) * pitch, dst + static_cast<size_t>(row) * width, width);
}
#endif

// a software encoder frame in the output format, its buffer comes from the session pool: reused instead of allocated (and page faulted) per frame,
// pinned on the GPU builds so the readbacks reach it directly. GPU encoding and detection never read frames back
AVFramePtr encoderFrame(VideoSession* s, const FrameTiming& timing) {
    constexpr int alignment = 64;
    const PlanarLayout layout = outputLayout(s);
    if (!s->encoderFramePool) {
        const int bytes = av_image_get_buffer_size(s->outputFormat, layout.width, layout.height, alignment);
        checkAv(bytes, "Invalid encoder frame size");
#if defined(_USE_GPU_)
        s->encoderFramePool.reset(av_buffer_pool_init(bytes, [](const size_t size) -> AVBufferRef* {
            try {
                auto* memory = new HostMemory<uint8_t>(size);
                AVBufferRef* buffer = av_buffer_create(memory->get(), size, [](void* opaque, uint8_t*) { delete static_cast<HostMemory<uint8_t>*>(opaque); }, memory, 0);
                if (!buffer)
                    delete memory;
                return buffer;
            } catch (...) {
                return nullptr; // reported as a failed frame allocation
            }
        }));
#else
        s->encoderFramePool.reset(av_buffer_pool_init(bytes, nullptr));
#endif
        checkError(!s->encoderFramePool, "Failed to allocate the encoder frame pool");
    }
    AVFramePtr encFrame(av_frame_alloc());
    checkError(!encFrame, "Failed to allocate encoder AVFrame");
    encFrame->buf[0] = av_buffer_pool_get(s->encoderFramePool.get());
    checkError(!encFrame->buf[0], "Failed to allocate encoder frame buffer");
    checkAv(av_image_fill_arrays(encFrame->data, encFrame->linesize, encFrame->buf[0]->data, s->outputFormat, layout.width, layout.height, alignment), "Failed to lay out the encoder frame");
    encFrame->format = s->outputFormat;
    encFrame->width = layout.width;
    encFrame->height = layout.height;
    encFrame->pts = timing.pts;
    encFrame->duration = timing.duration;
    return encFrame;
}

// upload CPU uint8 Y plane into GPU or Eigen input buffer
void loadInputFrame(VideoSession* s, const uint8_t* hostPtr, const int srcPitch) {
    const auto [height, width] = s->videoDims();
#if defined(_USE_CUDA_)
    const auto stream = CudaStreamManager::getInstance().getComputeStream();
    // upload host uint8 row-major to GPU, then use pitchedToFloat kernel (stride=width) to convert to column-major float
    CudaArray<uint8_t> hostGpu(height, width, stream);
    if (srcPitch == width)
        CUDA_CHECK(cudaMemcpyAsync(hostGpu.data(), hostPtr, static_cast<size_t>(width) * height, cudaMemcpyHostToDevice, stream));
    else // strided upload
        CUDA_CHECK(cudaMemcpy2DAsync(hostGpu.data(), width, hostPtr, srcPitch, width, height, cudaMemcpyHostToDevice, stream));
    cuda_utils::launchPitchedToFloatKernel(hostGpu.data(), lumaBuffer(s).data(), width, height, width, stream);
#elif defined(_USE_OPENCL_)
    auto& mgr = OclQueueManager::getInstance();
    auto queue = mgr.getQueueRaw();
    OclArray<uint8_t> hostGpu(height, width, queue);
    cl_int uploadStatus;
    if (srcPitch == width) {
        uploadStatus = clEnqueueWriteBuffer(queue, hostGpu.data(), CL_TRUE, 0, hostGpu.bytes(), hostPtr, 0, nullptr, nullptr);
    } else {
        // strided upload
        const size_t origin[3] = {0, 0, 0};
        const size_t region[3] = {static_cast<size_t>(width), static_cast<size_t>(height), 1};
        uploadStatus = clEnqueueWriteBufferRect(queue, hostGpu.data(), CL_TRUE, origin, origin, region, static_cast<size_t>(width), 0, static_cast<size_t>(srcPitch), 0, hostPtr, 0, nullptr, nullptr);
    }
    if (uploadStatus != CL_SUCCESS)
        throw std::runtime_error("Failed to upload video frame to OpenCL buffer: " + std::to_string(uploadStatus));
    cl_utils::launchPitchedToFloat(hostGpu.clBuffer(), lumaBuffer(s).clBuffer(), width, height, width, mgr.getQueue());
#else
    // row-major 8-bit Y plane (row pitch srcPitch) -> column-major float luma in the session buffer, transposed
    auto& luma = s->inputFrame.getGray();
    luma.resize(height, width);
    const int lumaRows = height;
    const int lumaCols = width;
    constexpr int tile = 64;
    const int rowTiles = (lumaRows + tile - 1) / tile;
    const int colTiles = (lumaCols + tile - 1) / tile;
#pragma omp parallel for collapse(2) schedule(static)
    for (int rowTile = 0; rowTile < rowTiles; rowTile++) {
        for (int colTile = 0; colTile < colTiles; colTile++) {
            const int row = rowTile * tile;
            const int col = colTile * tile;
            const int rows = std::min(tile, lumaRows - row);
            const int cols = std::min(tile, lumaCols - col);
            const Map<const Gray8Buffer, 0, OuterStride<>> source(hostPtr + (static_cast<size_t>(row) * srcPitch) + col, cols, rows, OuterStride<>(srcPitch));
            luma.block(row, col, rows, cols) = source.transpose().template cast<float>();
        }
    }
#endif
}

// watermark the frame and fill the Y plane of the encoder frame, the embedding writes the row-major Y plane directly (no transposition pass)
void embedAndFillYPlane(VideoSession* s, const ImageBuffer& buffer, AVFrame* encFrame) {
    s->watermarkObj->makeWatermark(buffer, s->watermarkedFrame, WatermarkBase::Layout::RowMajor);
    const auto [height, width] = s->videoDims();
    // read back straight into the pinned encoder frame
#if defined(_USE_CUDA_)
    s->watermarkedFrame.toHostPitched(encFrame->data[0], width, encFrame->linesize[0]);
#elif defined(_USE_OPENCL_)
    readPlane(OclQueueManager::getInstance().getQueue(), s->watermarkedFrame.clBuffer(), 0, width, height, encFrame->data[0], encFrame->linesize[0], true);
#else
    // the row-major Y plane is the transposed array copied with FFmpeg's row pitch
    copyPlane(encFrame->data[0], encFrame->linesize[0], s->watermarkedFrame.getGray().data(), width, width, height);
#endif
}

void fillYPlane(const AVFrame* frame, VideoSession* s, AVFrame* encFrame) {
    loadInputFrame(s, frame->data[0], frame->linesize[0]);
    embedAndFillYPlane(s, s->inputFrame, encFrame);
}

// writes the held packet to the output container
void writeHeldPacket(VideoSession* s) {
    if (!s->holdingPacket)
        return;
    s->holdingPacket = false;
    const int writeStatus = av_interleaved_write_frame(s->outputFormatCtx.get(), s->heldPacket.get());
    av_packet_unref(s->heldPacket.get());
    checkAv(writeStatus, "Failed to write encoded video packet");
}

// reads the pixel format from the first encoded packet: an encoder option can change the depth or chroma of the bitstream (NVENC -highbitdepth),
// the frames were processed in the output format either way
void reportEncodedFormat(const VideoSession* s, const AVPacket* packet) {
    const AVCodecContext* encoder = s->outputEncoderCtx.get();
    const auto closeParser = [](AVCodecParserContext* parser) { av_parser_close(parser); };
    std::unique_ptr<AVCodecParserContext, decltype(closeParser)> parser(av_parser_init(encoder->codec_id), closeParser);
    // the parser reads the sequence header from the extradata of a global header
    AVCodecContextPtr context(avcodec_alloc_context3(nullptr));
    std::unique_ptr<AVCodecParameters, detail::AVDeleter<avcodec_parameters_free>> parameters(avcodec_parameters_alloc());
    if (!parser || !context || !parameters || avcodec_parameters_from_context(parameters.get(), encoder) < 0 || avcodec_parameters_to_context(context.get(), parameters.get()) < 0)
        return;
    parser->flags |= PARSER_FLAG_COMPLETE_FRAMES;
    uint8_t* data = nullptr;
    int size = 0;
    av_parser_parse2(parser.get(), context.get(), &data, &size, packet->data, packet->size, packet->pts, packet->dts, -1);
    const auto encoded = static_cast<AVPixelFormat>(parser->format);
    if (encoded != AV_PIX_FMT_NONE && withoutFullRangeAlias(encoded) != s->outputFormat)
        cout << info(std::format("The encoded video is {} while the watermarked frames are {}: the encoder options changed its format, without adding precision.\n", pixelFormatName(encoded),
            pixelFormatName(s->outputFormat)));
}

// pull all ready packets from the encoder and write them to the output container (non-blocking), the newest one is held back
void drainEncoderPackets(VideoSession* s) {
    const AVPacketPtr pkt(av_packet_alloc());
    checkError(!pkt, "Failed to allocate encoder output packet");
    const AVStream* outVideoStream = s->outputFormatCtx->streams[s->outputVideoStreamIndex];
    while (true) {
        const int receiveRet = avcodec_receive_packet(s->outputEncoderCtx.get(), pkt.get());
        if (receiveRet == AVERROR(EAGAIN) || receiveRet == AVERROR_EOF)
            return;
        checkAv(receiveRet, "Failed to receive encoded video packet");
        if (s->encodedPackets == 0)
            reportEncodedFormat(s, pkt.get());

        s->frameDurations.restore(pkt.get());
        av_packet_rescale_ts(pkt.get(), s->outputEncoderCtx->time_base, outVideoStream->time_base);
        pkt->stream_index = s->outputVideoStreamIndex;
        pkt->pos = -1;
        if (enforceMonotonicDts(pkt.get(), s->lastWrittenVideoDts) && !s->reportedDtsRepair) {
            cout << info("Encoder emitted invalid or non-monotonic video DTS, repairing output timestamps.\n");
            s->reportedDtsRepair = true;
        }
        if (s->encodedPackets == 0 && pkt->pts != AV_NOPTS_VALUE && pkt->dts != AV_NOPTS_VALUE)
            s->firstReorderDelay = pkt->pts - pkt->dts;
        if (pkt->pts != AV_NOPTS_VALUE && pkt->duration > 0) {
            const int64_t packetEnd = pkt->pts + pkt->duration;
            s->presentationEnd = s->presentationEnd == AV_NOPTS_VALUE ? packetEnd : std::max(s->presentationEnd, packetEnd);
        }
        writeHeldPacket(s);
        av_packet_move_ref(s->heldPacket.get(), pkt.get());
        s->holdingPacket = true;
        s->encodedPackets++;
    }
}

// closes the output file and reports write errors, also the ones the final flush hits (for example, a full disk)
void closeOutputFile(VideoSession* s) {
    AVFormatContext* output = s->outputFormatCtx.get();
    if (output->pb == nullptr || (output->oformat->flags & AVFMT_NOFILE))
        return;
    avio_flush(output->pb);
    const int writeStatus = output->pb->error;
    const int closeStatus = avio_closep(&output->pb);
    checkAv(writeStatus < 0 ? writeStatus : closeStatus, "Failed to write the output file");
}

// encode thread: pops items, encodes frames OR writes passthrough packets, it is the
// only writer to outputFormatCtx so audio/subtitle remux and video encode never have race conditions
// does NOT flush the encoder on exit, flushAndFinalize handles that after join()
void encodeWorker(VideoSession* s, EncodeQueue& queue, std::exception_ptr& encErr) noexcept {
    try {
        while (auto item = queue.pop()) {
            if (std::holds_alternative<AVFramePtr>(*item)) {
                AVFramePtr frame = std::move(std::get<AVFramePtr>(*item));
#if defined(_USE_OPENCL_)
                // the GPU copy into an encoder surface completes while later frames are processed
                if (s->openclVideoOutput) {
                    s->openclVideoOutput->waitUntilCopied(frame.get());
                    frame->opaque = nullptr;
                }
#endif
                s->frameDurations.remember(frame.get());
                int ret = avcodec_send_frame(s->outputEncoderCtx.get(), frame.get());
                while (ret == AVERROR(EAGAIN)) {
                    drainEncoderPackets(s);
                    ret = avcodec_send_frame(s->outputEncoderCtx.get(), frame.get());
                }
                checkAv(ret, "Encode thread: avcodec_send_frame failed");
                drainEncoderPackets(s);
            } else {
                AVPacketPtr pkt = std::move(std::get<AVPacketPtr>(*item));
                checkAv(av_interleaved_write_frame(s->outputFormatCtx.get(), pkt.get()), "Encode thread: passthrough packet write failed");
            }
        }
    } catch (...) {
        encErr = std::current_exception();
        queue.abort(); // unblock main thread if it is waiting on a full queue
    }
}

#if defined(_USE_CUDA_)
// NVENC from CUDA frames, copy Y plane and chroma into hw frame and queue to encode thread
// The chroma is in the surface layout: interleaved UV rows (NV12, NV16) in chroma[0], or the U and V planes (YUV444P)
void encodeFrameGPU(
    VideoSession* s, const uint8_t* ySrc, const int yPitch, const std::array<const uint8_t*, 2>& chroma, const std::array<int, 2>& chromaPitch, const FrameTiming& timing, EncodeQueue& queue) {
    const PlanarLayout layout = outputLayout(s);
    const auto stream = CudaStreamManager::getInstance().getComputeStream();
    AVFramePtr encFrame(av_frame_alloc());
    checkError(!encFrame, "Failed to allocate NVENC hw frame");
    encFrame->format = AV_PIX_FMT_CUDA;
    encFrame->width = layout.width;
    encFrame->height = layout.height;
    encFrame->pts = timing.pts;
    encFrame->duration = timing.duration;
    checkError(av_hwframe_get_buffer(s->outputEncoderCtx->hw_frames_ctx, encFrame.get(), 0) < 0, "Failed to get NVENC hw frame buffer");
    CUDA_CHECK(cudaMemcpy2DAsync(encFrame->data[0], encFrame->linesize[0], ySrc, yPitch, layout.width, layout.height, cudaMemcpyDefault, stream));
    if (cudaSurfaceFormat(s->outputFormat) == AV_PIX_FMT_YUV444P)
        for (int plane = 1; plane < 3; ++plane)
            CUDA_CHECK(
                cudaMemcpy2DAsync(encFrame->data[plane], encFrame->linesize[plane], chroma[plane - 1], chromaPitch[plane - 1], layout.chromaWidth, layout.chromaHeight, cudaMemcpyDefault, stream));
    else
        CUDA_CHECK(cudaMemcpy2DAsync(encFrame->data[1], encFrame->linesize[1], chroma[0], chromaPitch[0], 2 * layout.chromaWidth, layout.chromaHeight, cudaMemcpyDefault, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    pushToEncoder(queue, std::move(encFrame));
}

// CPU decode + NVENC: upload a decoded planar frame into an NVENC surface, the luma comes from the GPU watermark when watermarked
void uploadFrameGPU(VideoSession* s, const AVFrame* frame, const bool watermarked, const FrameTiming& timing, EncodeQueue& queue) {
    checkError(withoutFullRangeAlias(static_cast<AVPixelFormat>(frame->format)) != s->outputFormat, "Decoded video frame is not in the output format");
    const PlanarLayout layout = outputLayout(s);
    const auto stream = CudaStreamManager::getInstance().getComputeStream();
    const uint8_t* luma = watermarked ? s->watermarkedFrame.data() : frame->data[0];
    const int lumaPitch = watermarked ? layout.width : frame->linesize[0];
    // 4:4:4 surfaces take the planes as they are
    if (cudaSurfaceFormat(s->outputFormat) == AV_PIX_FMT_YUV444P) {
        encodeFrameGPU(s, luma, lumaPitch, {frame->data[1], frame->data[2]}, {frame->linesize[1], frame->linesize[2]}, timing, queue);
        return;
    }
    // packed U then V, interleaved into NV12/NV16 on the GPU
    CudaArray<uint8_t> packed(2 * layout.chromaHeight, layout.chromaWidth, stream), interleaved(layout.chromaHeight, 2 * layout.chromaWidth, stream);
    for (int plane = 1; plane < 3; ++plane)
        CUDA_CHECK(cudaMemcpy2DAsync(packed.data() + (plane - 1) * layout.chromaBytes(), layout.chromaWidth, frame->data[plane], frame->linesize[plane], layout.chromaWidth, layout.chromaHeight,
            cudaMemcpyHostToDevice, stream));
    cuda_utils::launchYUV420pToNV12Kernel(packed.data(), interleaved.data(), 2 * layout.chromaWidth, layout.chromaWidth, layout.chromaHeight, stream);
    encodeFrameGPU(s, luma, lumaPitch, {interleaved.data(), nullptr}, {2 * layout.chromaWidth, 0}, timing, queue);
}
#endif

// an encoder frame with U and V from the decoded frame, the caller fills Y
AVFramePtr buildEncFrame(VideoSession* s, const FrameTiming& timing, const AVFrame* srcChromaFrame) {
    checkError(withoutFullRangeAlias(static_cast<AVPixelFormat>(srcChromaFrame->format)) != s->outputFormat, "Decoded video frame is not in the output format");
    const PlanarLayout layout = outputLayout(s);
    AVFramePtr encFrame = encoderFrame(s, timing);
    for (int plane = 1; plane < 3; ++plane)
        copyPlane(encFrame->data[plane], encFrame->linesize[plane], srcChromaFrame->data[plane], srcChromaFrame->linesize[plane], layout.chromaWidth, layout.chromaHeight);
    return encFrame;
}

#if defined(_USE_OPENCL_)
void embedHdrFrameOpenCL(VideoSession* s, int& framesCount, const AVFrame* frame, EncodeQueue& queue);
#elif defined(_USE_CUDA_)
void embedUploadedHdrFrame(VideoSession* s, int& framesCount, const AVFrame* frame, EncodeQueue& queue);
#endif

// software decode embed runs on async encode thread
void embedWatermark(VideoSession* s, int& framesCount, const AVFrame* frame, EncodeQueue& queue) {
#if defined(_USE_GPU_)
    // CPU-decoded PQ HDR is uploaded and tonemapped like hardware-decoded frames
    if (s->isHdr) {
#if defined(_USE_OPENCL_)
        embedHdrFrameOpenCL(s, framesCount, frame, queue);
#else
        embedUploadedHdrFrame(s, framesCount, frame, queue);
#endif
        return;
    }
#endif
    const FrameTiming timing = frameTiming(s, frame);
    const bool doEmbed = framesCount % s->settings.watermarkInterval == 0;
#if defined(_USE_GPU_)
#if defined(_USE_OPENCL_)
    const bool encoderSurfaces = s->openclVideoOutput != nullptr;
#else
    const bool encoderSurfaces = s->outputEncoderCtx->hw_frames_ctx != nullptr;
#endif
    if (encoderSurfaces) {
        // the hardware encoder reads GPU surfaces: CPU-decoded planes are uploaded into them, the watermarked luma stays on the GPU
        if (doEmbed) {
            cout << std::format(" [Embedding frame {}]\n", framesCount + 1);
            loadInputFrame(s, frame->data[0], frame->linesize[0]);
            s->watermarkObj->makeWatermark(s->inputFrame, s->watermarkedFrame, WatermarkBase::Layout::RowMajor);
        }
#if defined(_USE_OPENCL_)
        AVFramePtr encoded(s->openclVideoOutput->copyFrom(frame, doEmbed ? s->watermarkedFrame.clBuffer() : cl::Buffer{}));
        encoded->pts = timing.pts;
        encoded->duration = timing.duration;
        pushToEncoder(queue, std::move(encoded));
#else
        uploadFrameGPU(s, frame, doEmbed, timing, queue);
#endif
        ++framesCount;
        return;
    }
#endif
    if (doEmbed) {
        cout << std::format(" [Embedding frame {}]\n", framesCount + 1);
        // the encoder frame owns the destination while the encode thread processes earlier frames
        auto encFrame = buildEncFrame(s, timing, frame);
        fillYPlane(frame, s, encFrame.get());
        pushToEncoder(queue, std::move(encFrame));
    } else {
        // passthrough: take a refcounted reference to the decoded frame (zero data copy)
        AVFramePtr ref(av_frame_alloc());
        checkError(av_frame_ref(ref.get(), frame) < 0, "Failed to ref passthrough frame");
        // the deprecated full range (J) formats are relabeled, the encoder context already carries color_range=AVCOL_RANGE_JPEG
        ref->format = withoutFullRangeAlias(static_cast<AVPixelFormat>(ref->format));
        ref->pts = timing.pts;
        ref->duration = timing.duration;
        pushToEncoder(queue, std::move(ref));
    }
    framesCount++;
}

void detectWatermark(VideoSession* s, int& framesCount, const AVFrame* frame) {
    if (framesCount % s->settings.watermarkInterval != 0) {
        framesCount++;
        return;
    }
    loadInputFrame(s, frame->data[0], frame->linesize[0]);
    const float correlation = s->watermarkObj->detectWatermark(s->inputFrame);
    cout << "Correlation for frame: " << (framesCount + 1) << ": " << correlation << "\n";
    framesCount++;
}

#if defined(_USE_OPENCL_)
// unselected 8-bit SDR frames go straight from the decoder surface to an encoder surface
bool directOpenCLCopy(const VideoSession* s, const VideoMode mode, const bool selected) {
    return mode == VideoMode::EMBED && s->openclVideoOutput && s->outputFormat == AV_PIX_FMT_YUV420P && !selected && !s->isHdr && s->decodedFormat == AV_PIX_FMT_NV12;
}

// submits the GPU copies of a frame as soon as it is decoded, processOpenCLFrame consumes them in decode order
void stageOpenCLFrame(VideoSession* s, const AVFrame* frame, const int frameIndex, const VideoMode mode) {
    const bool selected = frameIndex % s->settings.watermarkInterval == 0;
    if (mode == VideoMode::DETECT && !selected)
        return;
    if (!s->openclVideoFrame || (mode == VideoMode::EMBED && s->openclVideoFrame->planarFormat() != s->outputFormat))
        s->openclVideoFrame = OclQueueManager::getInstance().getVideoInterop()->createFrame(frame, OclQueueManager::getInstance().getQueueRaw(), s->isHdr, s->outputFormat);
    if (directOpenCLCopy(s, mode, selected))
        s->openclVideoOutput->stage(frame);
    else // SDR detection reads luma only
        s->openclVideoFrame->stage(frame, mode == VideoMode::DETECT && !s->isHdr);
}

// the packed planar frame, with the watermarked luma when selected, read straight into a pinned encoder frame
AVFramePtr readBackFrame(VideoSession* s, const FrameTiming& timing, const cl::Buffer& packed, const bool watermarked) {
    AVFramePtr encFrame = encoderFrame(s, timing);
    const PlanarLayout layout = outputLayout(s);
    auto& queue = OclQueueManager::getInstance().getQueue();
    for (int plane = 0; plane < 3; ++plane) {
        const bool fromWatermark = plane == 0 && watermarked;
        const int width = plane == 0 ? layout.width : layout.chromaWidth;
        const int height = plane == 0 ? layout.height : layout.chromaHeight;
        // the in-order queue has finished every read when the last one returns
        readPlane(
            queue, fromWatermark ? s->watermarkedFrame.clBuffer() : packed, fromWatermark ? 0 : layout.offset(plane), width, height, encFrame->data[plane], encFrame->linesize[plane], plane == 2);
    }
    return encFrame;
}

// a converted 8-bit planar frame to the encoder, the watermarked luma replaces Y on selected frames
void encodePackedOpenCLFrame(VideoSession* s, const AVFrame* frame, const cl::Buffer& packed, const bool selected, EncodeQueue& encoder) {
    const auto timing = frameTiming(s, frame);
    if (s->openclVideoOutput) {
        AVFramePtr encoded(s->openclVideoOutput->copyFrom(packed, selected ? s->watermarkedFrame.clBuffer() : cl::Buffer{}));
        encoded->pts = timing.pts;
        encoded->duration = timing.duration;
        pushToEncoder(encoder, std::move(encoded));
    } else {
        pushToEncoder(encoder, readBackFrame(s, timing, packed, selected));
    }
}

// CPU-decoded PQ HDR: the 10-bit planes are uploaded and tonemapped like hardware-decoded frames
void embedHdrFrameOpenCL(VideoSession* s, int& framesCount, const AVFrame* frame, EncodeQueue& queue) {
    const bool selected = framesCount % s->settings.watermarkInterval == 0;
    if (!s->openclHdrTonemap) {
        const auto [height, width] = s->videoDims();
        s->openclHdrTonemap = std::make_unique<OclHdrTonemap>(width, height, OclQueueManager::getInstance().getQueueRaw());
    }
    const uint16_t* planes[3]{reinterpret_cast<const uint16_t*>(frame->data[0]), reinterpret_cast<const uint16_t*>(frame->data[1]), reinterpret_cast<const uint16_t*>(frame->data[2])};
    const int pitches[3]{frame->linesize[0], frame->linesize[1], frame->linesize[2]};
    const auto& packed = s->openclHdrTonemap->convertPlanar10(planes, pitches, hdrMobius(s), chromaSiting(s->inputDecoderCtx.get(), frame), selected ? lumaBuffer(s).clBuffer() : cl::Buffer{});
    if (selected) {
        cout << std::format(" [Embedding frame {}]\n", framesCount + 1);
        s->watermarkObj->makeWatermark(s->inputFrame, s->watermarkedFrame, WatermarkBase::Layout::RowMajor);
    }
    encodePackedOpenCLFrame(s, frame, packed, selected, queue);
    ++framesCount;
}

void processOpenCLFrame(VideoSession* s, const AVFrame* frame, int& framesCount, const VideoMode mode, EncodeQueue* encoder = nullptr) {
    const bool selected = framesCount % s->settings.watermarkInterval == 0;
    if (mode == VideoMode::DETECT && !selected) {
        ++framesCount;
        return;
    }
    auto& manager = OclQueueManager::getInstance();
    if (directOpenCLCopy(s, mode, selected)) {
        AVFramePtr encoded(s->openclVideoOutput->copyFrom(frame));
        const auto timing = frameTiming(s, frame);
        encoded->pts = timing.pts;
        encoded->duration = timing.duration;
        pushToEncoder(*encoder, std::move(encoded));
        ++framesCount;
        return;
    }
    const auto& copied = s->openclVideoFrame->copyFrom(frame);
    const auto& packed = s->isHdr ? s->openclHdrTonemap->convert(copied, hdrMobius(s), chromaSiting(s->inputDecoderCtx.get(), frame), selected ? lumaBuffer(s).clBuffer() : cl::Buffer{}) : copied;
    const auto [height, width] = s->videoDims();
    if (selected) {
        if (!s->isHdr)
            cl_utils::launchPitchedToFloat(packed, lumaBuffer(s).clBuffer(), width, height, width, manager.getQueue());
        if (mode == VideoMode::DETECT) {
            const float correlation = s->watermarkObj->detectWatermark(s->inputFrame);
            cout << "Correlation for frame: " << (framesCount + 1) << ": " << correlation << "\n";
        } else {
            cout << std::format(" [Embedding frame {}]\n", framesCount + 1);
            s->watermarkObj->makeWatermark(s->inputFrame, s->watermarkedFrame, WatermarkBase::Layout::RowMajor);
        }
    }
    if (mode == VideoMode::EMBED)
        encodePackedOpenCLFrame(s, frame, packed, selected, *encoder);
    ++framesCount;
}
#endif

#if defined(_USE_CUDA_)
// a P010 frame on the GPU: an NVDEC surface or an uploaded CPU-decoded frame, pitches in bytes
struct P010Planes {
    const uint16_t* y;
    int yPitch;
    const uint16_t* uv;
    int uvPitch;
};

P010Planes p010Planes(const AVFrame* frame) { return {reinterpret_cast<const uint16_t*>(frame->data[0]), frame->linesize[0], reinterpret_cast<const uint16_t*>(frame->data[1]), frame->linesize[1]}; }

// HDR P010LE to SDR in one pass, each output optional: float luma (watermark input), 8-bit luma and NV12 chroma (pitch width)
void tonemapHdr(VideoSession* s, const P010Planes& source, const AVFrame* frame, float* luma, uint8_t* y, uint8_t* uv, const cudaStream_t stream) {
    const auto [height, width] = s->videoDims();
    cuda_utils::launchP010HdrToSdrKernel(
        source.y, source.yPitch, source.uv, source.uvPitch, y, width, luma, uv, width, width, height, chromaSiting(s->inputDecoderCtx.get(), frame), hdrMobius(s), stream);
}

// interleaved U and V rows (NV12 half height, NV16 full height) into the planes of the encoder frame, asynchronous on the stream
void interleavedChromaToHost(const PlanarLayout& layout, const uint8_t* uv, const int uvPitch, AVFrame* encFrame, const cudaStream_t stream) {
    CudaArray<uint8_t> chroma(2 * layout.chromaHeight, layout.chromaWidth, stream);
    cuda_utils::launchNV12ToYUV420pKernel(uv, uvPitch, chroma.data(), layout.chromaWidth, layout.chromaHeight, stream);
    for (int plane = 1; plane < 3; ++plane)
        CUDA_CHECK(cudaMemcpy2DAsync(encFrame->data[plane], encFrame->linesize[plane], chroma.data() + (plane - 1) * layout.chromaBytes(), layout.chromaWidth, layout.chromaWidth, layout.chromaHeight,
            cudaMemcpyDeviceToHost, stream));
}

// U and V of an 8-bit NVDEC frame (NV12, NV16 or YUV444P) into the encoder frame, asynchronous on the stream
void chromaToHost(VideoSession* s, const AVFrame* frame, AVFrame* encFrame, const cudaStream_t stream) {
    const PlanarLayout layout = outputLayout(s);
    const AVPixelFormat surface = decodedPixelFormat(frame);
    checkError(surface != cudaSurfaceFormat(s->outputFormat), "NVDEC frame layout does not match the output format");
    if (surface != AV_PIX_FMT_YUV444P) {
        interleavedChromaToHost(layout, frame->data[1], frame->linesize[1], encFrame, stream);
        return;
    }
    for (int plane = 1; plane < 3; ++plane)
        CUDA_CHECK(
            cudaMemcpy2DAsync(encFrame->data[plane], encFrame->linesize[plane], frame->data[plane], frame->linesize[plane], layout.chromaWidth, layout.chromaHeight, cudaMemcpyDeviceToHost, stream));
}

// HDR P010LE tonemapped to BT.709 SDR in CUDA kernels, watermarked, then sent to encoder
// One conversion gives the chroma and the watermark input or the SDR luma
void embedHdrFrame(VideoSession* s, int& framesCount, const P010Planes& source, const AVFrame* frame, EncodeQueue& queue) {
    const auto stream = CudaStreamManager::getInstance().getComputeStream();
    const auto [height, width] = s->videoDims();
    const bool doEmbed = framesCount % s->settings.watermarkInterval == 0;
    const FrameTiming timing = frameTiming(s, frame);
    if (doEmbed)
        cout << std::format(" [Embedding frame {}]\n", framesCount + 1);
    CudaArray<uint8_t> uvNV12(height / 2, width, stream), hdrY;
    if (!doEmbed)
        hdrY = CudaArray<uint8_t>(height, width, stream);
    tonemapHdr(s, source, frame, doEmbed ? lumaBuffer(s).data() : nullptr, doEmbed ? nullptr : hdrY.data(), uvNV12.data(), stream);
    if (s->outputEncoderCtx->hw_frames_ctx) {
        if (doEmbed)
            s->watermarkObj->makeWatermark(s->inputFrame, s->watermarkedFrame, WatermarkBase::Layout::RowMajor);
        encodeFrameGPU(s, doEmbed ? s->watermarkedFrame.data() : hdrY.data(), width, {uvNV12.data(), nullptr}, {width, 0}, timing, queue);
    } else {
        // the planes are read back straight into the pinned encoder frame, the chroma planar
        AVFramePtr encFrame = encoderFrame(s, timing);
        interleavedChromaToHost(outputLayout(s), uvNV12.data(), width, encFrame.get(), stream);
        if (doEmbed)
            embedAndFillYPlane(s, s->inputFrame, encFrame.get());
        else
            hdrY.toHostPitched(encFrame->data[0], width, encFrame->linesize[0]);
        pushToEncoder(queue, std::move(encFrame));
    }
    framesCount++;
}

// CPU-decoded PQ HDR: the yuv420p10le planes go to the GPU as P010 and are converted like NVDEC frames
void embedUploadedHdrFrame(VideoSession* s, int& framesCount, const AVFrame* frame, EncodeQueue& queue) {
    const auto stream = CudaStreamManager::getInstance().getComputeStream();
    const auto [height, width] = s->videoDims();
    CudaArray<uint16_t> planar(height * 3 / 2, width, stream), y(height, width, stream), uv(height / 2, width, stream);
    for (int plane = 0; plane < 3; ++plane) {
        const int planeWidth = plane == 0 ? width : width / 2, planeHeight = plane == 0 ? height : height / 2;
        const size_t offset = plane == 0 ? 0 : static_cast<size_t>(width) * height + (plane - 1) * static_cast<size_t>(planeWidth) * planeHeight;
        CUDA_CHECK(cudaMemcpy2DAsync(
            planar.data() + offset, planeWidth * sizeof(uint16_t), frame->data[plane], frame->linesize[plane], planeWidth * sizeof(uint16_t), planeHeight, cudaMemcpyHostToDevice, stream));
    }
    cuda_utils::launchYUV420p10ToP010Kernel(planar.data(), y.data(), uv.data(), width, height, stream);
    embedHdrFrame(s, framesCount, {y.data(), width * 2, uv.data(), width * 2}, frame, queue);
}

// embed watermark in NVDEC frame
void embedWatermarkHWAccel(VideoSession* s, int& framesCount, const AVFrame* frame, EncodeQueue& queue) {
    if (s->isHdr) {
        embedHdrFrame(s, framesCount, p010Planes(frame), frame, queue);
        return;
    }
    const auto stream = CudaStreamManager::getInstance().getComputeStream();
    const auto [height, width] = s->videoDims();
    const bool doEmbed = framesCount % s->settings.watermarkInterval == 0;
    const FrameTiming timing = frameTiming(s, frame);

    if (doEmbed)
        cout << std::format(" [Embedding frame {}]\n", framesCount + 1);

    if (s->outputEncoderCtx->hw_frames_ctx) {
        // NVDEC + NVENC, copy Y plane once into NVENC frame
        const uint8_t* ySrc = frame->data[0];
        int yPitch = frame->linesize[0];
        if (doEmbed) {
            ImageBuffer& luma = lumaBuffer(s);
            cuda_utils::launchPitchedToFloatKernel(frame->data[0], luma.data(), width, height, frame->linesize[0], stream);
            s->watermarkObj->makeWatermark(luma, s->watermarkedFrame, WatermarkBase::Layout::RowMajor);
            ySrc = s->watermarkedFrame.data();
            yPitch = width;
        }
        encodeFrameGPU(s, ySrc, yPitch, {frame->data[1], frame->data[2]}, {frame->linesize[1], frame->linesize[2]}, timing, queue);
    } else {
        // NVDEC + SW encoder, read back planes straight into pinned frame
        AVFramePtr encFrame = encoderFrame(s, timing);
        chromaToHost(s, frame, encFrame.get(), stream);
        if (doEmbed) {
            cuda_utils::launchPitchedToFloatKernel(frame->data[0], lumaBuffer(s).data(), width, height, frame->linesize[0], stream);
            embedAndFillYPlane(s, s->inputFrame, encFrame.get());
        } else {
            CUDA_CHECK(cudaMemcpy2DAsync(encFrame->data[0], encFrame->linesize[0], frame->data[0], frame->linesize[0], width, height, cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        pushToEncoder(queue, std::move(encFrame));
    }
    framesCount++;
}

void detectWatermarkHWAccel(VideoSession* s, int& framesCount, const AVFrame* frame) {
    if (framesCount % s->settings.watermarkInterval != 0) {
        framesCount++;
        return;
    }
    const auto [height, width] = s->videoDims();
    const auto stream = CudaStreamManager::getInstance().getComputeStream();
    ImageBuffer& luma = lumaBuffer(s);
    if (s->isHdr) {
        tonemapHdr(s, p010Planes(frame), frame, luma.data(), nullptr, nullptr, stream);
    } else {
        cuda_utils::launchPitchedToFloatKernel(frame->data[0], luma.data(), width, height, frame->linesize[0], stream);
    }
    const float correlation = s->watermarkObj->detectWatermark(luma);
    cout << "Correlation for frame: " << (framesCount + 1) << ": " << correlation << "\n";
    framesCount++;
}
#endif

// decodes packets, watermarks selected frames and remuxes auxiliary packets
template <typename Func, typename Stage = std::nullptr_t>
int processFrames(VideoSession* s, const bool needsFilter, const int filterInterval, Func&& processFrameArg, Stage&& stageFrame = nullptr) {
    // callback invoked once per frame, never forward as rvalue
    Func& processFrame = processFrameArg;
    const AVPacketPtr packet(av_packet_alloc());
    AVFramePtr frame(av_frame_alloc());
    AVFramePtr filteredFrame(nullptr);
    if (needsFilter)
        filteredFrame.reset(av_frame_alloc());
    int framesCount = 0;
    long long droppedPackets = 0;
    StreamShapeWatch shapeWatch(s->videoStream->codecpar);
    // lookahead queue lets decode engine run ahead of watermark processing
    std::deque<AVFramePtr> lookahead;
    const auto decodedFrames = [&] { return framesCount + static_cast<int>(lookahead.size()); };
    const auto stage = [&](const AVFrame* decoded, const int frameIndex) {
        if constexpr (!std::is_null_pointer_v<std::remove_cvref_t<Stage>>)
            stageFrame(decoded, frameIndex);
    };

    auto drainDecodedFrames = [&] {
        while (true) {
            const int ret = avcodec_receive_frame(s->inputDecoderCtx.get(), frame.get());
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
                break;
            checkAv(ret, "FFmpeg decoding error");
#if defined(_USE_OPENCL_)
            mapOpenCLVideoFrame(frame, s);
#endif
            const auto [expectedHeight, expectedWidth] = s->videoDims();
            checkError(frame->width != expectedWidth || frame->height != expectedHeight, resolutionChangeMessage(expectedWidth, expectedHeight, frame->width, frame->height));
            const AVPixelFormat frameFormat = decodedPixelFormat(frame.get());
            checkError(frameFormat != s->decodedFormat, pixelFormatChangeMessage(s->decodedFormat, frameFormat));
            const int frameIndex = decodedFrames();
            if (needsFilter && frameIndex % filterInterval == 0)
                filterFrame(frame, filteredFrame, s);
            if (frame->format != AV_PIX_FMT_D3D11) {
                const AVPixelFormat layout = decodedPixelFormat(frame.get());
                int minimumStrides[4]{};
                checkAv(av_image_fill_linesizes(minimumStrides, layout, frame->width), "Invalid video plane layout");
                const int chromaHeight = AV_CEIL_RSHIFT(frame->height, av_pix_fmt_desc_get(layout)->log2_chroma_h);
                for (int plane = 0; plane < 4; ++plane)
                    if (minimumStrides[plane] > 0) {
                        checkError(!frame->data[plane], "Missing decoded video plane");
                        InternalUtils::checkVideoPitch(plane == 1 || plane == 2 ? chromaHeight : frame->height, minimumStrides[plane], frame->linesize[plane]);
                    }
            }
            if (s->decodeLookahead == 0) {
                stage(frame.get(), frameIndex);
                processFrame(frame.get(), framesCount);
                continue;
            }
            AVFramePtr held(av_frame_alloc());
            checkError(!held, "Failed to allocate a decoded frame reference");
            av_frame_move_ref(held.get(), frame.get());
            lookahead.push_back(std::move(held));
            if (lookahead.size() > s->decodeLookahead) {
                processFrame(lookahead.front().get(), framesCount);
                lookahead.pop_front();
            }
            // staged after processing older frame, in-order GPU work waits for decode
            stage(lookahead.back().get(), frameIndex);
        }
    };

    while (true) {
        const int readRet = av_read_frame(s->inputFormatCtx.get(), packet.get());
        if (readRet < 0) {
            checkError(readRet != AVERROR_EOF, std::format("Reading input video failed after {} frames: {}", decodedFrames(), avErrorText(readRet)));
            const std::string change = shapeWatch.flush();
            checkError(!change.empty(), change);
            break;
        }
        if (packet->stream_index == s->videoStreamIndex) {
            const std::string change = shapeWatch.inspect(packet.get());
            checkError(!change.empty(), change);
            int sendRet = avcodec_send_packet(s->inputDecoderCtx.get(), packet.get());
            while (sendRet == AVERROR(EAGAIN)) {
                drainDecodedFrames();
                sendRet = avcodec_send_packet(s->inputDecoderCtx.get(), packet.get());
            }
            if (sendRet < 0 && sendRet != AVERROR(EAGAIN)) {
                if (droppedPackets++ == 0)
                    cout << info(std::format("Corrupt video packet skipped at frame {} ({})\n", decodedFrames(), avErrorText(sendRet)));
            } else if (sendRet >= 0) {
                drainDecodedFrames();
            }
        } else if (s->outputFormatCtx) {
            std::string auxErr;
            checkError(!s->auxMux.routePacket(packet.get(), auxErr), auxErr);
        }
        av_packet_unref(packet.get());
    }

    if (droppedPackets > 0)
        cout << info(std::format("{} corrupt video packets skipped in total\n", droppedPackets));

    int flushRet = avcodec_send_packet(s->inputDecoderCtx.get(), nullptr);
    while (flushRet == AVERROR(EAGAIN)) {
        drainDecodedFrames();
        flushRet = avcodec_send_packet(s->inputDecoderCtx.get(), nullptr);
    }
    checkError(flushRet < 0 && flushRet != AVERROR_EOF, "Flushing decoder failed: " + avErrorText(flushRet));
    drainDecodedFrames();
    for (; !lookahead.empty(); lookahead.pop_front())
        processFrame(lookahead.front().get(), framesCount);

    checkError(framesCount == 0, "Decoded 0 frames: the input video stream could not be decoded. No valid output was written.");
    return framesCount;
}

} // namespace

int findVideoStream(const AVFormatContext* inputFormatCtx) { return av_find_best_stream(const_cast<AVFormatContext*>(inputFormatCtx), AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0); }

AVCodecContextPtr openDecoder(VideoSession* session) {
    session->useHwDecoder = false;
    AVCodecContextPtr decoder;
#if defined(_USE_CUDA_)
    if (session->settings.useHwDecoder)
        decoder = openDecoderHWAccel(session);
#elif defined(_USE_OPENCL_)
    if (session->settings.useHwDecoder)
        decoder = openDecoderOpenCL(session);
#endif
    if (!decoder)
        decoder = openSoftwareDecoder(session->videoStream->codecpar, session->videoStream->time_base);
#if defined(_USE_GPU_)
    // GPU converts HDR for hardware and supported software frames
    if (decoder && !session->useHwDecoder)
        session->isHdr = gpuTonemapsCpuFrames(session, decoder.get());
#endif
#if defined(_USE_CUDA_)
    // NVDEC opens HDR only for GPU conversion
    if (decoder && session->useHwDecoder)
        session->isHdr = isHDR(decoder.get());
#endif
    return decoder;
}

namespace {
bool configureFilterGraph(VideoSession* s, const AVCodecContext* decoder, const AVFrame* firstFrame) {
    if (s->filterGraph)
        return true;
    const string exceptionMessage = "Failed to initialize filter graph in: ";
    const string filterDesc = getFilterGraphString(s, decoder);
    if (filterDesc.empty())
        return false;

    const AVRational timeBase = s->videoStream->time_base;
    auto sourceFormat = decoder->pix_fmt;
    const char* pixFmtName = av_get_pix_fmt_name(sourceFormat);
    string args = std::format("video_size={}x{}:pix_fmt={}:time_base={}/{}:pixel_aspect={}/{}", decoder->width, decoder->height, pixFmtName, timeBase.num, timeBase.den,
        decoder->sample_aspect_ratio.num, decoder->sample_aspect_ratio.den);

    AVFilterGraphPtr graphPtr(avfilter_graph_alloc());
    checkError(!graphPtr, exceptionMessage + "avfilter_graph_alloc");

    const AVFilter* bufferSrc = avfilter_get_by_name("buffer");
    const AVFilter* bufferSink = avfilter_get_by_name("buffersink");
    checkError(!bufferSrc || !bufferSink, exceptionMessage + "avfilter_get_by_name");

    AVFilterContext* srcCtx = avfilter_graph_alloc_filter(graphPtr.get(), bufferSrc, "in");
    checkError(!srcCtx, exceptionMessage + "avfilter_graph_alloc_filter");

    AVBufferSrcParametersPtr par(av_buffersrc_parameters_alloc());
    checkError(!par, exceptionMessage + "av_buffersrc_parameters_alloc");
    // graph starts with source colorspace and range to silence first-frame warnings
    par->color_space = decoder->colorspace;
    par->color_range = decoder->color_range;
    AVBufferRefPtr hwFramesRef;
    if (s->useHwDecoder) {
        par->format = sourceFormat;
        par->time_base = timeBase;
        if (firstFrame && firstFrame->hw_frames_ctx)
            hwFramesRef.reset(av_buffer_ref(firstFrame->hw_frames_ctx));
        else if (decoder->hw_frames_ctx)
            hwFramesRef.reset(av_buffer_ref(decoder->hw_frames_ctx));
        else if (decoder->hw_device_ctx) {
            AVBufferRef* rawFrames = av_hwframe_ctx_alloc(decoder->hw_device_ctx);
            checkError(!rawFrames, exceptionMessage + "av_hwframe_ctx_alloc");
            hwFramesRef.reset(rawFrames);
            AVHWFramesContext* frames_ctx = (AVHWFramesContext*)hwFramesRef->data;
            frames_ctx->format = sourceFormat;
            frames_ctx->sw_format = decoder->sw_pix_fmt;
            frames_ctx->width = decoder->width;
            frames_ctx->height = decoder->height;
            if (frames_ctx->sw_format == AV_PIX_FMT_NONE)
                frames_ctx->sw_format = AV_PIX_FMT_NV12;
            checkError(av_hwframe_ctx_init(hwFramesRef.get()) < 0, exceptionMessage + "av_hwframe_ctx_init");
        }
        if (hwFramesRef)
            par->hw_frames_ctx = hwFramesRef.get();
    }
    checkError(av_buffersrc_parameters_set(srcCtx, par.get()) < 0, exceptionMessage + "av_buffersrc_parameters_set");
    checkError(avfilter_init_str(srcCtx, args.c_str()) < 0, exceptionMessage + "avfilter_init_str");

    AVFilterContext* sinkCtx = avfilter_graph_alloc_filter(graphPtr.get(), bufferSink, "out");
    checkError(!sinkCtx, exceptionMessage + "avfilter_graph_alloc_filter");
    checkError(avfilter_init_str(sinkCtx, nullptr) < 0, exceptionMessage + "avfilter_init_str");

    AVFilterInOutPtr outputs(avfilter_inout_alloc());
    AVFilterInOutPtr inputs(avfilter_inout_alloc());
    checkError(!outputs || !inputs, exceptionMessage + "avfilter_inout_alloc");
    outputs->name = av_strdup("in");
    outputs->filter_ctx = srcCtx;
    outputs->pad_idx = 0;
    outputs->next = nullptr;
    inputs->name = av_strdup("out");
    inputs->filter_ctx = sinkCtx;
    inputs->pad_idx = 0;
    inputs->next = nullptr;

    AVFilterInOut* inputsRaw = inputs.release();
    AVFilterInOut* outputsRaw = outputs.release();
    if (avfilter_graph_parse_ptr(graphPtr.get(), filterDesc.c_str(), &inputsRaw, &outputsRaw, nullptr) < 0) {
        inputs.reset(inputsRaw);
        outputs.reset(outputsRaw);
        throw std::runtime_error(exceptionMessage + "avfilter_graph_parse_ptr");
    }
    inputs.reset(inputsRaw);
    outputs.reset(outputsRaw);

    checkError(avfilter_graph_config(graphPtr.get(), nullptr) < 0, exceptionMessage + "avfilter_graph_config");
    s->filterGraph = std::move(graphPtr);
    s->buffersrcCtx = srcCtx;
    s->buffersinkCtx = sinkCtx;
    return true;
}
} // namespace

bool initFilterGraph(VideoSession* s) { return configureFilterGraph(s, s->inputDecoderCtx.get(), nullptr); }

int videoDispatcher(VideoSession* s, VideoMode op, const bool needsFilter) {
    if (op == VideoMode::DETECT) {
        const int interval = s->settings.watermarkInterval;
#if defined(_USE_OPENCL_)
        if (s->useHwDecoder)
            return processFrames(
                s, needsFilter, interval, [&](const AVFrame* frame, int& count) { processOpenCLFrame(s, frame, count, op); },
                [&](const AVFrame* frame, const int index) { stageOpenCLFrame(s, frame, index, op); });
#endif
#if defined(_USE_CUDA_)
        if (s->useHwDecoder)
            return processFrames(s, needsFilter, interval, [&](const AVFrame* frame, int& framesCount) { detectWatermarkHWAccel(s, framesCount, frame); });
#endif
        return processFrames(s, needsFilter, interval, [&](const AVFrame* frame, int& framesCount) { detectWatermark(s, framesCount, frame); });
    }

    // dedicated encode thread runs concurrently with decode and watermark thread
    EncodeQueue queue;
    std::exception_ptr encErr;
    std::thread encThread(encodeWorker, s, std::ref(queue), std::ref(encErr));
    int result = 0;
    try {
        s->auxMux.setSink([&queue](AVPacket* packet) {
            AVPacketPtr owned(av_packet_alloc());
            if (!owned || av_packet_ref(owned.get(), packet) < 0)
                return false;
            return queue.push(std::move(owned));
        });

#if defined(_USE_CUDA_)
        if (s->useHwDecoder)
            result = processFrames(s, needsFilter, 1, [&](const AVFrame* frame, int& framesCount) { embedWatermarkHWAccel(s, framesCount, frame, queue); });
        else
#elif defined(_USE_OPENCL_)
        if (s->useHwDecoder)
            result = processFrames(
                s, needsFilter, 1, [&](const AVFrame* frame, int& count) { processOpenCLFrame(s, frame, count, op, &queue); },
                [&](const AVFrame* frame, const int index) { stageOpenCLFrame(s, frame, index, op); });
        else
#endif
            result = processFrames(s, needsFilter, 1, [&](const AVFrame* frame, int& framesCount) { embedWatermark(s, framesCount, frame, queue); });
        queue.close(); // tell encode thread to drain and exit
    } catch (...) {
        queue.abort(); // unblock encode thread if waiting
        encThread.join();
        if (encErr)
            std::rethrow_exception(encErr);
        throw;
    }
    encThread.join();
    if (encErr)
        std::rethrow_exception(encErr);
    return result;
}

namespace {
std::string defaultHardwareEncodeOptions() {
#if defined(_USE_OPENCL_)
    const auto vendor = OclQueueManager::getInstance().getDevice().getInfo<CL_DEVICE_VENDOR_ID>();
    if (vendor == 0x8086)
        return "-c:v hevc_qsv -global_quality 20";
    else if (vendor == 0x1002)
        return "-c:v hevc_amf -quality balanced -rc cqp -qp_i 22 -qp_p 22 -g 120 -vbaq 0";
    else if (vendor == 0x10de)
        return "-c:v hevc_nvenc -preset p2 -tune hq -cq 28 -b:v 0";
    else
        throw std::runtime_error("No default hardware encoder for this OpenCL GPU; specify hw_encode_options or use CPU encoding");
#else
    return "-c:v hevc_nvenc -preset p2 -tune hq -cq 28 -b:v 0";
#endif
}

// 8-bit planar YUV format of encoded frames, source chroma if supported, else 4:2:0
// CUDA and D3D11/QSV validate native 4:2:2/4:4:4 surfaces on selected GPU
AVPixelFormat negotiateOutputFormat(const VideoSession* s, const AVCodec* encoder, const bool cudaSurfaces) {
    if (s->isHdr || isHDR(s->inputDecoderCtx.get()))
        return AV_PIX_FMT_YUV420P;
    const AVPixelFormat preferred = planar8Bit(s->decodedFormat);
    if (s->settings.useHwEncoder) {
        if (preferred == AV_PIX_FMT_YUV422P || preferred == AV_PIX_FMT_YUV444P) {
            if (cudaSurfaces && encoderSupportsPixFmt(encoder, cudaSurfaceFormat(preferred)))
                return preferred;
#if defined(_USE_OPENCL_)
            if (OclQueueManager::getInstance().getVideoInterop() && encoderSupportsPixFmt(encoder, openclSurfaceFormat(preferred)))
                return preferred;
#endif
        }
        return AV_PIX_FMT_YUV420P;
    }
    if (encoderSupportsPixFmt(encoder, preferred))
        return preferred;
    if (!s->useHwDecoder)
        for (const auto candidate : {AV_PIX_FMT_YUV422P, AV_PIX_FMT_YUV444P})
            if (keepsChroma(candidate, preferred) && encoderSupportsPixFmt(encoder, candidate))
                return candidate;
    return AV_PIX_FMT_YUV420P;
}

// true when output format loses chroma resolution of source
void reportChromaConversion(const VideoSession* s, const std::string& codecName, const bool rejected) {
    const auto* source = av_pix_fmt_desc_get(s->decodedFormat);
    if (!source || source->nb_components < 3 || keepsChroma(s->outputFormat, s->decodedFormat))
        return;
    const std::string reason = s->isHdr || isHDR(s->inputDecoderCtx.get()) ? "HDR tonemapping outputs 4:2:0"
                               : rejected                                  ? std::format("Encoder '{}' cannot encode {} on this GPU", codecName, pixelFormatName(planar8Bit(s->decodedFormat)))
                               : s->settings.useHwEncoder                  ? "Hardware encoder '" + codecName + "' receives 4:2:0 surfaces"
                                                                           : "Encoder '" + codecName + "' cannot keep this chroma resolution";
    cout << info(std::format("{}: the source chroma ({}) is converted to {}.\n", reason, pixelFormatName(s->decodedFormat), pixelFormatName(s->outputFormat)));
}

#if defined(_USE_OPENCL_)
void configureOpenclEncoder(VideoSession* s, const std::string& codecName, const AVCodec* encoder, AVCodecContext* encCtx, OptionDict& opts) {
    if (!s->settings.useHwEncoder)
        return;
    const int width = encCtx->width, height = encCtx->height;
    auto& manager = OclQueueManager::getInstance();
    const auto vendor = manager.getDevice().getInfo<CL_DEVICE_VENDOR_ID>();
    const bool qsv = codecName.ends_with("_qsv");
    const bool amf = codecName.ends_with("_amf");
    const bool nvenc = codecName.ends_with("_nvenc");
    const bool mf = codecName.ends_with("_mf");
    checkError((qsv && vendor != 0x8086) || (amf && vendor != 0x1002) || (nvenc && vendor != 0x10de), "Hardware encoder '" + codecName + "' does not match the selected OpenCL GPU vendor");
    const auto interop = manager.getVideoInterop();
    if (nvenc && !interop)
        bindHostNvencDevice(manager.getDevice()(), opts);
    if (qsv || amf || mf || (nvenc && interop)) {
        checkError(!interop, "Hardware encoding on this OpenCL GPU requires D3D11 sharing (cl_khr_d3d11_sharing or cl_nv_d3d11_sharing); use CPU encoding instead");
        const auto pixelFormat = qsv ? AV_PIX_FMT_QSV : AV_PIX_FMT_D3D11;
        checkError(!encoderSupportsPixFmt(encoder, pixelFormat), "Encoder '" + codecName + "' cannot accept the selected GPU frame format");
        AVBufferRefPtr d3dDevice(interop->createDevice());
        checkError(!d3dDevice, "Failed to initialize D3D11 hardware encoder device");
        AVBufferRef* rawDevice = nullptr;
        if (qsv)
            checkAv(av_hwdevice_ctx_create_derived(&rawDevice, AV_HWDEVICE_TYPE_QSV, d3dDevice.get(), 0), "Derive QSV from selected D3D11 device");
        else
            rawDevice = av_buffer_ref(d3dDevice.get());
        encCtx->hw_device_ctx = rawDevice;
        checkError(!rawDevice, "Failed to reference hardware encoder device");
        encCtx->hw_frames_ctx = av_hwframe_ctx_alloc(rawDevice);
        checkError(!encCtx->hw_frames_ctx, "Failed to allocate hardware encoder frame pool");
        auto* frames = reinterpret_cast<AVHWFramesContext*>(encCtx->hw_frames_ctx->data);
        frames->format = pixelFormat;
        frames->sw_format = openclSurfaceFormat(s->outputFormat);
        frames->width = width;
        frames->height = height;
        // dynamic pools allocate separate textures, fixed D3D11 arrays unsupported on driver
        frames->initial_pool_size = 0;
        if (!qsv)
            static_cast<AVD3D11VAFramesContext*>(frames->hwctx)->BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_VIDEO_ENCODER;
        checkAv(av_hwframe_ctx_init(encCtx->hw_frames_ctx), "Initialize hardware encoder frame pool");
        encCtx->pix_fmt = pixelFormat;
        encCtx->sw_pix_fmt = frames->sw_format;
        if (qsv)
            av_dict_set(opts.ptr(), "async_depth", "4", AV_DICT_DONT_OVERWRITE);
        s->openclVideoOutput = interop->createOutput(encCtx->hw_frames_ctx, manager.getQueueRaw());
    }
}
#endif
} // namespace

void initOutputEncoder(VideoSession* s) {
    checkError(s->settings.encodeOutputPath.empty(), "No output path specified for video encode");
    checkError(sameFileOnDisk(pathFromUtf8(s->settings.videoFile), pathFromUtf8(s->settings.encodeOutputPath)),
        "Output path points to the same physical file as the input file (" + s->settings.encodeOutputPath + "). Overwriting the input video is forbidden.");

    const std::string optionsText = s->settings.useHwEncoder && s->settings.encodeOptions.empty() ? defaultHardwareEncodeOptions() : s->settings.encodeOptions;
    ParsedEncodeOptions parsed = parseEncodeOptions(optionsText);
    const std::string& codecName = parsed.codecName;
    OptionDict& opts = parsed.dictionary; // options consumed by encoder then muxer
    const MediaLog log = [](const std::string& msg) { cout << info(msg + "\n"); };
    reportParsedOptions(parsed, log);

    if (codecName.empty())
        throw std::runtime_error("No video encoder in the options string, needs '-c:v <encoder>' (e.g. -c:v hevc_nvenc)");

    const AVCodec* encoder = avcodec_find_encoder_by_name(codecName.c_str());
    if (!encoder)
        throw std::runtime_error("Encoder not found: '" + codecName + "' (check the video encoder options)");

    // hw_encode_options requires hardware encoder, encode_codec_options requires software
    const bool encIsHw = (encoder->capabilities & (AV_CODEC_CAP_HARDWARE | AV_CODEC_CAP_HYBRID)) != 0;
    if (s->settings.useHwEncoder && !encIsHw)
        throw std::runtime_error("hw_encoder is ON but '" + codecName + "' is a CPU (software) encoder. Use a hardware encoder in hw_encode_options, or turn hw_encoder off.");
    if (!s->settings.useHwEncoder && encIsHw)
        throw std::runtime_error("hw_encoder is OFF but '" + codecName + "' is a hardware encoder. Use a software encoder like libx265 in encode_codec_options, or turn hw_encoder on.");

    if (s->settings.useHwEncoder && codecName.ends_with("_mf")) {
        const auto* hardware = av_dict_get(opts.get(), "hw_encoding", nullptr, 0);
        checkError(hardware && std::string_view(hardware->value) != "1", "MF encoding requires -hw_encoding 1 when hw_encoder is enabled");
        av_dict_set(opts.ptr(), "hw_encoding", "1", 0);
    }

    AVFormatContext* rawOutFmt = nullptr;
    checkAv(avformat_alloc_output_context2(&rawOutFmt, nullptr, nullptr, s->settings.encodeOutputPath.c_str()), "Failed to create output format context for: " + s->settings.encodeOutputPath);
    s->outputFormatCtx.reset(rawOutFmt);

    // reject incompatible container and codec combinations early
    if (avformat_query_codec(s->outputFormatCtx->oformat, encoder->id, FF_COMPLIANCE_NORMAL) == 0)
        throw std::runtime_error("Codec '" + codecName + "' cannot be stored in the '" + std::string(s->outputFormatCtx->oformat->name) + "' container (" + s->settings.encodeOutputPath +
                                 "). Pick a container-compatible codec (prefer same container).");

    const int height = s->videoStream->codecpar->height;
    const int width = s->videoStream->codecpar->width;

    if (codecName.ends_with("_nvenc")) {
        const int maxNvencRes = (codecName == "h264_nvenc") ? 4096 : 8192;
        checkError(width > maxNvencRes || height > maxNvencRes, std::format("Output resolution ({}x{}) exceeds the NVENC hardware encoder limit of {}x{}.", width, height, maxNvencRes, maxNvencRes));
    }

#if defined(_USE_CUDA_)
    // NVENC reads CUDA frames on selected GPU, CPU planes uploaded into them
    bool useGpuPipeline = s->settings.useHwEncoder && encoderSupportsPixFmt(encoder, AV_PIX_FMT_CUDA);
#else
    bool useGpuPipeline = false;
#endif
    // 8-bit output keeps source chroma when supported, failed open retries with 4:2:0
    s->outputFormat = negotiateOutputFormat(s, encoder, useGpuPipeline);
    bool rejected = false;
    AVCodecContextPtr encCtx;
    while (true) {
        const auto* outputDescriptor = av_pix_fmt_desc_get(s->outputFormat);
        const int widthStep = 1 << outputDescriptor->log2_chroma_w, heightStep = 1 << outputDescriptor->log2_chroma_h;
        checkError(width % widthStep != 0 || height % heightStep != 0,
            std::format("{} output requires dimensions divisible by {}x{}; input is {}x{}.", pixelFormatName(s->outputFormat), widthStep, heightStep, width, height));

        encCtx.reset(avcodec_alloc_context3(encoder));
        checkError(!encCtx, "Failed to allocate encoder context");

        encCtx->thread_count = 0;
        encCtx->width = width;
        encCtx->height = height;
        encCtx->pix_fmt = useGpuPipeline ? AV_PIX_FMT_CUDA : s->outputFormat;
        encCtx->time_base = s->videoStream->time_base;
        encCtx->framerate = s->frameRate;
        encCtx->sample_aspect_ratio = s->videoStream->codecpar->sample_aspect_ratio;

        // tonemapped HDR to SDR writes SDR metadata, not source HDR flags
        const bool inputIsHDR = s->isHdr || isHDR(s->inputDecoderCtx.get());
        // deprecated full range J formats are full range even without tag
        const AVColorRange sourceRange = s->videoStream->codecpar->color_range == AVCOL_RANGE_UNSPECIFIED && withoutFullRangeAlias(s->inputDecoderCtx->pix_fmt) != s->inputDecoderCtx->pix_fmt
                                             ? AVCOL_RANGE_JPEG
                                             : s->videoStream->codecpar->color_range;
        encCtx->color_range = inputIsHDR && s->useHwDecoder ? AVCOL_RANGE_MPEG : sourceRange;
        encCtx->color_primaries = inputIsHDR ? AVCOL_PRI_BT709 : s->inputDecoderCtx->color_primaries;
        encCtx->color_trc = inputIsHDR ? AVCOL_TRC_BT709 : s->inputDecoderCtx->color_trc;
        const AVColorSpace sourceMatrix = s->inputDecoderCtx->colorspace == AVCOL_SPC_RESERVED ? AVCOL_SPC_UNSPECIFIED : s->inputDecoderCtx->colorspace;
        encCtx->colorspace = inputIsHDR ? AVCOL_SPC_BT709 : sourceMatrix;
        // unresampled chroma planes keep source sample position
        const bool sameChroma = keepsChroma(s->outputFormat, s->decodedFormat) && keepsChroma(s->decodedFormat, s->outputFormat);
        encCtx->chroma_sample_location = sameChroma ? s->inputDecoderCtx->chroma_sample_location : AVCHROMA_LOC_UNSPECIFIED;
        if (s->outputFormatCtx->oformat->flags & AVFMT_GLOBALHEADER)
            encCtx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

#if defined(_USE_CUDA_)
        // create hw_frames_ctx so NVENC reads CUDA frames directly from VRAM
        if (useGpuPipeline) {
            const AVBufferRefPtr cpuDecodeDevice = s->inputDecoderCtx->hw_device_ctx ? nullptr : createCudaDevice();
            checkError(!s->inputDecoderCtx->hw_device_ctx && !cpuDecodeDevice, "Failed to create the CUDA device for NVENC");
            AVBufferRef* framesRef = av_hwframe_ctx_alloc(s->inputDecoderCtx->hw_device_ctx ? s->inputDecoderCtx->hw_device_ctx : cpuDecodeDevice.get());
            checkError(!framesRef, "Failed to allocate hw_frames_ctx for NVENC");
            auto* fc = reinterpret_cast<AVHWFramesContext*>(framesRef->data);
            fc->format = AV_PIX_FMT_CUDA;
            fc->sw_format = cudaSurfaceFormat(s->outputFormat);
            fc->width = width;
            fc->height = height;
            fc->initial_pool_size = 8; // match EncodeQueue::MAX_DEPTH
            if (av_hwframe_ctx_init(framesRef) < 0) {
                av_buffer_unref(&framesRef);
                throw std::runtime_error("Failed to init hw_frames_ctx for NVENC");
            }
            encCtx->hw_frames_ctx = framesRef;
        }
#endif

#if defined(_USE_OPENCL_)
        try {
            configureOpenclEncoder(s, codecName, encoder, encCtx.get(), opts);
        } catch (const std::exception& error) {
            if (s->outputFormat == AV_PIX_FMT_YUV420P)
                throw;
            cout << info(std::string("Native chroma encoder surfaces are unavailable: ") + error.what() + ". Retrying 4:2:0.\n");
            s->openclVideoOutput.reset();
            rejected = true;
            s->outputFormat = AV_PIX_FMT_YUV420P;
            continue;
        }
        useGpuPipeline = encCtx->hw_frames_ctx != nullptr;
#endif

        const AVPixelFormat wantFmt = encCtx->pix_fmt;
        if (!encoderSupportsPixFmt(encoder, wantFmt))
            throw std::runtime_error(std::format("Encoder '{}' cannot encode {}{}. Pick another codec or change options.", codecName, pixelFormatName(wantFmt), useGpuPipeline ? " (GPU output)" : ""));

        // encoder consumes options from copy, muxer gets remainder of successful attempt
        OptionDict attemptOptions;
        checkAv(av_dict_copy(attemptOptions.ptr(), opts.get(), 0), "Failed to copy encoder options");
        const int opened = avcodec_open2(encCtx.get(), encoder, attemptOptions.ptr());
        if (opened >= 0) {
            opts = std::move(attemptOptions);
            break;
        }
        if (!useGpuPipeline || s->outputFormat == AV_PIX_FMT_YUV420P)
            checkAv(opened, "Failed to open encoder: " + codecName);
#if defined(_USE_OPENCL_)
        s->openclVideoOutput.reset();
#endif
        rejected = true;
        s->outputFormat = AV_PIX_FMT_YUV420P;
    }
    reportChromaConversion(s, codecName, rejected);

    AVStream* outVideoStream = avformat_new_stream(s->outputFormatCtx.get(), nullptr);
    checkError(!outVideoStream, "Failed to create output video stream");
    checkAv(avcodec_parameters_from_context(outVideoStream->codecpar, encCtx.get()), "Failed to copy encoder parameters to output video stream");
    outVideoStream->time_base = encCtx->time_base;
    // matroska derives default duration from framerate, without it durations drop
    outVideoStream->avg_frame_rate = encCtx->framerate;
    if (useGpuPipeline)
        outVideoStream->codecpar->format = s->outputFormat;
    if (!parsed.codecTag.empty())
        outVideoStream->codecpar->codec_tag = codecTagFromString(parsed.codecTag);

    // copy display matrix for rotation, omit stream side data for tonemapped SDR
    const AVCodecParameters* inputParams = s->videoStream->codecpar;
    if (const AVPacketSideData* displayMatrix = av_packet_side_data_get(inputParams->coded_side_data, inputParams->nb_coded_side_data, AV_PKT_DATA_DISPLAYMATRIX)) {
        AVPacketSideData* copied =
            av_packet_side_data_new(&outVideoStream->codecpar->coded_side_data, &outVideoStream->codecpar->nb_coded_side_data, AV_PKT_DATA_DISPLAYMATRIX, displayMatrix->size, 0);
        if (copied)
            std::memcpy(copied->data, displayMatrix->data, displayMatrix->size);
    }
    // copy stream tags and disposition, drop rotate tag
    av_dict_copy(&outVideoStream->metadata, s->videoStream->metadata, 0);
    av_dict_set(&outVideoStream->metadata, "rotate", nullptr, 0);
    outVideoStream->disposition = s->videoStream->disposition;
    s->outputVideoStreamIndex = outVideoStream->index;
    s->stretchLastPacket = durationsFollowDecodeOrder(s->outputFormatCtx->oformat);
    s->heldPacket.reset(av_packet_alloc());
    checkError(!s->heldPacket, "Failed to allocate encoder output packet");

    // remux audio tracks and transcode subtitles via AuxiliaryMux
    AuxiliaryMuxSetup auxSetup;
    auxSetup.input = s->inputFormatCtx.get();
    auxSetup.output = s->outputFormatCtx.get();
    auxSetup.outputPath = s->settings.encodeOutputPath;
    auxSetup.videoWidth = width;
    auxSetup.videoHeight = height;
    auxSetup.log = log;

    std::string auxError;
    const bool auxConfigured = s->auxMux.configure(auxSetup, auxError);
    checkError(!auxConfigured, auxError);

    checkError((s->outputFormatCtx->oformat->flags & AVFMT_NOFILE) != 0, "Video output requires a single-file container");
    s->outputFile.prepare(s->settings.encodeOutputPath);
    const std::string temporaryPath = s->outputFile.path();
    // faststart muxers reopen temporary file during finalization
    av_freep(&s->outputFormatCtx->url);
    s->outputFormatCtx->url = av_strdup(temporaryPath.c_str());
    checkError(!s->outputFormatCtx->url, "Failed to allocate temporary output URL");
    checkAv(avio_open(&s->outputFormatCtx->pb, temporaryPath.c_str(), AVIO_FLAG_WRITE), "Failed to open temporary output file");

    // unconsumed options forward to container muxer
    const int headerRet = avformat_write_header(s->outputFormatCtx.get(), opts.ptr());
    if (headerRet >= 0)
        reportUnusedOptions(opts.get(), codecName, s->outputFormatCtx->oformat->name, log);
    checkAv(headerRet, "Failed to write output container header");

    s->outputEncoderCtx = std::move(encCtx);
    cout << info("Encoder: " + codecName + ": \"" + s->settings.encodeOutputPath + "\"\n\n");
#if defined(_USE_OPENCL_)
    if (s->openclVideoOutput)
        cout << info("OpenCL hardware encode uses GPU copies into owned encoder surfaces.\n");
#endif
}

void flushAndFinalize(VideoSession* s) {
    if (!s->outputEncoderCtx || !s->outputFormatCtx)
        return;
    // send EOS to encoder, retry on EAGAIN
    int sendRet = avcodec_send_frame(s->outputEncoderCtx.get(), nullptr);
    while (sendRet == AVERROR(EAGAIN)) {
        drainEncoderPackets(s);
        sendRet = avcodec_send_frame(s->outputEncoderCtx.get(), nullptr);
    }
    checkError(sendRet < 0 && sendRet != AVERROR_EOF, "Failed to flush encoder: " + avErrorText(sendRet));
    drainEncoderPackets(s);
    checkError(s->encodedPackets == 0, "The encoder produced no video packets. No valid output was written.");
    // reader ends track at last decode timestamp plus duration
    if (s->holdingPacket && s->stretchLastPacket && s->presentationEnd != AV_NOPTS_VALUE && s->heldPacket->dts != AV_NOPTS_VALUE) {
        const int64_t decodeEnd = s->heldPacket->dts + s->firstReorderDelay;
        s->heldPacket->duration = std::max(s->heldPacket->duration, s->presentationEnd - decodeEnd);
    }
    writeHeldPacket(s);
    checkAv(av_write_trailer(s->outputFormatCtx.get()), "Failed to finalize encoded video");
    closeOutputFile(s);
    s->outputFile.commit();
}

void discardOutput(VideoSession* s) noexcept {
    s->outputFormatCtx.reset(); // close file
    s->outputFile.discard();
}

} // namespace video_utils
