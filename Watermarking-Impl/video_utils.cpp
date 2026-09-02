#include "AuxiliaryMux.hpp"
#include "AvUtil.hpp"
#include "buffer.hpp"
#include "common_utils.hpp"
#include "EncodeOptions.hpp"
#include "include/WatermarkCore.hpp"
#include "include/WatermarkTypes.hpp"
#include "video_defines.hpp"
#include "video_utils.hpp"
#include "VideoProcessingContext.hpp"
#include "WatermarkBase.hpp"
#include <algorithm>
#include <cerrno>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <format>
#include <functional>
#include <iostream>
#include <mutex>
#include <optional>
#include <queue>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <variant>

#if defined(_USE_CUDA_)
#include "CudaArray.hpp"
#include "CudaStreamManager.hpp"
#include "cuda_utils.hpp"
#include <cuda_runtime.h>
#include <cuda.h>
extern "C" {
#include "libavcodec/codec_id.h"
#include "libavutil/buffer.h"
#include "libavutil/hwcontext_cuda.h"
}
#elif defined(_USE_OPENCL_)
#include "OclQueueManager.hpp"
#include "OclArray.hpp"
#include "opencl_utils.hpp"
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
#include "libavcodec/defs.h"
}

#if defined(_USE_EIGEN_)
using namespace Eigen;
#endif

using namespace CommonUtils;
using namespace WatermarkCore;
using std::cout;
using std::string;

namespace {
constexpr AVPixelFormat supportedFormats[] = {AV_PIX_FMT_YUV420P, AV_PIX_FMT_YUVJ420P, AV_PIX_FMT_YUV420P10LE, AV_PIX_FMT_CUDA};
#if defined(_USE_CUDA_)
constexpr AVPixelFormat supportedHwFormats[] = {AV_PIX_FMT_NV12, AV_PIX_FMT_P010LE, AV_PIX_FMT_P016LE};
#endif
} // namespace

namespace video_utils {
namespace {
// internal implementations

bool isPixelFormatSupported(const std::span<const AVPixelFormat> formats, const AVPixelFormat format) {
    return std::ranges::any_of(formats, [&](auto f) { return f == format; });
}

// Queue that decouples the decode/watermark main thread from the encode background thread
// Holds either a frame to encode or an already prepared packet to remux directly, the
// encode thread is the only writer to outputFormatCtx, eliminating muxer race conditions
struct EncodeQueue {
    using Item = std::variant<AVFramePtr, AVPacketPtr>;
    static constexpr int MAX_DEPTH = 8;
    std::queue<Item> q;
    std::mutex mtx;
    std::condition_variable cv;
    bool closed = false;
    bool aborted = false;

    // push item (frame or packet), blocks when the queue is full
    void push(Item item) {
        std::unique_lock lk(mtx);
        cv.wait(lk, [&] { return static_cast<int>(q.size()) < MAX_DEPTH || aborted; });
        if (!aborted)
            q.push(std::move(item));
        cv.notify_all();
    }

    // signal EOS (end of stream), no more pushes will follow
    void close() {
        std::unique_lock lk(mtx);
        closed = true;
        cv.notify_all();
    }

    // returns nullopt on end (closed and drained) or abort
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

    // unblock any waiting caller immediately (error path)
    void abort() {
        std::unique_lock lk(mtx);
        aborted = true;
        cv.notify_all();
    }
};

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
    if (inputDecoder->capabilities & AV_CODEC_CAP_FRAME_THREADS)
        ctx->thread_type = FF_THREAD_FRAME;
    else if (inputDecoder->capabilities & AV_CODEC_CAP_SLICE_THREADS)
        ctx->thread_type = FF_THREAD_SLICE;
    else
        ctx->thread_count = 1;
    ctx->pkt_timebase = pktTimebase; // give the decoder the stream timebase (silences "Invalid pkt_timebase", correct VFR timestamps)
    if (avcodec_open2(ctx.get(), inputDecoder, nullptr) < 0)
        return nullptr;
    checkError(!isPixelFormatSupported(supportedFormats, ctx->pix_fmt), "Error: Video frame format not supported, aborting");
    return ctx;
}

#if defined(_USE_CUDA_)
// H264 NVDEC extra checks: It opens these profiles but decodes them wrong (unsupported by NVDEC hardware)
bool nvdecMisdecodesProfile(const AVCodecParameters* codecParams) {
    return codecParams->codec_id == AV_CODEC_ID_H264 && (codecParams->profile == AV_PROFILE_H264_HIGH_444_PREDICTIVE || codecParams->profile == AV_PROFILE_H264_CAVLC_444);
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

// try to open a CUDA hardware accelerated decoder, falls back to software on any failure
AVCodecContextPtr openDecoderHWAccel(const AVCodecParameters* inputCodecParams, bool& useHwDecoder, AVRational pktTimebase) {
    useHwDecoder = false;
    const char* decoderName = cuvidNameFor(inputCodecParams->codec_id);
    if (!decoderName) {
        cout << info(std::format("NVDEC cannot decode codec '{}', falling back to software decoder (CPU).\n", avcodec_get_name(inputCodecParams->codec_id)));
        return openSoftwareDecoder(inputCodecParams, pktTimebase);
    }
    if (nvdecMisdecodesProfile(inputCodecParams)) {
        const char* profileName = avcodec_profile_name(inputCodecParams->codec_id, inputCodecParams->profile);
        cout << info(std::format("NVDEC decodes H.264 profile '{}' incorrectly, falling back to software decoder (CPU).\n", profileName ? profileName : "?"));
        return openSoftwareDecoder(inputCodecParams, pktTimebase);
    }
    const AVCodec* inputDecoder = avcodec_find_decoder_by_name(decoderName);
    if (!inputDecoder)
        return openSoftwareDecoder(inputCodecParams, pktTimebase);
    AVCodecContextPtr ctx(avcodec_alloc_context3(inputDecoder));
    if (!ctx)
        return openSoftwareDecoder(inputCodecParams, pktTimebase);
    if (avcodec_parameters_to_context(ctx.get(), inputCodecParams) < 0)
        return openSoftwareDecoder(inputCodecParams, pktTimebase);

    // share our existing primary CUDA context with the decoder
    AVBufferRef* raw_hw_device_ctx = av_hwdevice_ctx_alloc(AV_HWDEVICE_TYPE_CUDA);
    if (!raw_hw_device_ctx)
        return openSoftwareDecoder(inputCodecParams, pktTimebase);
    AVBufferRefPtr hw_device_ctx(raw_hw_device_ctx);
    auto* hwCtx = reinterpret_cast<AVHWDeviceContext*>(hw_device_ctx->data);
    auto* cudaCtx = reinterpret_cast<AVCUDADeviceContext*>(hwCtx->hwctx);
    int runtimeDevice = 0;
    cudaGetDevice(&runtimeDevice);
    CUdevice cuDev = 0;
    if (cuDeviceGet(&cuDev, runtimeDevice) != CUDA_SUCCESS)
        return openSoftwareDecoder(inputCodecParams, pktTimebase);
    CUcontext primaryCtx = nullptr;
    if (cuDevicePrimaryCtxRetain(&primaryCtx, cuDev) != CUDA_SUCCESS)
        return openSoftwareDecoder(inputCodecParams, pktTimebase);
    cudaCtx->cuda_ctx = primaryCtx;
    cudaCtx->stream = CudaStreamManager::getInstance().getComputeStream();
    hwCtx->user_opaque = reinterpret_cast<void*>(static_cast<intptr_t>(cuDev));
    hwCtx->free = [](AVHWDeviceContext* c) {
        const CUdevice d = static_cast<CUdevice>(reinterpret_cast<intptr_t>(c->user_opaque));
        cuDevicePrimaryCtxRelease(d);
    };
    if (av_hwdevice_ctx_init(hw_device_ctx.get()) < 0)
        return openSoftwareDecoder(inputCodecParams, pktTimebase);
    if (cuCtxSetCurrent(primaryCtx) != CUDA_SUCCESS)
        return openSoftwareDecoder(inputCodecParams, pktTimebase);
    ctx->hw_device_ctx = av_buffer_ref(hw_device_ctx.get());
    ctx->get_format = [](AVCodecContext*, const enum AVPixelFormat*) { return AV_PIX_FMT_CUDA; };
    ctx->pkt_timebase = pktTimebase; // give cuvid the stream timebase (silences "Invalid pkt_timebase", correct VFR timestamps)

    // adaptive deinterlace for cuvid (does not regress or cost for progressive)
    AVDictionary* decOpts = nullptr;
    av_dict_set(&decOpts, "deint", "adaptive", 0);
    av_dict_set(&decOpts, "drop_second_field", "1", 0);
    int openStatus = avcodec_open2(ctx.get(), inputDecoder, &decOpts);
    av_dict_free(&decOpts);
    if (openStatus < 0) {
        openStatus = avcodec_open2(ctx.get(), inputDecoder, nullptr);
    }
    if (openStatus < 0) {
        cout << info(std::format("NVDEC decoder '{}' could not open input, falling back to software decoder (CPU).\n", decoderName));
        return openSoftwareDecoder(inputCodecParams, pktTimebase);
    }
    if (!isPixelFormatSupported(supportedHwFormats, ctx->sw_pix_fmt)) {
        cout << info(std::format("NVDEC output format '{}' unsupported, falling back to software decoder (CPU).\n",
            av_get_pix_fmt_name(ctx->sw_pix_fmt) ? av_get_pix_fmt_name(ctx->sw_pix_fmt) : "?"));
        return openSoftwareDecoder(inputCodecParams, pktTimebase);
    }
    useHwDecoder = true;
    return ctx;
}
#endif

// Decide the filter graph based on input depth, HDR status, and decoder type:
// 8-bit SDR (any decoder) -> "" (no filter needed)
// 10-bit SDR + NVDEC -> scale_cuda=format=nv12 (GPU 10->8-bit downscale)
// 10-bit SDR + SW decoder -> format=yuv420p (CPU 10->8-bit)
// HDR + NVDEC -> "" (tonemapped by custom CUDA kernels in embedWatermarkHWAccel)
// HDR + SW decoder -> zscale+tonemap (CPU tonemap)
string getFilterGraphString(const VideoSession* s) {
    if (!is10bit(s->inputDecoderCtx.get(), s->videoStream))
        return ""; // 8-bit SDR, no filtering (save processing time)
    if (!isHDR(s->inputDecoderCtx.get()))
        return s->useHwDecoder ? "scale_cuda=format=nv12" : "format=yuv420p"; // 10-bit SDR, fast downscale to 8-bit
    // NVDEC + HDR: conversion is done by CUDA kernels, no filter graph needed
    if (s->useHwDecoder)
        return "";
    // SW decoder + HDR: zscale needs explicit input primaries/matrix
    const char* primaries = av_color_primaries_name(s->inputDecoderCtx.get()->color_primaries);
    const char* matrix = av_color_space_name(s->inputDecoderCtx.get()->colorspace);
    // fallback to safe HDR10 defaults if any field is unspecified
    if (!primaries || s->inputDecoderCtx.get()->color_primaries == AVCOL_PRI_UNSPECIFIED)
        primaries = "bt2020";
    if (!matrix || s->inputDecoderCtx.get()->colorspace == AVCOL_SPC_UNSPECIFIED)
        matrix = "bt2020nc";
    return std::format("zscale=primaries={}:transfer=linear:matrix={}:npl=100,tonemap=mobius,zscale=transfer=bt709:primaries=bt709:matrix=bt709,format=yuv420p", primaries, matrix);
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

// upload a CPU uint8 Y plane into the GPU / Eigen input buffer
void loadInputFrame(VideoSession* s, const uint8_t* hostPtr) {
    const auto [height, width] = s->videoDims();
#if defined(_USE_CUDA_)
    const auto stream = CudaStreamManager::getInstance().getComputeStream();
    // upload host uint8 row-major to GPU, then use pitchedToFloat kernel (stride=width) to convert to column-major float
    CudaArray<uint8_t> hostGpu(height, width, hostPtr, stream);
    s->inputFrame = CudaArray<float>(height, width, stream);
    cuda_utils::launchPitchedToFloatKernel(hostGpu.data(), s->inputFrame.data(), width, height, width, stream);
#elif defined(_USE_OPENCL_)
    auto& mgr = OclQueueManager::getInstance();
    auto queue = mgr.getQueueRaw();
    OclArray<uint8_t> hostGpu(height, width, hostPtr, queue);
    s->inputFrame = OclArray<float>(height, width, queue);
    cl_utils::launchPitchedToFloat(hostGpu.clBuffer(), s->inputFrame.clBuffer(), width, height, width, mgr.getQueue());
#else
    s->inputFrame = Map<const Gray8Buffer>(hostPtr, width, height).transpose().template cast<float>();
#endif
}

// watermark the frame and fill hostFrame[0..W*H-1] with the processed Y plane
void embedAndFillYPlane(VideoSession* s, const ImageBuffer& buffer) {
    s->watermarkObj->makeWatermark(buffer, buffer, s->watermarkedFrame, MaskMethod::ME);
#if defined(_USE_CUDA_)
    {
        const auto stream = CudaStreamManager::getInstance().getComputeStream();
        CudaArray<uint8_t> rowMajorOut(s->watermarkedFrame.getRows(), s->watermarkedFrame.getCols(), stream);
        cuda_utils::launchColMajorToRowMajorU8Kernel(s->watermarkedFrame.data(), rowMajorOut.data(), static_cast<int>(s->watermarkedFrame.getCols()), static_cast<int>(s->watermarkedFrame.getRows()),
                                                     1, stream);
        rowMajorOut.toHost(s->hostFrame->get());
    }
#elif defined(_USE_OPENCL_)
    {
        const auto [height, width] = s->videoDims();
        auto& mgr = OclQueueManager::getInstance();
        OclArray<uint8_t> rowMajorOut(height, width, mgr.getQueueRaw());
        cl_utils::launchColMajorToRowMajorU8(s->watermarkedFrame.clBuffer(), rowMajorOut.clBuffer(), width, height, 1, mgr.getQueue());
        rowMajorOut.toHost(s->hostFrame->get());
    }
#elif defined(_USE_EIGEN_)
    {
        const auto [height, width] = s->videoDims();
        s->grayFrame = s->watermarkedFrame.getGray().transpose();
        std::memcpy(s->hostFrame->get(), s->grayFrame.data(), static_cast<size_t>(width) * height);
    }
#endif
}

void fillYPlane(const AVFrame* frame, VideoSession* s) {
    const auto [height, width] = s->videoDims();
    uint8_t* hostY = s->hostFrame->get();
    const uint8_t* srcY = frame->data[0];
    if (frame->linesize[0] != width) {
        for (int y = 0; y < height; y++)
            std::memcpy(hostY + y * width, srcY + y * frame->linesize[0], width);
        srcY = hostY;
    }
    loadInputFrame(s, srcY);
    embedAndFillYPlane(s, s->inputFrame);
}

// pull all ready packets from the encoder and write them to the output container (non-blocking)
void drainEncoderPackets(VideoSession* s) {
    const AVPacketPtr pkt(av_packet_alloc());
    const AVStream* outVideoStream = s->outputFormatCtx->streams[s->outputVideoStreamIndex];
    while (avcodec_receive_packet(s->outputEncoderCtx.get(), pkt.get()) == 0) {
        av_packet_rescale_ts(pkt.get(), s->outputEncoderCtx->time_base, outVideoStream->time_base);
        pkt->stream_index = s->outputVideoStreamIndex;
        checkAv(av_interleaved_write_frame(s->outputFormatCtx.get(), pkt.get()), "Failed to write encoded video packet");
        av_packet_unref(pkt.get());
    }
}

// encode thread: pops items, encodes frames OR writes passthrough packets, it is the
// only writer to outputFormatCtx so audio/subtitle remux and video encode never have race conditions
// does NOT flush the encoder on exit — flushAndFinalize handles that after join()
void encodeWorker(VideoSession* s, EncodeQueue& queue, std::exception_ptr& encErr) noexcept {
    try {
        while (auto item = queue.pop()) {
            if (std::holds_alternative<AVFramePtr>(*item)) {
                AVFramePtr frame = std::move(std::get<AVFramePtr>(*item));
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
// NVDEC+NVENC zero-copy: wrap CUDA Y+UV buffers in a hw AVFrame and give it to the encode thread
// uvOverride: when not null, use this preconverted uint8_t NV12 UV (HDR path) instead of srcFrame->data[1]
void encodeFrameGPU(VideoSession* s, const CudaArray<uint8_t>& yRowMajor, const AVFrame* srcFrame, const int64_t pts, EncodeQueue& queue, const uint8_t* uvOverride = nullptr) {
    const auto [height, width] = s->videoDims();
    const auto stream = CudaStreamManager::getInstance().getComputeStream();
    AVFramePtr encFrame(av_frame_alloc());
    checkError(!encFrame, "Failed to allocate NVENC hw frame");
    encFrame->format = AV_PIX_FMT_CUDA;
    encFrame->width = width;
    encFrame->height = height;
    encFrame->pts = pts;
    checkError(av_hwframe_get_buffer(s->outputEncoderCtx->hw_frames_ctx, encFrame.get(), 0) < 0, "Failed to get NVENC hw frame buffer");
    cudaMemcpy2DAsync(encFrame->data[0], encFrame->linesize[0], yRowMajor.data(), width, width, height, cudaMemcpyDeviceToDevice, stream);
    // HDR path passes preconverted uint8_t NV12 UV (stride=width) -> SDR path copies directly from decoded frame
    const uint8_t* uvSrc = uvOverride ? uvOverride : static_cast<const uint8_t*>(srcFrame->data[1]);
    const int uvPitch = uvOverride ? width : srcFrame->linesize[1];
    cudaMemcpy2DAsync(encFrame->data[1], encFrame->linesize[1], uvSrc, uvPitch, width, height / 2, cudaMemcpyDeviceToDevice, stream);
    cudaStreamSynchronize(stream);
    queue.push(std::move(encFrame));
}
#endif

// Y always comes from hostFrame (watermarked output)
// Chroma: copy directly from srcChromaFrame when provided (SW-decode path) to avoid a
// double copy through hostFrame, when null (NVDEC+SW-encoder path) chroma is already in
// hostFrame after the GPU NV12->YUV420p kernel so we read it from there
AVFramePtr buildEncFrame(VideoSession* s, const int64_t pts, const AVFrame* srcChromaFrame = nullptr) {
    const auto [height, width] = s->videoDims();
    const uint8_t* src = s->hostFrame->get();
    AVFramePtr encFrame(av_frame_alloc());
    checkError(!encFrame, "Failed to allocate encoder AVFrame");
    encFrame->format = AV_PIX_FMT_YUV420P;
    encFrame->width = width;
    encFrame->height = height;
    encFrame->pts = pts;
    checkError(av_frame_get_buffer(encFrame.get(), 0) < 0, "Failed to allocate encoder frame buffer");
    // Y: av_frame_get_buffer with align=0 sets linesize==width, so one memcpy covers the whole plane
    // fall back to row-by-row only when the encoder has a different stride
    if (encFrame->linesize[0] == width)
        std::memcpy(encFrame->data[0], src, static_cast<size_t>(width) * height);
    else
        for (int y = 0; y < height; y++)
            std::memcpy(encFrame->data[0] + y * encFrame->linesize[0], src + y * width, width);
    // chroma
    const int chromaH = height / 2;
    const int chromaW = width / 2;
    if (srcChromaFrame) {
        // SW-decode: read U+V directly from the decoded frame — no hostFrame hop
        for (int y = 0; y < chromaH; y++) {
            std::memcpy(encFrame->data[1] + y * encFrame->linesize[1], srcChromaFrame->data[1] + y * srcChromaFrame->linesize[1], chromaW);
            std::memcpy(encFrame->data[2] + y * encFrame->linesize[2], srcChromaFrame->data[2] + y * srcChromaFrame->linesize[2], chromaW);
        }
    } else {
        // NVDEC+SW-encoder: chroma already de-interleaved into hostFrame by the GPU kernel
        for (int y = 0; y < chromaH; y++) {
            std::memcpy(encFrame->data[1] + y * encFrame->linesize[1], src + static_cast<size_t>(width) * height + y * chromaW, chromaW);
            std::memcpy(encFrame->data[2] + y * encFrame->linesize[2], src + static_cast<size_t>(width) * height * 5 / 4 + y * chromaW, chromaW);
        }
    }
    return encFrame;
}

// SW-decode embed: always runs on the async encode thread path.
void embedWatermark(VideoSession* s, int& framesCount, const AVFrame* frame, EncodeQueue& queue) {
    const bool doEmbed = framesCount % s->settings.watermarkInterval == 0;
    if (doEmbed) {
        cout << std::format(" [Embedding frame {}]\n", framesCount + 1);
        fillYPlane(frame, s);
        // chroma goes decoded frame → encFrame directly inside buildEncFrame (no hostFrame hop)
        queue.push(buildEncFrame(s, framePts(frame), frame));
    } else {
        // passthrough: take a refcounted reference to the decoded frame (zero data copy)
        AVFramePtr ref(av_frame_alloc());
        checkError(av_frame_ref(ref.get(), frame) < 0, "Failed to ref passthrough frame");
        // if FULL range we normalize to YUV420P (YUVJ420p is deprecated)
        // encoder context already carries color_range=AVCOL_RANGE_JPEG, so players can understand it's full range
        if (ref->format == AV_PIX_FMT_YUVJ420P)
            ref->format = AV_PIX_FMT_YUV420P;
        ref->pts = framePts(frame);
        queue.push(std::move(ref));
    }
    framesCount++;
}

void detectWatermark(VideoSession* s, int& framesCount, const AVFrame* frame) {
    if (framesCount % s->settings.watermarkInterval != 0) {
        framesCount++;
        return;
    }
    const auto [height, width] = s->videoDims();
    uint8_t* srcY = frame->data[0];
    if (frame->linesize[0] != width) {
        for (int y = 0; y < height; y++)
            std::memcpy(s->hostFrame.get()->get() + y * width, frame->data[0] + y * frame->linesize[0], width);
        srcY = s->hostFrame.get()->get();
    }
    loadInputFrame(s, srcY);
    const float correlation = s->watermarkObj->detectWatermark(s->inputFrame, MaskMethod::ME);
    cout << "Correlation for frame: " << (framesCount + 1) << ": " << correlation << "\n";
    framesCount++;
}

#if defined(_USE_CUDA_)
// read HDR peak luminance from stream MaxCLL side data, normalized to 100 units (nits / 100)
// falls back to 10.0 (1000 nits) which is the most common
static float getHdrPeak(const VideoSession* s) {
    for (int i = 0; i < s->videoStream->codecpar->nb_coded_side_data; i++) {
        const AVPacketSideData& sd = s->videoStream->codecpar->coded_side_data[i];
        if (sd.type == AV_PKT_DATA_CONTENT_LIGHT_LEVEL) {
            const auto* cll = reinterpret_cast<const AVContentLightMetadata*>(sd.data);
            if (cll->MaxCLL > 0)
                return static_cast<float>(cll->MaxCLL) / 100.0f;
        }
    }
    return 10.0f; // 1000 nits default
}

// HDR helpers: convert P010LE CUDA planes to SDR format
static void loadHdrLuma(VideoSession* s, const AVFrame* frame, CudaArray<float>& dst, const MobiusParams& mobius, const cudaStream_t stream) {
    const auto [height, width] = s->videoDims();
    cuda_utils::launchP010HdrYToSdrFloatKernel(reinterpret_cast<const uint16_t*>(frame->data[0]), frame->linesize[0], reinterpret_cast<const uint16_t*>(frame->data[1]), frame->linesize[1], dst.data(),
                                               width, height, mobius, stream);
}

static CudaArray<uint8_t> convertHdrUV(VideoSession* s, const AVFrame* frame, const MobiusParams& mobius, const cudaStream_t stream) {
    const auto [height, width] = s->videoDims();
    CudaArray<uint8_t> uvNV12(width * height / 2, stream);
    cuda_utils::launchP010HdrUVToSdrNV12Kernel(reinterpret_cast<const uint16_t*>(frame->data[0]), frame->linesize[0], reinterpret_cast<const uint16_t*>(frame->data[1]), frame->linesize[1],
                                               uvNV12.data(), width, height, mobius, stream);
    return uvNV12;
}

// embed watermark in a NVDEC frame: Two encoder paths (NVENC, SW) and two content paths (SDR, HDR).
// HDR frames (P010LE PQ) are tonemapped to BT.709 SDR entirely in custom CUDA kernels
void embedWatermarkHWAccel(VideoSession* s, int& framesCount, const AVFrame* frame, EncodeQueue& queue) {
    const auto stream = CudaStreamManager::getInstance().getComputeStream();
    const auto [height, width] = s->videoDims();
    const bool doEmbed = framesCount % s->settings.watermarkInterval == 0;
    const bool isHdr = s->isHdr;
    const MobiusParams& mobius = s->mobius;

    if (doEmbed)
        cout << std::format(" [Embedding frame {}]\n", framesCount + 1);

    if (s->outputEncoderCtx->hw_frames_ctx) {
        // NVDEC + NVENC
        CudaArray<uint8_t> yRowMajor(height, width, stream);
        if (doEmbed) {
            CudaArray<float> lumaFloat(height, width, stream);
            if (isHdr)
                loadHdrLuma(s, frame, lumaFloat, mobius, stream);
            else
                cuda_utils::launchPitchedToFloatKernel(frame->data[0], lumaFloat.data(), width, height, frame->linesize[0], stream);
            s->watermarkObj->makeWatermark(lumaFloat, lumaFloat, s->watermarkedFrame, MaskMethod::ME);
            cuda_utils::launchColMajorToRowMajorU8Kernel(s->watermarkedFrame.data(), yRowMajor.data(), width, height, 1, stream);
        } else {
            if (isHdr)
                cuda_utils::launchP010HdrYToSdrU8Kernel(reinterpret_cast<const uint16_t*>(frame->data[0]), frame->linesize[0], reinterpret_cast<const uint16_t*>(frame->data[1]), frame->linesize[1],
                                                        yRowMajor.data(), width, height, mobius, stream);
            else
                cudaMemcpy2DAsync(yRowMajor.data(), width, frame->data[0], frame->linesize[0], width, height, cudaMemcpyDeviceToDevice, stream);
        }
        if (isHdr) {
            CudaArray<uint8_t> uvNV12 = convertHdrUV(s, frame, mobius, stream);
            encodeFrameGPU(s, yRowMajor, frame, framePts(frame), queue, uvNV12.data());
        } else {
            encodeFrameGPU(s, yRowMajor, frame, framePts(frame), queue);
        }
    } else {
        // NVDEC + SW encoder
        if (isHdr) {
            // convert HDR UV: P010LE -> NV12 uint8_t -> planar YUV420P in hostFrame
            CudaArray<uint8_t> uvNV12 = convertHdrUV(s, frame, mobius, stream);
            CudaArray<uint8_t> chromaBuffer(width * height / 2, stream);
            cuda_utils::launchNV12ToYUV420pKernel(uvNV12.data(), width, chromaBuffer.data(), width / 2, height / 2, stream);
            chromaBuffer.toHostAsync(s->hostFrame->get() + width * height);
            if (doEmbed) {
                CudaArray<float> lumaFloat(height, width, stream);
                loadHdrLuma(s, frame, lumaFloat, mobius, stream);
                cudaStreamSynchronize(stream);
                embedAndFillYPlane(s, lumaFloat);
            } else {
                cuda_utils::launchP010HdrYToSdrU8Kernel(reinterpret_cast<const uint16_t*>(frame->data[0]), frame->linesize[0], reinterpret_cast<const uint16_t*>(frame->data[1]), frame->linesize[1],
                                                        s->hostFrame->get(), width, height, mobius, stream);
                cudaStreamSynchronize(stream);
            }
        } else {
            CudaArray<uint8_t> chromaBuffer(width * height / 2, stream);
            cuda_utils::launchNV12ToYUV420pKernel(frame->data[1], frame->linesize[1], chromaBuffer.data(), width / 2, height / 2, stream);
            chromaBuffer.toHostAsync(s->hostFrame->get() + width * height);
            if (doEmbed) {
                CudaArray<float> lumaBuffer(height, width, stream);
                cuda_utils::launchPitchedToFloatKernel(frame->data[0], lumaBuffer.data(), width, height, frame->linesize[0], stream);
                cudaStreamSynchronize(stream);
                embedAndFillYPlane(s, lumaBuffer);
            } else {
                cudaMemcpy2DAsync(s->hostFrame->get(), width, frame->data[0], frame->linesize[0], width, height, cudaMemcpyDeviceToHost, stream);
                cudaStreamSynchronize(stream);
            }
        }
        queue.push(buildEncFrame(s, framePts(frame)));
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
    CudaArray<float> lumaBuffer(height, width, stream);
    if (s->isHdr) {
        loadHdrLuma(s, frame, lumaBuffer, s->mobius, stream);
    } else {
        cuda_utils::launchPitchedToFloatKernel(frame->data[0], lumaBuffer.data(), width, height, frame->linesize[0], stream);
    }
    const float correlation = s->watermarkObj->detectWatermark(lumaBuffer, MaskMethod::ME);
    cout << "Correlation for frame: " << (framesCount + 1) << ": " << correlation << "\n";
    framesCount++;
}
#endif

// main frames loop: decodes packets, calls processFrame for each video frame,
// remuxes audio/subtitle packets directly when outputFormatCtx is set (embed mode)
template <typename Func>
int processFrames(VideoSession* s, const bool needsFilter, Func&& processFrame) {
    const AVPacketPtr packet(av_packet_alloc());
    AVFramePtr frame(av_frame_alloc());
    AVFramePtr filteredFrame(nullptr);
    if (needsFilter)
        filteredFrame.reset(av_frame_alloc());
    int framesCount = 0;
    long long droppedPackets = 0;

    auto drainDecodedFrames = [&] {
        while (true) {
            const int ret = avcodec_receive_frame(s->inputDecoderCtx.get(), frame.get());
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
                break;
            checkAv(ret, "FFmpeg decoding error");
            if (needsFilter)
                filterFrame(frame, filteredFrame, s);
            std::forward<Func>(processFrame)(frame.get(), framesCount);
        }
    };

    while (true) {
        const int readRet = av_read_frame(s->inputFormatCtx.get(), packet.get());
        if (readRet < 0) {
            checkError(readRet != AVERROR_EOF, std::format("Reading input video failed after {} frames: {}", framesCount, avErrorText(readRet)));
            break;
        }
        if (packet->stream_index == s->videoStreamIndex) {
            int sendRet = avcodec_send_packet(s->inputDecoderCtx.get(), packet.get());
            while (sendRet == AVERROR(EAGAIN)) {
                drainDecodedFrames();
                sendRet = avcodec_send_packet(s->inputDecoderCtx.get(), packet.get());
            }
            if (sendRet < 0 && sendRet != AVERROR(EAGAIN)) {
                if (droppedPackets++ == 0)
                    cout << info(std::format("Corrupt video packet skipped at frame {} ({})\n", framesCount, avErrorText(sendRet)));
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

    checkError(framesCount == 0, "Decoded 0 frames: the input video stream could not be decoded. No valid output was written.");
    return framesCount;
}

} // namespace

// public API

int findVideoStream(const AVFormatContext* inputFormatCtx) { return av_find_best_stream(const_cast<AVFormatContext*>(inputFormatCtx), AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0); }

AVCodecContextPtr openDecoder(const AVCodecParameters* inputCodecParams, const bool useHwDecoderRequested, bool& useHwDecoder, AVRational pktTimebase) {
#if defined(_USE_CUDA_)
    if (useHwDecoderRequested)
        return openDecoderHWAccel(inputCodecParams, useHwDecoder, pktTimebase);
#endif
    useHwDecoder = false;
    return openSoftwareDecoder(inputCodecParams, pktTimebase);
}

bool initFilterGraph(VideoSession* s) {
    const string exceptionMessage = "Failed to initialize filter graph in: ";
    const string filterDesc = getFilterGraphString(s);
    if (filterDesc.empty())
        return false;

    const AVRational timeBase = s->videoStream->time_base;
    const char* pixFmtName = av_get_pix_fmt_name((AVPixelFormat)s->inputDecoderCtx->pix_fmt);
    string args = std::format("video_size={}x{}:pix_fmt={}:time_base={}/{}:pixel_aspect={}/{}", s->inputDecoderCtx->width, s->inputDecoderCtx->height, pixFmtName, timeBase.num, timeBase.den,
                              s->inputDecoderCtx->sample_aspect_ratio.num, s->inputDecoderCtx->sample_aspect_ratio.den);

    AVFilterGraphPtr graphPtr(avfilter_graph_alloc());
    checkError(!graphPtr, exceptionMessage + "avfilter_graph_alloc");

    const AVFilter* bufferSrc = avfilter_get_by_name("buffer");
    const AVFilter* bufferSink = avfilter_get_by_name("buffersink");
    checkError(!bufferSrc || !bufferSink, exceptionMessage + "avfilter_get_by_name");

    AVFilterContext* srcCtx = avfilter_graph_alloc_filter(graphPtr.get(), bufferSrc, "in");
    checkError(!srcCtx, exceptionMessage + "avfilter_graph_alloc_filter");

    if (s->useHwDecoder) {
        AVBufferSrcParametersPtr par(av_buffersrc_parameters_alloc());
        checkError(!par, exceptionMessage + "av_buffersrc_parameters_alloc");
        par->format = s->inputDecoderCtx->pix_fmt;
        par->time_base = timeBase;
        AVBufferRefPtr hwFramesRef;
        if (s->inputDecoderCtx->hw_frames_ctx)
            hwFramesRef.reset(av_buffer_ref(s->inputDecoderCtx->hw_frames_ctx));
        else if (s->inputDecoderCtx->hw_device_ctx) {
            AVBufferRef* rawFrames = av_hwframe_ctx_alloc(s->inputDecoderCtx->hw_device_ctx);
            checkError(!rawFrames, exceptionMessage + "av_hwframe_ctx_alloc");
            hwFramesRef.reset(rawFrames);
            AVHWFramesContext* frames_ctx = (AVHWFramesContext*)hwFramesRef->data;
            frames_ctx->format = s->inputDecoderCtx->pix_fmt;
            frames_ctx->sw_format = s->inputDecoderCtx->sw_pix_fmt;
            frames_ctx->width = s->inputDecoderCtx->width;
            frames_ctx->height = s->inputDecoderCtx->height;
            if (frames_ctx->sw_format == AV_PIX_FMT_NONE)
                frames_ctx->sw_format = AV_PIX_FMT_NV12;
            checkError(av_hwframe_ctx_init(hwFramesRef.get()) < 0, exceptionMessage + "av_hwframe_ctx_init");
        }
        if (hwFramesRef)
            par->hw_frames_ctx = hwFramesRef.get();
        checkError(av_buffersrc_parameters_set(srcCtx, par.get()) < 0, exceptionMessage + "av_buffersrc_parameters_set");
    }
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

int videoDispatcher(VideoSession* s, VideoMode op, const bool needsFilter) {
#if defined(_USE_CUDA_)
    // cache HDR info: read the video peak once and precompute the Mobius coefficients from it
    s->isHdr = isHDR(s->inputDecoderCtx.get());
    if (s->isHdr)
        s->mobius = MobiusParams::fromHdrPeak(getHdrPeak(s));
#endif
    // detect paths: no encoder, no thread, no queue
    if (op == VideoMode::DETECT) {
#if defined(_USE_CUDA_)
        if (s->useHwDecoder)
            return processFrames(s, needsFilter, [&](const AVFrame* frame, int& framesCount) { detectWatermarkHWAccel(s, framesCount, frame); });
#endif
        return processFrames(s, needsFilter, [&](const AVFrame* frame, int& framesCount) { detectWatermark(s, framesCount, frame); });
    }

    // embed paths: start a dedicated encode thread so decode+watermark (main) and
    // avcodec_send_frame+write (background) run concurrently, video frames and audio/subtitle packets all go in the queue
    EncodeQueue queue;
    std::exception_ptr encErr;
    std::thread encThread(encodeWorker, s, std::ref(queue), std::ref(encErr));
    int result = 0;
    try {
        s->auxMux.setSink([&queue](AVPacket* packet) {
            AVPacketPtr owned(av_packet_alloc());
            if (!owned || av_packet_ref(owned.get(), packet) < 0)
                return false;
            queue.push(std::move(owned));
            return true;
        });

#if defined(_USE_CUDA_)
        if (s->useHwDecoder)
            result = processFrames(s, needsFilter, [&](const AVFrame* frame, int& framesCount) { embedWatermarkHWAccel(s, framesCount, frame, queue); });
        else
#endif
            result = processFrames(s, needsFilter, [&](const AVFrame* frame, int& framesCount) { embedWatermark(s, framesCount, frame, queue); });
        queue.close(); // END: tell the encode thread to drain and exit
    } catch (...) {
        queue.abort(); // unblock encode thread if it is waiting
        encThread.join();
        throw;
    }
    encThread.join();
    if (encErr)
        std::rethrow_exception(encErr);
    return result;
}

void initOutputEncoder(VideoSession* s) {
    checkError(s->settings.encodeOutputPath.empty(), "No output path specified for video encode");
    checkError(sameFileOnDisk(s->settings.videoFile, s->settings.encodeOutputPath),
               "Output path points to the same physical file as the input file (" + s->settings.encodeOutputPath + "). Overwriting the input video is forbidden.");

    ParsedEncodeOptions parsed = parseEncodeOptions(s->settings.encodeOptions);
    const std::string& codecName = parsed.codecName;
    OptionDict opts(parsed.dictionary);
    const MediaLog log = [](const std::string& msg) { cout << info(msg + "\n"); };
    reportParsedOptions(parsed, log);

    if (codecName.empty())
        throw std::runtime_error("No video encoder in the options string, needs '-c:v <encoder>' (e.g. -c:v hevc_nvenc)");

    const AVCodec* encoder = avcodec_find_encoder_by_name(codecName.c_str());
    if (!encoder)
        throw std::runtime_error("Encoder not found: '" + codecName + "' (check encode_codec_options / hw_encode_options in settings.ini)");

    // backend and codec must agree: hw_encode_options must name a hardware encoder, encode_codec_options a software one
    const bool encIsHw = (encoder->capabilities & AV_CODEC_CAP_HARDWARE) != 0;
    if (s->settings.useHwEncoder && !encIsHw)
        throw std::runtime_error("cuda_hw_encoder is ON but '" + codecName +
                                 "' is a CPU (software) encoder. Use a hardware encoder like hevc_nvenc in hw_encode_options, or turn cuda_hw_encoder off.");
    if (!s->settings.useHwEncoder && encIsHw)
        throw std::runtime_error("cuda_hw_encoder is OFF but '" + codecName + "' is a hardware encoder. Use a software encoder like libx265 in encode_codec_options, or turn cuda_hw_encoder on.");

    // create the output muxer (format context)
    AVFormatContext* rawOutFmt = nullptr;
    checkAv(avformat_alloc_output_context2(&rawOutFmt, nullptr, nullptr, s->settings.encodeOutputPath.c_str()), "Failed to create output format context for: " + s->settings.encodeOutputPath);
    s->outputFormatCtx.reset(rawOutFmt);

    // fail fast on incompatible container/codec combos (like HEVC into AVI), which would mux into a broken file
    if (avformat_query_codec(s->outputFormatCtx->oformat, encoder->id, FF_COMPLIANCE_NORMAL) == 0)
        throw std::runtime_error("Codec '" + codecName + "' cannot be stored in the '" + std::string(s->outputFormatCtx->oformat->name) + "' container (" + s->settings.encodeOutputPath +
                                 "). Pick a container-compatible codec (prefer same container).");

    const int height = s->videoStream->codecpar->height;
    const int width = s->videoStream->codecpar->width;

    if (s->settings.useHwEncoder) {
        const int maxNvencRes = (codecName == "h264_nvenc") ? 4096 : 8192;
        checkError(width > maxNvencRes || height > maxNvencRes,
                   std::format("Output resolution ({}x{}) exceeds the NVENC hardware encoder limit of {}x{}.", width, height, maxNvencRes, maxNvencRes));
    }

    // build the encoder context
    AVCodecContextPtr encCtx(avcodec_alloc_context3(encoder));
    checkError(!encCtx, "Failed to allocate encoder context");

    encCtx->thread_count = 0;
    encCtx->width = width;
    encCtx->height = height;

    const bool useGpuPipeline = s->useHwDecoder && s->settings.useHwEncoder;
    encCtx->pix_fmt = useGpuPipeline ? AV_PIX_FMT_CUDA : AV_PIX_FMT_YUV420P;
    encCtx->time_base = s->videoStream->time_base;
    encCtx->framerate = s->videoStream->avg_frame_rate;
    encCtx->sample_aspect_ratio = s->videoStream->codecpar->sample_aspect_ratio;

    // if we tonemapped HDR->SDR write SDR metadata, not the original HDR flags
    const bool inputIsHDR = isHDR(s->inputDecoderCtx.get());
    encCtx->color_range = s->videoStream->codecpar->color_range;
    encCtx->color_primaries = inputIsHDR ? AVCOL_PRI_BT709 : s->inputDecoderCtx->color_primaries;
    encCtx->color_trc = inputIsHDR ? AVCOL_TRC_BT709 : s->inputDecoderCtx->color_trc;
    encCtx->colorspace = inputIsHDR ? AVCOL_SPC_BT709 : s->inputDecoderCtx->colorspace;
    if (s->outputFormatCtx->oformat->flags & AVFMT_GLOBALHEADER)
        encCtx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

#if defined(_USE_CUDA_)
    // NVDEC+NVENC: create hw_frames_ctx so NVENC reads CUDA NV12 frames directly from VRAM
    if (useGpuPipeline) {
        AVBufferRef* framesRef = av_hwframe_ctx_alloc(s->inputDecoderCtx->hw_device_ctx);
        checkError(!framesRef, "Failed to allocate hw_frames_ctx for NVENC");
        auto* fc = reinterpret_cast<AVHWFramesContext*>(framesRef->data);
        fc->format = AV_PIX_FMT_CUDA;
        fc->sw_format = AV_PIX_FMT_NV12;
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

    const AVPixelFormat wantFmt = useGpuPipeline ? AV_PIX_FMT_CUDA : AV_PIX_FMT_YUV420P;
    if (!encoderSupportsPixFmt(encoder, wantFmt))
        throw std::runtime_error(std::format("Encoder '{}' cannot encode {} ({} output). Pick another codec or change options.", codecName,
                                             av_get_pix_fmt_name(wantFmt) ? av_get_pix_fmt_name(wantFmt) : "?", useGpuPipeline ? "NVENC CUDA" : "YUV420P"));

    checkAv(avcodec_open2(encCtx.get(), encoder, opts.ptr()), "Failed to open encoder: " + codecName);

    AVStream* outVideoStream = avformat_new_stream(s->outputFormatCtx.get(), nullptr);
    checkError(!outVideoStream, "Failed to create output video stream");
    checkAv(avcodec_parameters_from_context(outVideoStream->codecpar, encCtx.get()), "Failed to copy encoder parameters to output video stream");
    outVideoStream->time_base = encCtx->time_base;
    if (useGpuPipeline)
        outVideoStream->codecpar->format = AV_PIX_FMT_YUV420P;
    if (!parsed.codecTag.empty())
        outVideoStream->codecpar->codec_tag = codecTagFromString(parsed.codecTag);

    // copy side data (rotation, display matrix) -> skip HDR metadata when tonemapped to SDR
    for (int i = 0; i < s->videoStream->codecpar->nb_coded_side_data; i++) {
        const AVPacketSideData& sd = s->videoStream->codecpar->coded_side_data[i];
        if (inputIsHDR && (sd.type == AV_PKT_DATA_MASTERING_DISPLAY_METADATA || sd.type == AV_PKT_DATA_CONTENT_LIGHT_LEVEL))
            continue;
        AVPacketSideData* dst = av_packet_side_data_new(&outVideoStream->codecpar->coded_side_data, &outVideoStream->codecpar->nb_coded_side_data, sd.type, sd.size, 0);
        if (dst)
            std::memcpy(dst->data, sd.data, sd.size);
    }
    // carry the video stream's tags and disposition, rotate is dropped
    av_dict_copy(&outVideoStream->metadata, s->videoStream->metadata, 0);
    av_dict_set(&outVideoStream->metadata, "rotate", nullptr, 0);
    outVideoStream->disposition = s->videoStream->disposition;
    s->outputVideoStreamIndex = outVideoStream->index;

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

    // open the output file and write the container header
    if (!(s->outputFormatCtx->oformat->flags & AVFMT_NOFILE))
        checkAv(avio_open(&s->outputFormatCtx->pb, s->settings.encodeOutputPath.c_str(), AVIO_FLAG_WRITE), "Failed to open output file: " + s->settings.encodeOutputPath);
    s->outputFormatCtx->max_interleave_delta = 0;

    // leftover unconsumed options (like -movflags +faststart) go to the container muxer
    const int headerRet = avformat_write_header(s->outputFormatCtx.get(), opts.ptr());
    if (headerRet >= 0)
        reportUnusedOptions(opts.get(), codecName, s->outputFormatCtx->oformat->name, log);
    checkAv(headerRet, "Failed to write output container header");

    s->outputEncoderCtx = std::move(encCtx);
    cout << info("Encoder: " + codecName + ": \"" + s->settings.encodeOutputPath + "\"\n\n");
}

void flushAndFinalize(VideoSession* s) {
    if (!s->outputEncoderCtx || !s->outputFormatCtx)
        return;
    // send EOS to the encoder, retry on EAGAIN
    int sendRet = avcodec_send_frame(s->outputEncoderCtx.get(), nullptr);
    while (sendRet == AVERROR(EAGAIN)) {
        drainEncoderPackets(s);
        sendRet = avcodec_send_frame(s->outputEncoderCtx.get(), nullptr);
    }
    checkError(sendRet < 0 && sendRet != AVERROR_EOF, "Failed to flush encoder: " + avErrorText(sendRet));
    drainEncoderPackets(s);
    checkAv(av_write_trailer(s->outputFormatCtx.get()), "Failed to write container trailer (output file may be truncated)");
}

} // namespace video_utils
