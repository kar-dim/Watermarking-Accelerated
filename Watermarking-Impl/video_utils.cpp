#include "buffer.hpp"
#include "common_utils.hpp"
#include "include/WatermarkCore.hpp"
#include "include/WatermarkTypes.hpp"
#include "video_defines.hpp"
#include "video_utils.hpp"
#include "VideoProcessingContext.hpp"
#include "WatermarkBase.hpp"
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <format>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

#if defined(_USE_CUDA_)
#include "CudaStreamManager.hpp"
#include "cuda_utils.hpp"
extern "C" {
#include "libavutil/buffer.h"
#include "libavutil/hwcontext_cuda.h"
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
#include "libavutil/avutil.h"
#include "libavutil/display.h"
#include "libavutil/error.h"
#include "libavutil/frame.h"
#include "libavutil/mem.h"
#include "libavutil/pixdesc.h"
#include "libavutil/pixfmt.h"
#include "libavutil/rational.h"
#include "libavutil/hwcontext.h"
}

#if defined(_USE_EIGEN_)
using namespace Eigen;
#endif

using namespace CommonUtils;
using namespace WatermarkCore;
using std::cout;
using std::string;

namespace video_utils {
#if defined(_USE_CUDA_)
// try to open a CUDA hardware accelerated decoder, if the user specified one, if it fails , fallback to a software decoder
AVCodecContextPtr openDecoderHWAccel(const AVCodecParameters* inputCodecParams, const string& userHwDecoder, bool& useHwDecoder) {
    if (userHwDecoder.empty())
        return openSoftwareDecoder(inputCodecParams);
    const AVCodec* inputDecoder = avcodec_find_decoder_by_name(userHwDecoder.c_str());
    if (!inputDecoder)
        return openSoftwareDecoder(inputCodecParams);
    AVCodecContextPtr ctx(avcodec_alloc_context3(inputDecoder));
    if (!ctx)
        return openSoftwareDecoder(inputCodecParams);
    if (avcodec_parameters_to_context(ctx.get(), inputCodecParams) < 0)
        return openSoftwareDecoder(inputCodecParams);
    AVBufferRef* raw_hw_device_ctx = nullptr;
    if (av_hwdevice_ctx_create(&raw_hw_device_ctx, AV_HWDEVICE_TYPE_CUDA, NULL, NULL, AV_CUDA_USE_CURRENT_CONTEXT) < 0)
        return openSoftwareDecoder(inputCodecParams);
    AVBufferRefPtr hw_device_ctx(raw_hw_device_ctx);
    if (av_hwdevice_ctx_init(hw_device_ctx.get()) < 0)
        return openSoftwareDecoder(inputCodecParams);
    ctx->hw_device_ctx = av_buffer_ref(hw_device_ctx.get());
    ctx->get_format = [](AVCodecContext*, const enum AVPixelFormat*) { return AV_PIX_FMT_CUDA; };
    if (avcodec_open2(ctx.get(), inputDecoder, nullptr) < 0)
        return openSoftwareDecoder(inputCodecParams);
    useHwDecoder = true;
    checkPixelFormatSupport(supportedHwFormats, ctx->sw_pix_fmt);
    return ctx;
}

// embed watermark in a video frame by using CUDA hardware acceleration
void embedWatermarkHWAccel(VideoSession* s, int& framesCount, const AVFrame* frame, FILE* ffmpegPipe) {
    const auto afStream = CudaStreamManager::getInstance().getAfStream();
    const auto videoStream = CudaStreamManager::getInstance().getCustomStream();
    const auto [height, width] = s->videoDims();
    const ImageBuffer lumaBuffer(height, width, f32);
    const ImageBuffer chromaBuffer(width, height / 2, u8);
    // launch NV12 to YUV420 kernel (for UV planes)
    cuda_utils::launchNV12ToYUV420pKernel(frame->data[1], frame->linesize[1], chromaBuffer.device<uint8_t>(), width / 2, height / 2, afStream);
    chromaBuffer.unlock();

    if (framesCount % s->settings.watermarkInterval == 0) {
        // overlap kernel and host copy
        cuda_utils::launchPitchedToFloatKernel(frame->data[0], lumaBuffer.device<float>(), width, height, frame->linesize[0], videoStream);
        chromaBuffer.host(s->hostFrame.get()->get() + (width * height));
        cudaStreamSynchronize(videoStream);
        // write Y + UV packed
        embedAndWriteFrame(s, lumaBuffer, width * height * 3 / 2, ffmpegPipe);
    } else {
        // try to overlap the two D2H copies
        cudaMemcpy2DAsync(s->hostFrame.get()->get(), width, frame->data[0], frame->linesize[0], width, height, cudaMemcpyDeviceToHost, videoStream);
        chromaBuffer.host(s->hostFrame.get()->get() + (width * height));
        cudaStreamSynchronize(videoStream);
        // write Y + UV packed
        fwrite(s->hostFrame.get()->get(), 1, width * height * 3 / 2, ffmpegPipe);
    }
    framesCount++;
}

// detect a watermark in a video frame using hardware acceleration
// directly use the GPU memory from the cuda decoder, no need to copy the data to host and back to GPU
void detectWatermarkHWAccel(VideoSession* s, int& framesCount, const AVFrame* frame) {
    // early exit, check if we should skip detection for this frame
    if (framesCount % s->settings.watermarkInterval != 0) {
        framesCount++;
        return;
    }
    // detect watermark after watermarkInterval frames
    const auto [height, width] = s->videoDims();
    const auto afStream = CudaStreamManager::getInstance().getAfStream();
    const ImageBuffer lumaBuffer(height, width, f32);
    cuda_utils::launchPitchedToFloatKernel(frame->data[0], lumaBuffer.device<float>(), width, height, frame->linesize[0], afStream);
    lumaBuffer.unlock();
    float correlation = s->watermarkObj->detectWatermark(lumaBuffer, MaskMethod::ME);
    cout << "Correlation for frame: " << (framesCount + 1) << ": " << correlation << "\n";
    framesCount++;
}
#endif

// get HDR info from the codec context in order to pass it to the filter graph for correct SDR tonemapping
// if 10-bit HDR video -> tonemap to SDR with ffmpeg CPU filters and convert to 8-bit
// if 10-bit SDR video -> convert to 8-bit fast
string getFilterGraphString(const VideoSession* s) {
    if (!is10bit(s->inputDecoderCtx.get(), s->videoStream))
        return ""; // 8-bit SDR, no filtering (save processing time)
    if (!isHDR(s->inputDecoderCtx.get()))
        return s->useHwDecoder ? "scale_cuda=format=nv12" : "format=yuv420p"; // 10-bit SDR, fast downscale to 8-bit
    // HDR10 / 10-bit HDR GPU case -> unfortunately no way to tonemap in GPU with cuda filters yet! Should use CPU decoder instead
    checkError(s->useHwDecoder, "Cannot tonemap HDR input to SDR with Hardware Accelerated Decoder yet. Use CPU decoder instead.");
    // HDR10 / 10-bit HDR CPU case -> scaler needs more input info
    const char* primaries = av_color_primaries_name(s->inputDecoderCtx.get()->color_primaries);
    const char* matrix = av_color_space_name(s->inputDecoderCtx.get()->colorspace);
    // fallback to safe HDR10 defaults if any field is unspecified
    if (!primaries || s->inputDecoderCtx.get()->color_primaries == AVCOL_PRI_UNSPECIFIED)
        primaries = "bt2020";
    if (!matrix || s->inputDecoderCtx.get()->colorspace == AVCOL_SPC_UNSPECIFIED)
        matrix = "bt2020nc";
    return std::format("zscale=primaries={}:transfer=linear:matrix={}:npl=100,tonemap=mobius,zscale=transfer=bt709:primaries=bt709:matrix=bt709,format=yuv420p", primaries, matrix);
}

// filter a single frame
void filterFrame(AVFramePtr& frame, AVFramePtr& filteredFrame, const VideoSession* s) {
    av_frame_unref(filteredFrame.get());
    int ret = av_buffersrc_add_frame_flags(s->buffersrcCtx, frame.get(), AV_BUFFERSRC_FLAG_KEEP_REF);
    checkError(ret < 0, "Failed to add frame to filter graph");
    ret = av_buffersink_get_frame(s->buffersinkCtx, filteredFrame.get());
    // don't exit if more frames (buffering is used) are needed to produce one output frame, just keep the original
    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
        return;
    checkError(ret < 0, "Failed to get filtered frame: " + std::to_string(ret));
    // replace original frame with filtered one
    av_frame_unref(frame.get());
    av_frame_move_ref(frame.get(), filteredFrame.get());
}

// load an uint8_t buffer into a GPU or Eigen buffer
void loadInputFrame(VideoSession* s, const uint8_t* hostPtr) {
    const auto [height, width] = s->videoDims();
#if defined(_USE_GPU_)
    s->inputFrame = Gray8Buffer(width, height, hostPtr, afHost).T().as(f32);
#else
    s->inputFrame = Map<const Gray8Buffer>(hostPtr, width, height).transpose().template cast<float>();
#endif
}

// initialize the filter graph for 10-bit to 8-bit conversion and HDR to SDR tonemapping
bool initFilterGraph(VideoSession* s) {
    const string exceptionMessage = "Failed to initialize filter graph in: ";
    const string filterDesc = getFilterGraphString(s);

    if (filterDesc.empty())
        return false; // no need for filtering

    // parse the filter graph description in order to initialize the filter graph later
    const AVRational timeBase = getTimeBase(s->videoStream);
    const char* pixFmtName = av_get_pix_fmt_name((AVPixelFormat)s->inputDecoderCtx->pix_fmt);
    string args = std::format("video_size={}x{}:pix_fmt={}:time_base={}/{}:pixel_aspect={}/{}", s->inputDecoderCtx->width, s->inputDecoderCtx->height, pixFmtName, timeBase.num, timeBase.den,
                              s->inputDecoderCtx->sample_aspect_ratio.num, s->inputDecoderCtx->sample_aspect_ratio.den);

    // allocate filter graph and source/sink filters
    AVFilterGraphPtr graphPtr(avfilter_graph_alloc());
    checkError(!graphPtr, exceptionMessage + "avfilter_graph_alloc");

    const AVFilter* bufferSrc = avfilter_get_by_name("buffer");
    const AVFilter* bufferSink = avfilter_get_by_name("buffersink");
    checkError(!bufferSrc || !bufferSink, exceptionMessage + "avfilter_get_by_name");

    AVFilterContext* srcCtx = avfilter_graph_alloc_filter(graphPtr.get(), bufferSrc, "in");
    checkError(!srcCtx, exceptionMessage + "avfilter_graph_alloc_filter");

    // if using hardware decoder, need to pass hw_frames_ctx to the source filter
    if (s->useHwDecoder) {
        AVBufferSrcParametersPtr par(av_buffersrc_parameters_alloc());
        checkError(!par, exceptionMessage + "av_buffersrc_parameters_alloc");
        par->format = s->inputDecoderCtx->pix_fmt;
        par->time_base = timeBase;

        AVBufferRefPtr hwFramesRef;
        // decoder provides a Frames Context (Ideal)
        if (s->inputDecoderCtx->hw_frames_ctx)
            hwFramesRef.reset(av_buffer_ref(s->inputDecoderCtx->hw_frames_ctx));
        // decoder only provides Device Context (must manually create hw_frames_ctx)
        else if (s->inputDecoderCtx->hw_device_ctx) {
            AVBufferRef* rawFrames = av_hwframe_ctx_alloc(s->inputDecoderCtx->hw_device_ctx);
            checkError(!rawFrames, exceptionMessage + "av_hwframe_ctx_alloc");
            hwFramesRef.reset(rawFrames);
            AVHWFramesContext* frames_ctx = (AVHWFramesContext*)hwFramesRef->data;
            frames_ctx->format = s->inputDecoderCtx->pix_fmt;       // AV_PIX_FMT_CUDA, etc
            frames_ctx->sw_format = s->inputDecoderCtx->sw_pix_fmt; // AV_PIX_FMT_P010, etc
            frames_ctx->width = s->inputDecoderCtx->width;
            frames_ctx->height = s->inputDecoderCtx->height;
            // fallback if sw_format is unknown
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

    // link and config in and out filters with the graph string
    AVFilterInOutPtr outputs(avfilter_inout_alloc());
    AVFilterInOutPtr inputs(avfilter_inout_alloc());
    checkError(!outputs || !inputs, exceptionMessage + "avfilter_inout_alloc");

    // maps to the Source (feeds into the graph)
    outputs->name = av_strdup("in");
    outputs->filter_ctx = srcCtx;
    outputs->pad_idx = 0;
    outputs->next = nullptr;
    // maps to the Sink (feeds out of the graph)
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

    // since we manually created source/sink, we just config
    checkError(avfilter_graph_config(graphPtr.get(), nullptr) < 0, exceptionMessage + "avfilter_graph_config");
    // store the filter graph pointers, give back control of graphPtr, we don't want to automatically delete it yet
    s->filterGraph = std::move(graphPtr);
    s->buffersrcCtx = srcCtx;
    s->buffersinkCtx = sinkCtx;

    return true;
}

// helper method to embed the watermark in the video frame and write it to the ffmpeg pipe
void embedAndWriteFrame(VideoSession* s, const ImageBuffer& buffer, const int elements, FILE* ffmpegPipe) {
    s->watermarkObj->makeWatermark(buffer, buffer, s->watermarkedFrame, MaskMethod::ME);
#if defined(_USE_GPU_)
    s->watermarkedFrame.T().host(s->hostFrame.get()->get());
    fwrite(s->hostFrame.get()->get(), 1, elements, ffmpegPipe);
#elif defined(_USE_EIGEN_)
    s->grayFrame = s->watermarkedFrame.getGray().transpose();
    fwrite(s->grayFrame.data(), 1, elements, ffmpegPipe);
#endif
}

// helper method to dispatch the correct watermarking or detection method
// clang-format off
int videoDispatcher(VideoSession* s, VideoMode op, const bool needsFilter, FILE* ffmpegPipe) {
#if defined(_USE_CUDA_)
    if (s->useHwDecoder)
        return needsFilter ? 
            processFrames<true>(s,[&](const AVFrame* frame, int& framesCount) {
                op == VideoMode::EMBED ? embedWatermarkHWAccel(s, framesCount, frame, ffmpegPipe) : detectWatermarkHWAccel(s, framesCount, frame);
            }) : 
            processFrames<false>(s, [&](const AVFrame* frame, int& framesCount) {
                op == VideoMode::EMBED ? embedWatermarkHWAccel(s, framesCount, frame, ffmpegPipe) : detectWatermarkHWAccel(s, framesCount, frame);
            });
#endif
    return needsFilter ? 
        processFrames<true>(s, [&](const AVFrame* frame,int& framesCount) { 
            op == VideoMode::EMBED ? embedWatermark(s, framesCount, frame, ffmpegPipe) : detectWatermark(s, framesCount, frame);
        }) : 
        processFrames<false>(s, [&](const AVFrame* frame, int& framesCount) {
            op == VideoMode::EMBED ? embedWatermark(s, framesCount, frame, ffmpegPipe) : detectWatermark(s, framesCount, frame);
        });
}
// clang-format on

// embed watermark in a video frame
void embedWatermark(VideoSession* s, int& framesCount, const AVFrame* frame, FILE* ffmpegPipe) {
    const bool embedWatermark = framesCount % s->settings.watermarkInterval == 0;
    processAndWriteYPlane(embedWatermark, frame, s, ffmpegPipe);
    writeChromaPlanes(frame, s, ffmpegPipe);
    framesCount++;
}

// detect the watermark for a video frame
void detectWatermark(VideoSession* s, int& framesCount, const AVFrame* frame) {
    // early exit, check if we should skip detection for this frame
    if (framesCount % s->settings.watermarkInterval != 0) {
        framesCount++;
        return;
    }
    const auto [height, width] = s->videoDims();
    // detect watermark after watermarkInterval frames, else early return
    uint8_t* srcY = frame->data[0];
    // if there is row padding (for alignment), we must copy the data to a contiguous block!
    if (frame->linesize[0] != width) {
        for (int y = 0; y < height; y++)
            memcpy(s->hostFrame.get()->get() + y * width, frame->data[0] + y * frame->linesize[0], width);
        srcY = s->hostFrame.get()->get();
    }
    loadInputFrame(s, srcY);
    float correlation = s->watermarkObj->detectWatermark(s->inputFrame, MaskMethod::ME);
    cout << "Correlation for frame: " << (framesCount + 1) << ": " << correlation << "\n";
    framesCount++;
}

// find the first video stream index
int findVideoStream(const AVFormatContext* inputFormatCtx) {
    for (unsigned int i = 0; i < inputFormatCtx->nb_streams; i++)
        if (inputFormatCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO)
            return i;
    return -1;
}

// get the rotation angle of the video stream (if any) to apply as a filter in the encoder
string getStreamRotation(const AVStream* st) {
    if (!st || !st->codecpar)
        return "";
    for (int i = 0; i < st->codecpar->nb_coded_side_data; i++) {
        const AVPacketSideData& sd = st->codecpar->coded_side_data[i];
        if (sd.type == AV_PKT_DATA_DISPLAYMATRIX && sd.data && sd.size >= 9 * sizeof(int32_t)) {
            const double rotation = -av_display_rotation_get(reinterpret_cast<const int32_t*>(sd.data));
            const int angle = (static_cast<int>(std::lrint(rotation)) + 360) % 360; // normalize to [0, 360)
            switch (angle) {
            case 90: return "-vf \"transpose=1\" ";
            case 180: return "-vf \"hflip,vflip\" ";
            case 270: return "-vf \"transpose=2\" ";
            default: return "";
            }
        }
    }
    return "";
}

// if CUDA, try to open hw decoder (if requested), else fallback to open software decoder context for video
// otherwise, just open software decoder
AVCodecContextPtr openDecoder(const AVCodecParameters* inputCodecParams, const string& userHwDecoder, bool& useHwDecoder) {
#if defined(_USE_CUDA_)
    return openDecoderHWAccel(inputCodecParams, userHwDecoder, useHwDecoder);
#else
    return openSoftwareDecoder(inputCodecParams);
#endif
}

// open software decoder context for video
AVCodecContextPtr openSoftwareDecoder(const AVCodecParameters* inputCodecParams) {
    const AVCodec* inputDecoder = avcodec_find_decoder(inputCodecParams->codec_id);
    if (!inputDecoder)
        return nullptr;
    AVCodecContextPtr ctx(avcodec_alloc_context3(inputDecoder));
    if (!ctx)
        return nullptr;
    if (avcodec_parameters_to_context(ctx.get(), inputCodecParams) < 0)
        return nullptr;
    // multithreading decode
    ctx->thread_count = 0;
    if (inputDecoder->capabilities & AV_CODEC_CAP_FRAME_THREADS)
        ctx->thread_type = FF_THREAD_FRAME;
    else if (inputDecoder->capabilities & AV_CODEC_CAP_SLICE_THREADS)
        ctx->thread_type = FF_THREAD_SLICE;
    else
        ctx->thread_count = 1; // don't use multithreading
    if (avcodec_open2(ctx.get(), inputDecoder, nullptr) < 0)
        return nullptr;
    checkPixelFormatSupport(supportedFormats, ctx->pix_fmt);
    return ctx;
}

// check if the pixel format provided is in the list of provided supported formats
bool checkPixelFormatSupport(const std::span<const AVPixelFormat> formats, const AVPixelFormat format) {
    const bool isValidFormat = std::ranges::any_of(formats, [&](auto f) { return f == format; });
    checkError(!isValidFormat, "Error: Video frame format not supported, aborting");
    return isValidFormat;
}

// get the input video FPS (average)
string getFrameRate(const AVStream* st) {
    const AVRational frameRate = st->avg_frame_rate;
    return std::format("{:.3f}", static_cast<float>(frameRate.num) / frameRate.den);
}

// get the input video time base (should be 1/average(fps))
AVRational getTimeBase(const AVStream* st) {
    const AVRational fps = st->avg_frame_rate;
    if (fps.num > 0 && fps.den > 0)
        return av_inv_q(fps);
    // fallback if avg_frame_rate is garbage
    if (st->time_base.num > 0 && st->time_base.den > 0)
        return st->time_base;
    return AVRational{1, 30};
}

// runs the watermark creation for a video frame and writes the watermarked frame to the ffmpeg pipe
void processAndWriteYPlane(const bool embedWatermark, const AVFrame* frame, VideoSession* s, FILE* ffmpegPipe) {
    const auto [height, width] = s->videoDims();
    uint8_t* srcY = frame->data[0];
    // if there is row padding (for alignment), we must copy the data to a contiguous block!
    if (frame->linesize[0] != width) {
        for (int y = 0; y < height; y++)
            memcpy(s->hostFrame.get()->get() + y * width, srcY + y * frame->linesize[0], width);
        srcY = s->hostFrame.get()->get();
    }
    if (embedWatermark) {
        loadInputFrame(s, srcY);
        embedAndWriteFrame(s, s->inputFrame, width * height, ffmpegPipe);
    } else
        fwrite(srcY, 1, width * height, ffmpegPipe);
}

// writes the chroma planes (U and V) to the ffmpeg pipe, either assuming aligned pointers or not
void writeChromaPlanes(const AVFrame* frame, VideoSession* s, FILE* ffmpegPipe) {
    const auto [height, width] = s->videoDims();
    // lambda to write a single chroma plane
    auto writePlane = [&](const uint8_t* src, const int linesize) {
        if (linesize != width / 2) {
            for (int y = 0; y < height / 2; y++)
                fwrite(src + y * linesize, 1, width / 2, ffmpegPipe);
        } else
            fwrite(src, 1, width * height / 4, ffmpegPipe);
    };
    // write U
    writePlane(frame->data[1], frame->linesize[1]);
    // write V
    writePlane(frame->data[2], frame->linesize[2]);
}
} // namespace video_utils