#include "buffer.hpp"
#include "common_utils.hpp"
#include "include/WatermarkCore.hpp"
#include "include/WatermarkTypes.hpp"
#include "video_defines.hpp"
#include "video_utils.hpp"
#include "VideoProcessingContext.hpp"
#include "WatermarkBase.hpp"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <format>
#include <iostream>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(_USE_CUDA_)
#include "CudaArray.hpp"
#include "CudaStreamManager.hpp"
#include "cuda_utils.hpp"
#include <cuda.h>
extern "C" {
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

namespace {
constexpr AVPixelFormat supportedFormats[] = {AV_PIX_FMT_YUV420P, AV_PIX_FMT_YUVJ420P, AV_PIX_FMT_YUV420P10LE, AV_PIX_FMT_CUDA};
#if defined(_USE_CUDA_)
constexpr AVPixelFormat supportedHwFormats[] = {AV_PIX_FMT_NV12, AV_PIX_FMT_P010LE, AV_PIX_FMT_P016LE};
#endif
} // namespace

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
    // manually create the hw device context to share our existing primary CUDA context
    AVBufferRef* raw_hw_device_ctx = av_hwdevice_ctx_alloc(AV_HWDEVICE_TYPE_CUDA);
    if (!raw_hw_device_ctx)
        return openSoftwareDecoder(inputCodecParams);
    AVBufferRefPtr hw_device_ctx(raw_hw_device_ctx);
    auto* hwCtx = reinterpret_cast<AVHWDeviceContext*>(hw_device_ctx->data);
    auto* cudaCtx = reinterpret_cast<AVCUDADeviceContext*>(hwCtx->hwctx);
    int runtimeDevice = 0;
    cudaGetDevice(&runtimeDevice);
    CUdevice cuDev = 0;
    if (cuDeviceGet(&cuDev, runtimeDevice) != CUDA_SUCCESS)
        return openSoftwareDecoder(inputCodecParams);
    CUcontext primaryCtx = nullptr;
    if (cuDevicePrimaryCtxRetain(&primaryCtx, cuDev) != CUDA_SUCCESS)
        return openSoftwareDecoder(inputCodecParams);
    cudaCtx->cuda_ctx = primaryCtx;
    hwCtx->user_opaque = reinterpret_cast<void*>(static_cast<intptr_t>(cuDev));
    hwCtx->free = [](AVHWDeviceContext* c) {
        const CUdevice d = static_cast<CUdevice>(reinterpret_cast<intptr_t>(c->user_opaque));
        cuDevicePrimaryCtxRelease(d);
    };
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
void embedWatermarkHWAccel(VideoSession* s, int& framesCount, const AVFrame* frame) {
    const auto stream = CudaStreamManager::getInstance().getComputeStream();
    const auto [height, width] = s->videoDims();
    CudaArray<uint8_t> chromaBuffer(width * height / 2, stream);
    // convert NV12 chroma -> YUV420P planar and download async to hostFrame[W*H..]
    cuda_utils::launchNV12ToYUV420pKernel(frame->data[1], frame->linesize[1], chromaBuffer.data(), width / 2, height / 2, stream);
    chromaBuffer.toHostAsync(s->hostFrame->get() + width * height);

    if (framesCount % s->settings.watermarkInterval == 0) {
        cout << std::format(" [Embedding frame {}]\n", framesCount + 1);
        CudaArray<float> lumaBuffer(height, width, stream);
        cuda_utils::launchPitchedToFloatKernel(frame->data[0], lumaBuffer.data(), width, height, frame->linesize[0], stream);
        cudaStreamSynchronize(stream);     // UV async download + luma float conversion complete
        embedAndFillYPlane(s, lumaBuffer); // watermark, download Y to hostFrame[0..W*H-1]
    } else {
        // no watermark, copy luma directly to hostFrame[0..W*H-1]
        cudaMemcpy2DAsync(s->hostFrame->get(), width, frame->data[0], frame->linesize[0], width, height, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
    }
    // hostFrame now contains complete YUV420P, send to encoder (NVENC async, returns fast)
    encodeFrame(s, frame->pts);
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
    const auto stream = CudaStreamManager::getInstance().getComputeStream();
    CudaArray<float> lumaBuffer(height, width, stream);
    cuda_utils::launchPitchedToFloatKernel(frame->data[0], lumaBuffer.data(), width, height, frame->linesize[0], stream);
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
    // throw if filter graph has buffered frames or it died somehow, we strictly want 1-to-1 frame relation (scaling and tonemap are always 1-to-1, but let's make sure)
    checkError(ret == AVERROR(EAGAIN) || ret == AVERROR_EOF, "Filter graph buffered the frame. 1-to-1 filtering strictly required!");
    checkError(ret < 0, "Failed to get filtered frame: " + std::to_string(ret));
    // replace original frame with filtered one
    av_frame_unref(frame.get());
    av_frame_move_ref(frame.get(), filteredFrame.get());
}

// load an uint8_t buffer into a GPU or Eigen buffer
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

// initialize the filter graph for 10-bit to 8-bit conversion and HDR to SDR tonemapping
bool initFilterGraph(VideoSession* s) {
    const string exceptionMessage = "Failed to initialize filter graph in: ";
    const string filterDesc = getFilterGraphString(s);

    if (filterDesc.empty())
        return false; // no need for filtering

    // timeBase is CRITICAL for VFR, pass as is
    const AVRational timeBase = s->videoStream->time_base;
    const char* pixFmtName = av_get_pix_fmt_name((AVPixelFormat)s->inputDecoderCtx->pix_fmt);
    string args = std::format("video_size={}x{}:pix_fmt={}:time_base={}/{}:pixel_aspect={}/{}", s->inputDecoderCtx->width, s->inputDecoderCtx->height, pixFmtName, timeBase.num, timeBase.den,
                              s->inputDecoderCtx->sample_aspect_ratio.num, s->inputDecoderCtx->sample_aspect_ratio.den);

    // allocate filter graph and source/sink filters (multi-threaded by default for CPU filters like tonemap)
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

// watermark the frame and fill hostFrame[0..W*H-1] with the processed Y plane
void embedAndFillYPlane(VideoSession* s, const ImageBuffer& buffer) {
    s->watermarkObj->makeWatermark(buffer, buffer, s->watermarkedFrame, MaskMethod::ME);
#if defined(_USE_CUDA_)
    {
        const auto stream = CudaStreamManager::getInstance().getComputeStream();
        CudaArray<uint8_t> rowMajorOut(s->watermarkedFrame.getRows(), s->watermarkedFrame.getCols(), stream);
        cuda_utils::launchColMajorToRowMajorU8Kernel(s->watermarkedFrame.data(), rowMajorOut.data(), static_cast<int>(s->watermarkedFrame.getCols()), static_cast<int>(s->watermarkedFrame.getRows()),
                                                     1, stream);
        rowMajorOut.toHost(s->hostFrame->get()); // fills hostFrame[0..W*H-1]
    }
#elif defined(_USE_OPENCL_)
    {
        const auto [height, width] = s->videoDims();
        auto& mgr = OclQueueManager::getInstance();
        OclArray<uint8_t> rowMajorOut(height, width, mgr.getQueueRaw());
        cl_utils::launchColMajorToRowMajorU8(s->watermarkedFrame.clBuffer(), rowMajorOut.clBuffer(), width, height, 1, mgr.getQueue());
        rowMajorOut.toHost(s->hostFrame->get()); // fills hostFrame[0..W*H-1]
    }
#elif defined(_USE_EIGEN_)
    {
        const auto [height, width] = s->videoDims();
        s->grayFrame = s->watermarkedFrame.getGray().transpose();
        std::memcpy(s->hostFrame->get(), s->grayFrame.data(), static_cast<size_t>(width) * height);
    }
#endif
}

// dispatch the correct watermarking or detection method for each frame
// clang-format off
int videoDispatcher(VideoSession* s, VideoMode op, const bool needsFilter) {
#if defined(_USE_CUDA_)
    if (s->useHwDecoder)
        return needsFilter ?
            processFrames<true>(s, [&](const AVFrame* frame, int& framesCount) {
                op == VideoMode::EMBED ? embedWatermarkHWAccel(s, framesCount, frame) : detectWatermarkHWAccel(s, framesCount, frame);
            }) :
            processFrames<false>(s, [&](const AVFrame* frame, int& framesCount) {
                op == VideoMode::EMBED ? embedWatermarkHWAccel(s, framesCount, frame) : detectWatermarkHWAccel(s, framesCount, frame);
            });
#endif
    return needsFilter ?
        processFrames<true>(s, [&](const AVFrame* frame, int& framesCount) {
            op == VideoMode::EMBED ? embedWatermark(s, framesCount, frame) : detectWatermark(s, framesCount, frame);
        }) :
        processFrames<false>(s, [&](const AVFrame* frame, int& framesCount) {
            op == VideoMode::EMBED ? embedWatermark(s, framesCount, frame) : detectWatermark(s, framesCount, frame);
        });
}
// clang-format on

// embed watermark in a video frame (software decode path)
void embedWatermark(VideoSession* s, int& framesCount, const AVFrame* frame) {
    const bool doEmbed = framesCount % s->settings.watermarkInterval == 0;
    if (doEmbed)
        cout << std::format(" [Embedding frame {}]\n", framesCount + 1);
    fillYPlane(doEmbed, frame, s);
    fillChromaPlanes(frame, s);
    encodeFrame(s, frame->pts);
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
            std::memcpy(s->hostFrame.get()->get() + y * width, frame->data[0] + y * frame->linesize[0], width);
        srcY = s->hostFrame.get()->get();
    }
    loadInputFrame(s, srcY);
    const float correlation = s->watermarkObj->detectWatermark(s->inputFrame, MaskMethod::ME);
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

// if CUDA, try to open hw decoder (if requested), else fallback to open software decoder context for video, else just open software decoder
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

// fill hostFrame[0..W*H-1] with the Y plane (watermarked or passthrough)
void fillYPlane(const bool doEmbed, const AVFrame* frame, VideoSession* s) {
    const auto [height, width] = s->videoDims();
    uint8_t* hostY = s->hostFrame->get();
    const uint8_t* srcY = frame->data[0];
    // destride into hostFrame if the decoder added alignment padding
    if (frame->linesize[0] != width) {
        for (int y = 0; y < height; y++)
            std::memcpy(hostY + y * width, srcY + y * frame->linesize[0], width);
        srcY = hostY;
    }
    if (doEmbed) {
        loadInputFrame(s, srcY);
        embedAndFillYPlane(s, s->inputFrame);
    } else if (srcY != hostY) {
        // no watermark, frame was not destrided: copy Y directly into hostFrame
        std::memcpy(hostY, srcY, static_cast<size_t>(width) * height);
    }
}

// fill hostFrame[W*H..W*H*3/2-1] with U and V planes (de-stride if needed)
void fillChromaPlanes(const AVFrame* frame, VideoSession* s) {
    const auto [height, width] = s->videoDims();
    const int chromaHeight = height / 2;
    const int chromaWidth = width / 2;
    uint8_t* uDst = s->hostFrame->get() + width * height; // U starts right after Y
    uint8_t* vDst = uDst + chromaWidth * chromaHeight;    // V follows U
    // lambda: copy one chroma plane, de-striding if the decoder added alignment padding
    auto copyPlane = [&](const uint8_t* src, const int linesize, uint8_t* dst) {
        if (linesize != chromaWidth) {
            for (int y = 0; y < chromaHeight; y++)
                std::memcpy(dst + y * chromaWidth, src + y * linesize, chromaWidth);
        } else {
            std::memcpy(dst, src, static_cast<size_t>(chromaWidth) * chromaHeight);
        }
    };
    copyPlane(frame->data[1], frame->linesize[1], uDst);
    copyPlane(frame->data[2], frame->linesize[2], vDst);
}

// parse "-c:v codec -key val ..." into {codecName, AVDictionary*}
// strips stream specifiers (:v, :a, :s) so AVOptions names match libav
static std::pair<std::string, AVDictionary*> parseEncodeOptions(const std::string& optStr) {
    std::string codecName;
    AVDictionary* opts = nullptr;

    auto stripStreamSpec = [](std::string key) {
        const auto pos = key.rfind(':');
        if (pos != std::string::npos && pos + 1 < key.size()) {
            const char spec = key[pos + 1];
            if (spec == 'v' || spec == 'a' || spec == 's' || spec == 'd' || spec == 't')
                key.erase(pos);
        }
        return key;
    };

    std::istringstream ss(optStr);
    std::vector<std::string> tokens;
    for (std::string tok; ss >> tok;)
        tokens.push_back(std::move(tok));

    for (size_t i = 0; i < tokens.size();) {
        if (tokens[i].size() > 1 && tokens[i][0] == '-') {
            const std::string key = stripStreamSpec(tokens[i].substr(1));
            if (i + 1 < tokens.size() && (tokens[i + 1].empty() || tokens[i + 1][0] != '-')) {
                const std::string& val = tokens[i + 1];
                if (key == "c" || key == "codec")
                    codecName = val;
                else
                    av_dict_set(&opts, key.c_str(), val.c_str(), 0);
                i += 2;
                continue;
            }
        }
        ++i;
    }
    return {codecName, opts};
}

// pull ready packets from the encoder and write them to the output file
static void drainEncoderPackets(VideoSession* s) {
    const AVPacketPtr pkt(av_packet_alloc());
    const AVStream* outVideoStream = s->outputFormatCtx->streams[s->outputVideoStreamIndex];
    while (avcodec_receive_packet(s->outputEncoderCtx.get(), pkt.get()) == 0) {
        av_packet_rescale_ts(pkt.get(), s->outputEncoderCtx->time_base, outVideoStream->time_base);
        pkt->stream_index = s->outputVideoStreamIndex;
        checkError(av_interleaved_write_frame(s->outputFormatCtx.get(), pkt.get()) < 0, "Failed to write encoded video packet");
        av_packet_unref(pkt.get());
    }
    // AVERROR(EAGAIN) = encoder needs more frames before producing output, (we ignore it, keep going)
}

// initialize the output encoder (AVCodecContext) and muxer (AVFormatContext)
// parses encodeOptions like "-c:v hevc_nvenc -preset p6 -cq 26" or "-c:v libx265 -preset fast -crf 23"
// audio and subtitle streams are remuxed (copied) from the input, rotation metadata is preserved as display matrix side data
void initOutputEncoder(VideoSession* s) {
    checkError(s->settings.encodeOutputPath.empty(), "No output path specified for video encode");

    auto [codecName, rawOpts] = parseEncodeOptions(s->settings.encodeOptions);
    AVDictionary* opts = rawOpts; // avcodec_open2 will consume recognized entries
    const AVCodec* encoder = avcodec_find_encoder_by_name(codecName.c_str());
    if (!encoder) {
        av_dict_free(&opts);
        throw std::runtime_error("Encoder not found: " + codecName + " (check encode_codec_options / hw_encode_options in settings.ini)");
    }
    // create the output muxer (format context)
    AVFormatContext* rawOutFmt = nullptr;
    checkError(avformat_alloc_output_context2(&rawOutFmt, nullptr, nullptr, s->settings.encodeOutputPath.c_str()) < 0, "Failed to create output format context for: " + s->settings.encodeOutputPath);
    s->outputFormatCtx.reset(rawOutFmt);

    // build the encoder context
    AVCodecContextPtr encCtx(avcodec_alloc_context3(encoder));
    checkError(!encCtx, "Failed to allocate encoder context");

    const int height = s->videoStream->codecpar->height;
    const int width = s->videoStream->codecpar->width;
    encCtx->width = width;
    encCtx->height = height;
    encCtx->pix_fmt = AV_PIX_FMT_YUV420P;          // hostFrame is always YUV420P planar
    encCtx->time_base = s->videoStream->time_base; // preserves original PTS for VFR
    encCtx->framerate = s->videoStream->avg_frame_rate;
    encCtx->sample_aspect_ratio = s->videoStream->codecpar->sample_aspect_ratio;
    // if we tonemapped HDR->SDR the pixels are now BT.709, so we MUST write SDR metadata instead of the original HDR flags
    // isHDR checks color_trc for PQ (SMPTE 2084) or HLG, those are only set on true HDR input
    const bool inputIsHDR = isHDR(s->inputDecoderCtx.get());
    encCtx->color_range = s->videoStream->codecpar->color_range;
    encCtx->color_primaries = inputIsHDR ? AVCOL_PRI_BT709 : s->inputDecoderCtx->color_primaries;
    encCtx->color_trc = inputIsHDR ? AVCOL_TRC_BT709 : s->inputDecoderCtx->color_trc;
    encCtx->colorspace = inputIsHDR ? AVCOL_SPC_BT709 : s->inputDecoderCtx->colorspace;
    // MP4/MOV containers require the extradata to be embedded in a global header
    if (s->outputFormatCtx->oformat->flags & AVFMT_GLOBALHEADER)
        encCtx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

    // open encoder, avcodec_open2 consumes recognised entries from opts
    const int openRet = avcodec_open2(encCtx.get(), encoder, &opts);
    av_dict_free(&opts);
    checkError(openRet < 0, "Failed to open encoder: " + codecName);
    // add streams to the output container
    s->inputToOutputStreamMap.assign(s->inputFormatCtx->nb_streams, -1);
    // video stream (encoded)
    AVStream* outVideoStream = avformat_new_stream(s->outputFormatCtx.get(), nullptr);
    checkError(!outVideoStream, "Failed to create output video stream");
    checkError(avcodec_parameters_from_context(outVideoStream->codecpar, encCtx.get()) < 0, "Failed to copy encoder parameters to output video stream");
    outVideoStream->time_base = encCtx->time_base;
    // copy side data (display matrix / rotation, etc) from the input stream
    // when we tonemapped HDR->SDR, skip HDR entries, the pixel data is already SDR
    // leaving those in causes video players to tonemap again!
    for (int i = 0; i < s->videoStream->codecpar->nb_coded_side_data; i++) {
        const AVPacketSideData& sd = s->videoStream->codecpar->coded_side_data[i];
        if (inputIsHDR && (sd.type == AV_PKT_DATA_MASTERING_DISPLAY_METADATA || sd.type == AV_PKT_DATA_CONTENT_LIGHT_LEVEL))
            continue;
        AVPacketSideData* dst = av_packet_side_data_new(&outVideoStream->codecpar->coded_side_data, &outVideoStream->codecpar->nb_coded_side_data, sd.type, sd.size, 0);
        if (dst)
            std::memcpy(dst->data, sd.data, sd.size);
    }
    s->inputToOutputStreamMap[s->videoStreamIndex] = outVideoStream->index;
    s->outputVideoStreamIndex = outVideoStream->index;

    // audio + subtitle streams are REMUXED as is (similar to map in ffmpeg cli)
    for (unsigned i = 0; i < s->inputFormatCtx->nb_streams; i++) {
        const AVStream* inSt = s->inputFormatCtx->streams[i];
        const AVMediaType type = inSt->codecpar->codec_type;
        if (type != AVMEDIA_TYPE_AUDIO && type != AVMEDIA_TYPE_SUBTITLE)
            continue;
        AVStream* outSt = avformat_new_stream(s->outputFormatCtx.get(), nullptr);
        if (!outSt)
            continue; // non-fatal: skip this stream
        if (avcodec_parameters_copy(outSt->codecpar, inSt->codecpar) < 0)
            continue;
        outSt->time_base = inSt->time_base;
        s->inputToOutputStreamMap[i] = outSt->index;
    }

    // open the output file and write the container header
    if (!(s->outputFormatCtx->oformat->flags & AVFMT_NOFILE)) {
        checkError(avio_open(&s->outputFormatCtx->pb, s->settings.encodeOutputPath.c_str(), AVIO_FLAG_WRITE) < 0, "Failed to open output file: " + s->settings.encodeOutputPath);
    }
    s->outputFormatCtx->max_interleave_delta = 0; // same as -max_interleave_delta 0 in the ffmpeg cli
    AVDictionary* hdrOpts = nullptr;
    checkError(avformat_write_header(s->outputFormatCtx.get(), &hdrOpts) < 0, "Failed to write output container header");
    av_dict_free(&hdrOpts);

    s->outputEncoderCtx = std::move(encCtx);
    cout << info("Encoder: " + codecName + ": \"" + s->settings.encodeOutputPath + "\"\n\n");
}

// wrap the current hostFrame (YUV420P planar, W*H*3/2 bytes) in an AVFrame and submit it
// to the encoder, for NVENC this returns almost immediately (async),
// for software encoders the internal frame-thread pool handles parallelism
// non-blocking drain -> we pull whatever packets the encoder has already finished.
void encodeFrame(VideoSession* s, const int64_t pts) {
    const auto [height, width] = s->videoDims();
    const uint8_t* src = s->hostFrame->get();

    AVFramePtr encFrame(av_frame_alloc());
    checkError(!encFrame, "Failed to allocate encoder AVFrame");
    encFrame->format = AV_PIX_FMT_YUV420P;
    encFrame->width = width;
    encFrame->height = height;
    encFrame->pts = pts; // pass original decoded PTS -> VFR preserved end-to-end, important!
    checkError(av_frame_get_buffer(encFrame.get(), 0) < 0, "Failed to allocate encoder frame buffer");

    // copy YUV planes
    for (int y = 0; y < height; y++)
        std::memcpy(encFrame->data[0] + y * encFrame->linesize[0], src + y * width, width);
    for (int y = 0; y < height / 2; y++) {
        std::memcpy(encFrame->data[1] + y * encFrame->linesize[1], src + width * height + y * (width / 2), width / 2);
        std::memcpy(encFrame->data[2] + y * encFrame->linesize[2], src + width * height * 5 / 4 + y * (width / 2), width / 2);
    }

    const int ret = avcodec_send_frame(s->outputEncoderCtx.get(), encFrame.get());
    checkError(ret < 0 && ret != AVERROR(EAGAIN), "Encoder send_frame failed");
    drainEncoderPackets(s);
}

// flush the encoder pipeline and finalise the output container, called once after all input frames have been processed
void flushAndFinalize(VideoSession* s) {
    if (!s->outputEncoderCtx || !s->outputFormatCtx)
        return;
    // send flush signal, encoder will drain its internal lookahead/B-frame buffers
    avcodec_send_frame(s->outputEncoderCtx.get(), nullptr);
    drainEncoderPackets(s);                     // drain all remaining encoded packets
    av_write_trailer(s->outputFormatCtx.get()); // write container trailer (MOOV atom for MP4, etc etc)
}

} // namespace video_utils