#include "AvUtil.hpp"
#include "common_utils.hpp"
#include "HostMemory.hpp"
#include "video_defines.hpp"
#include "video_utils.hpp"
#include "VideoPreviewParams.hpp"
#include "VideoProcessingContext.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <format>
#include <iostream>
#include <memory>
#include <string>

#if defined(_USE_CUDA_)
#include "cuda_utils.hpp"
#include "CudaArray.hpp"
#include "CudaCheck.hpp"
#include "CudaStreamManager.hpp"
#elif defined(_USE_OPENCL_)
#include "OclQueueManager.hpp"
#include "OclVideoInterop.hpp"
#include "opencl_utils.hpp"
#endif

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavcodec/packet.h>
#include "libavutil/avutil.h"
#include <libavutil/hwcontext.h>
#include <libavutil/pixdesc.h>
#include <libswscale/swscale.h>
}

/*!
 * \brief Video progress reporting and sampled CPU/GPU preview conversion
 * \author Dimitris Karatzas
 */
namespace WatermarkCore {
struct VideoPreview {
    std::chrono::steady_clock::time_point last{};
    bool disabled = false;
    int width = 0, height = 0;
    SwsContext* scaler = nullptr;
    std::unique_ptr<HostMemory<uint8_t>> host;
#if defined(_USE_CUDA_)
    CudaArray<uint8_t> pixels;
#elif defined(_USE_OPENCL_)
    cl::Buffer pixels;
    cl::Kernel kernel;
    std::unique_ptr<OclVideoFrame> source;
#endif
    ~VideoPreview() { sws_freeContext(scaler); }
};

VideoSession::VideoSession() = default;
VideoSession::~VideoSession() = default;
} // namespace WatermarkCore

namespace video_utils {
using namespace WatermarkCore;

void checkCancelled(const VideoSession* s) {
    if (s->settings.stopToken.stop_requested())
        throw VideoCancelled();
}

int interruptVideo(void* session) { return static_cast<VideoSession*>(session)->settings.stopToken.stop_requested() ? 1 : 0; }

void videoLog(const VideoSession* s, const std::string& text) {
    if (s->settings.observer)
        s->settings.observer->onLog(text);
    else
        std::cout << CommonUtils::info(text);
}

void reportProgress(VideoSession* s, const VideoState state, const bool force, const AVFrame* frame) {
    if (!s->settings.observer)
        return;
    const auto now = std::chrono::steady_clock::now();
    auto& progress = s->progress;
    progress.state = state;
    // Keep a timestamp origin for streams that do not report a start time
    if (frame && s->progressOrigin == AV_NOPTS_VALUE)
        s->progressOrigin = frame->best_effort_timestamp != AV_NOPTS_VALUE ? frame->best_effort_timestamp : frame->pts;
    if (!force && now - s->lastProgress < std::chrono::milliseconds(150))
        return;
    s->lastProgress = now;
    progress.elapsedSeconds = std::chrono::duration<double>(now - s->started).count();
    progress.fps = progress.elapsedSeconds > 0.0 ? progress.frames / progress.elapsedSeconds : 0.0;
    if (s->videoStream && s->inputDecoderCtx) {
        const auto* stream = s->videoStream;
        const auto* descriptor = av_pix_fmt_desc_get(static_cast<AVPixelFormat>(stream->codecpar->format));
        progress.source = std::format("{} x {}  |  {}  |  {}{}", stream->codecpar->width, stream->codecpar->height, avcodec_get_name(stream->codecpar->codec_id),
            descriptor ? descriptor->name : "unknown format", isHDR(s->inputDecoderCtx.get()) ? " HDR" : " SDR");
        if (descriptor)
            progress.source += std::format("  |  {}-bit", descriptor->comp[0].depth);
        const auto* transform = av_packet_side_data_get(stream->codecpar->coded_side_data, stream->codecpar->nb_coded_side_data, AV_PKT_DATA_DISPLAYMATRIX);
        if (transform && transform->size >= 9 * sizeof(int32_t)) {
            const auto* matrix = reinterpret_cast<const int32_t*>(transform->data);
            const double horizontal = std::hypot(static_cast<double>(matrix[0]), static_cast<double>(matrix[1]));
            const double vertical = std::hypot(static_cast<double>(matrix[3]), static_cast<double>(matrix[4]));
            if (horizontal > 0.0 && vertical > 0.0)
                progress.displayTransform = {matrix[0] / horizontal, matrix[1] / horizontal, matrix[3] / vertical, matrix[4] / vertical};
        }
        std::string decoder = "CPU";
        if (s->useHwDecoder) {
#if defined(_USE_CUDA_)
            decoder = "NVDEC";
#elif defined(_USE_OPENCL_)
            const AVBufferRef* device = s->inputDecoderCtx->hw_device_ctx;
            decoder = device && reinterpret_cast<const AVHWDeviceContext*>(device->data)->type == AV_HWDEVICE_TYPE_QSV ? "QSV" : "D3D11VA";
#endif
        }
        const std::string backend = getBackendName();
        progress.pipeline = "Decode: " + decoder + "  |  Process: " + (backend == "eigen" ? "Eigen (CPU)" : backend == "cuda" ? "CUDA" : "OpenCL");
        if (s->outputEncoderCtx)
            progress.pipeline += "  |  Encode: " + std::string(s->outputEncoderCtx->codec->name);
        const double timeBase = av_q2d(stream->time_base);
        const double duration = stream->duration > 0 ? stream->duration * timeBase : (s->inputFormatCtx->duration > 0 ? s->inputFormatCtx->duration / static_cast<double>(AV_TIME_BASE) : -1.0);
        if (duration > 0.0)
            progress.source += std::format("  |  {:.1f} s", duration);
        if (stream->nb_frames > 0)
            progress.fraction = progress.frames / static_cast<double>(stream->nb_frames);
        else if (frame && duration > 0.0) {
            const int64_t pts = frame->best_effort_timestamp != AV_NOPTS_VALUE ? frame->best_effort_timestamp : frame->pts;
            if (pts != AV_NOPTS_VALUE) {
                const int64_t start = stream->start_time != AV_NOPTS_VALUE ? stream->start_time : s->progressOrigin != AV_NOPTS_VALUE ? s->progressOrigin : 0;
                progress.fraction = ((pts - start) * timeBase + std::max<int64_t>(0, frame->duration) * timeBase) / duration;
            }
        }
        if (progress.fraction >= 0.0)
            progress.fraction = std::clamp(progress.fraction, 0.0, 0.999);
        progress.remainingSeconds = progress.fraction > 0.0 ? progress.elapsedSeconds * (1.0 - progress.fraction) / progress.fraction : -1.0;
    }
    if (state == VideoState::Completed) {
        progress.fraction = 1.0;
        progress.remainingSeconds = 0.0;
    }
    s->settings.observer->onProgress(progress);
}

namespace {
#if defined(_USE_OPENCL_)
const char* previewKernel = R"CLC(
typedef struct {
    int width, height, outWidth, outHeight;
    int yPitch, uPitch, vPitch;
    int shiftX, shiftY, interleaved, gray;
    float yOffset, yScale, uvScale, redV, greenU, greenV, blueU;
} Params;
__kernel void preview(__global const uchar* src, int uOffset, int vOffset, __global uchar4* dst, Params p) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= p.outWidth || y >= p.outHeight) return;
    int sx = min(p.width - 1, (int)((x + 0.5f) * p.width / p.outWidth));
    int sy = min(p.height - 1, (int)((y + 0.5f) * p.height / p.outHeight));
    int cx = sx >> p.shiftX, cy = sy >> p.shiftY;
    float yy = (src[sy * p.yPitch + sx] - p.yOffset) * p.yScale;
    float u = p.gray ? 0.f : (src[uOffset + cy * p.uPitch + cx * (1 + p.interleaved)] - 128.f) * p.uvScale;
    float v = p.gray ? 0.f : (src[vOffset + cy * p.vPitch + cx * (1 + p.interleaved)] - 128.f) * p.uvScale;
    dst[y * p.outWidth + x] = (uchar4)(convert_uchar_sat_rte(yy + p.redV * v), convert_uchar_sat_rte(yy + p.greenU * u + p.greenV * v), convert_uchar_sat_rte(yy + p.blueU * u), 255);
}
)CLC";

cl::Program buildPreviewProgram() {
    auto& manager = OclQueueManager::getInstance();
    cl::Program program(manager.getContext(), previewKernel);
    program.build({manager.getDevice()});
    return program;
}
#endif

void renderPreview(VideoSession* s, const AVFrame* original, const bool encoded, VideoPreview& preview) {
    AVFramePtr mapped;
    const AVFrame* frame = original;
    AVPixelFormat format = static_cast<AVPixelFormat>(frame->format);
    const AVCodecContext* colors = encoded ? s->outputEncoderCtx.get() : s->inputDecoderCtx.get();
    const bool gpu = frame->hw_frames_ctx != nullptr;
    if (gpu)
        format = reinterpret_cast<const AVHWFramesContext*>(frame->hw_frames_ctx->data)->sw_format;
    VideoPreviewParams params{};
    params.width = frame->width;
    params.height = frame->height;
    int boxWidth = 960, boxHeight = 540;
    s->settings.observer->previewSize(boxWidth, boxHeight);
    boxWidth = std::clamp(boxWidth, 1, 1920);
    boxHeight = std::clamp(boxHeight, 1, 1080);
    AVRational aspect = s->videoStream->sample_aspect_ratio;
    if (aspect.num <= 0 || aspect.den <= 0)
        aspect = colors->sample_aspect_ratio;
    const double ratio = aspect.num > 0 && aspect.den > 0 ? av_q2d(aspect) : 1.0;
    const double scale = std::min({1.0, boxWidth / (params.width * ratio), boxHeight / static_cast<double>(params.height)});
    params.outWidth = std::max(1, static_cast<int>(std::lround(params.width * ratio * scale)));
    params.outHeight = std::max(1, static_cast<int>(std::lround(params.height * scale)));
    const size_t bytes = static_cast<size_t>(params.outWidth) * params.outHeight * 4;
    if (preview.width != params.outWidth || preview.height != params.outHeight) {
        preview.host = std::make_unique<HostMemory<uint8_t>>(bytes);
        preview.width = params.outWidth;
        preview.height = params.outHeight;
#if defined(_USE_CUDA_)
        if (gpu)
            preview.pixels = CudaArray<uint8_t>(static_cast<int>(bytes), CudaStreamManager::getInstance().getComputeStream());
#elif defined(_USE_OPENCL_)
        if (gpu)
            preview.pixels = cl::Buffer(OclQueueManager::getInstance().getContext(), CL_MEM_WRITE_ONLY, bytes);
#endif
    }
    const auto* descriptor = av_pix_fmt_desc_get(format);
    if (!descriptor || descriptor->comp[0].depth != 8)
        throw std::runtime_error("Preview requires an 8-bit frame");
    params.shiftX = descriptor->log2_chroma_w;
    params.shiftY = descriptor->log2_chroma_h;
    params.gray = descriptor->nb_components < 3;
    params.interleaved = !params.gray && descriptor->comp[1].step == 2;
    params.yPitch = frame->linesize[0];
    params.uPitch = frame->linesize[1];
    params.vPitch = params.interleaved ? frame->linesize[1] : frame->linesize[2];
    const bool full = colors->color_range == AVCOL_RANGE_JPEG || format == AV_PIX_FMT_YUVJ420P || format == AV_PIX_FMT_YUVJ422P || format == AV_PIX_FMT_YUVJ444P;
    params.yOffset = full ? 0.0f : 16.0f;
    params.yScale = full ? 1.0f : 255.0f / 219.0f;
    params.uvScale = full ? 1.0f : 255.0f / 224.0f;
    const float kr = colors->colorspace == AVCOL_SPC_BT2020_NCL ? .2627f : colors->colorspace == AVCOL_SPC_BT709 ? .2126f : .299f;
    const float kb = colors->colorspace == AVCOL_SPC_BT2020_NCL ? .0593f : colors->colorspace == AVCOL_SPC_BT709 ? .0722f : .114f;
    params.redV = 2 * (1 - kr);
    params.blueU = 2 * (1 - kb);
    params.greenU = -kb * params.blueU / (1 - kr - kb);
    params.greenV = -kr * params.redV / (1 - kr - kb);
    if (!gpu) {
        preview.scaler = sws_getCachedContext(preview.scaler, frame->width, frame->height, format, params.outWidth, params.outHeight, AV_PIX_FMT_RGBA, SWS_BILINEAR, nullptr, nullptr, nullptr);
        if (!preview.scaler)
            throw std::runtime_error("Could not create preview converter");
        const int matrix = colors->colorspace == AVCOL_SPC_BT709 ? SWS_CS_ITU709 : colors->colorspace == AVCOL_SPC_BT2020_NCL ? SWS_CS_BT2020 : SWS_CS_ITU601;
        const int* coefficients = sws_getCoefficients(matrix);
        checkAv(sws_setColorspaceDetails(preview.scaler, coefficients, full, coefficients, 1, 0, 1 << 16, 1 << 16), "Preview color conversion");
        uint8_t* planes[]{preview.host->get(), nullptr, nullptr, nullptr};
        const int pitches[]{params.outWidth * 4, 0, 0, 0};
        checkAv(sws_scale(preview.scaler, frame->data, frame->linesize, 0, frame->height, planes, pitches), "Preview scaling");
    } else {
#if defined(_USE_CUDA_)
        const auto stream = CudaStreamManager::getInstance().getComputeStream();
        // Read back only thumbnail pixels, full-size frames stay on the GPU
        cuda_utils::launchVideoPreview(frame->data[0], frame->data[1], params.interleaved ? frame->data[1] + 1 : frame->data[2], preview.pixels.data(), params, stream);
        CUDA_CHECK(cudaMemcpyAsync(preview.host->get(), preview.pixels.data(), bytes, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
#elif defined(_USE_OPENCL_)
        auto& manager = OclQueueManager::getInstance();
        // Encoder copies can still be pending when the preview reads the frame
        if (encoded && s->openclVideoOutput)
            s->openclVideoOutput->waitUntilCopied(frame);
        if (frame->format == AV_PIX_FMT_QSV) {
            mapped.reset(av_frame_alloc());
            if (!mapped)
                throw std::bad_alloc();
            mapped->format = AV_PIX_FMT_D3D11;
            checkAv(av_hwframe_map(mapped.get(), frame, AV_HWFRAME_MAP_READ | AV_HWFRAME_MAP_DIRECT), "Map preview surface");
            frame = mapped.get();
        }
        if (!preview.source)
            preview.source = manager.getVideoInterop()->createFrame(frame, manager.getQueueRaw());
        preview.source->stage(frame);
        const auto& packed = preview.source->copyFrom(frame);
        const auto* planar = av_pix_fmt_desc_get(preview.source->planarFormat());
        params.shiftX = planar->log2_chroma_w;
        params.shiftY = planar->log2_chroma_h;
        params.interleaved = 0;
        params.yPitch = frame->width;
        params.uPitch = params.vPitch = AV_CEIL_RSHIFT(frame->width, params.shiftX);
        const int uOffset = frame->width * frame->height;
        const int vOffset = uOffset + params.uPitch * AV_CEIL_RSHIFT(frame->height, params.shiftY);
        if (!preview.kernel()) {
            preview.kernel = cl::Kernel(cl_utils::DeviceProgramCache<buildPreviewProgram>::getProgram(), "preview");
        }
        preview.kernel.setArg(0, packed);
        preview.kernel.setArg(1, uOffset);
        preview.kernel.setArg(2, vOffset);
        preview.kernel.setArg(3, preview.pixels);
        preview.kernel.setArg(4, params);
        manager.getQueue().enqueueNDRangeKernel(preview.kernel, cl::NullRange, cl::NDRange(params.outWidth, params.outHeight));
        manager.getQueue().enqueueReadBuffer(preview.pixels, CL_TRUE, 0, bytes, preview.host->get());
#else
        throw std::runtime_error("GPU preview unavailable in the CPU build");
#endif
    }
    s->settings.observer->onPreview(preview.host->get(), params.outWidth, params.outHeight, params.outWidth * 4);
}
} // namespace

void previewVideoFrame(VideoSession* s, const AVFrame* frame, const bool encoded) {
    if (!s->settings.observer || !s->settings.observer->wantsPreview())
        return;
    if (!s->preview)
        s->preview = std::make_unique<VideoPreview>();
    auto& preview = *s->preview;
    const auto now = std::chrono::steady_clock::now();
    if (preview.disabled || now - preview.last < std::chrono::milliseconds(33))
        return;
    preview.last = now;
    try {
        renderPreview(s, frame, encoded, preview);
    } catch (const std::exception& error) {
        // A preview failure must not discard an otherwise valid output
        preview.disabled = true;
        videoLog(s, std::string("Preview disabled: ") + error.what() + "\n");
    }
}
} // namespace video_utils
