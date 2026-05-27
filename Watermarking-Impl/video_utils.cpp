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
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <exception>
#include <format>
#include <iostream>
#include <mutex>
#include <queue>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(_USE_CUDA_)
#include "CudaArray.hpp"
#include "CudaStreamManager.hpp"
#include "cuda_utils.hpp"
#include <cuda_runtime.h>
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
namespace {
// internal implementations

bool checkPixelFormatSupport(const std::span<const AVPixelFormat> formats, const AVPixelFormat format) {
    const bool isValidFormat = std::ranges::any_of(formats, [&](auto f) { return f == format; });
    checkError(!isValidFormat, "Error: Video frame format not supported, aborting");
    return isValidFormat;
}

// Queue that decouples the decode/watermark main thread from the encode background thread
struct EncodeQueue {
    static constexpr int MAX_DEPTH = 8;
    std::queue<AVFramePtr> q;
    std::mutex mtx;
    std::condition_variable cv;
    bool aborted = false;

    // push frame (or nullptr for end), blocks when the queue is full
    void push(AVFramePtr f) {
        std::unique_lock lk(mtx);
        cv.wait(lk, [&] { return static_cast<int>(q.size()) < MAX_DEPTH || aborted; });
        if (!aborted)
            q.push(std::move(f));
        cv.notify_all();
    }

    // returns nullptr on end, or abort
    AVFramePtr pop() {
        std::unique_lock lk(mtx);
        cv.wait(lk, [&] { return !q.empty() || aborted; });
        if (q.empty())
            return nullptr;
        AVFramePtr f = std::move(q.front());
        q.pop();
        cv.notify_all();
        return f;
    }

    // unblock any waiting caller immediately (error path)
    void abort() {
        std::unique_lock lk(mtx);
        aborted = true;
        cv.notify_all();
    }
};

AVCodecContextPtr openSoftwareDecoder(const AVCodecParameters* inputCodecParams) {
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
    if (avcodec_open2(ctx.get(), inputDecoder, nullptr) < 0)
        return nullptr;
    checkPixelFormatSupport(supportedFormats, ctx->pix_fmt);
    return ctx;
}

#if defined(_USE_CUDA_)
// try to open a CUDA hardware accelerated decoder, falls back to software on any failure
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
    // share our existing primary CUDA context with the decoder
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

// parse "-c:v codec -key val ..." into {codecName, AVDictionary*}
// strips stream specifiers (:v, :a, :s) so AVOptions names match libav
std::pair<std::string, AVDictionary*> parseEncodeOptions(const std::string& optStr) {
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

// pull all ready packets from the encoder and write them to the output container (non-blocking)
void drainEncoderPackets(VideoSession* s) {
    const AVPacketPtr pkt(av_packet_alloc());
    const AVStream* outVideoStream = s->outputFormatCtx->streams[s->outputVideoStreamIndex];
    while (avcodec_receive_packet(s->outputEncoderCtx.get(), pkt.get()) == 0) {
        av_packet_rescale_ts(pkt.get(), s->outputEncoderCtx->time_base, outVideoStream->time_base);
        pkt->stream_index = s->outputVideoStreamIndex;
        checkError(av_interleaved_write_frame(s->outputFormatCtx.get(), pkt.get()) < 0, "Failed to write encoded video packet");
        av_packet_unref(pkt.get());
    }
    // AVERROR(EAGAIN) -> encoder needs more frames before producing output (we ignore it, keep going)
}

// encode thread: pops frames, sends to encoder, drains output packets (nullptr frame signals END)
// runs on a background thread and errors are marshalled back via encErr
// does NOT flush the encoder on exit — flushAndFinalize handles that after join()
void encodeWorker(VideoSession* s, EncodeQueue& queue, std::exception_ptr& encErr) noexcept {
    try {
        while (AVFramePtr frame = queue.pop()) {
            const int ret = avcodec_send_frame(s->outputEncoderCtx.get(), frame.get());
            checkError(ret < 0 && ret != AVERROR(EAGAIN), "Encode thread: avcodec_send_frame failed");
            drainEncoderPackets(s);
        }
        // nullptr popped -> all frames sent and flushAndFinalize will send the null frame + final drain
    } catch (...) {
        encErr = std::current_exception();
        queue.abort(); // unblock main thread if it is waiting on a full queue
    }
}

#if defined(_USE_CUDA_)
// NVDEC+NVENC zero-copy: wrap CUDA Y+UV buffers in a hw AVFrame and send directly to NVENC
void encodeFrameGPU(VideoSession* s, const CudaArray<uint8_t>& yRowMajor, const AVFrame* srcFrame, const int64_t pts) {
    const auto [height, width] = s->videoDims();
    const auto stream = CudaStreamManager::getInstance().getComputeStream();
    AVFramePtr encFrame(av_frame_alloc());
    checkError(!encFrame, "Failed to allocate NVENC hw frame");
    encFrame->format = AV_PIX_FMT_CUDA;
    encFrame->width = width;
    encFrame->height = height;
    encFrame->pts = pts;
    checkError(av_hwframe_get_buffer(s->outputEncoderCtx->hw_frames_ctx, encFrame.get(), 0) < 0, "Failed to get NVENC hw frame buffer");
    // D2D: watermarked Y (row-major, stride=width) -> NVENC Y plane (NVENC-aligned stride)
    cudaMemcpy2DAsync(encFrame->data[0], encFrame->linesize[0], yRowMajor.data(), width, width, height, cudaMemcpyDeviceToDevice, stream);
    // D2D: original NV12 UV stays interleaved —> NVENC expects NV12, no conversion needed
    cudaMemcpy2DAsync(encFrame->data[1], encFrame->linesize[1], srcFrame->data[1], srcFrame->linesize[1], width, height / 2, cudaMemcpyDeviceToDevice, stream);
    cudaStreamSynchronize(stream);
    const int ret = avcodec_send_frame(s->outputEncoderCtx.get(), encFrame.get());
    checkError(ret < 0 && ret != AVERROR(EAGAIN), "NVENC send_frame failed");
    drainEncoderPackets(s);
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
        queue.push(buildEncFrame(s, frame->pts, frame));
    } else {
        // passthrough: take a refcounted reference to the decoded frame (zero data copy)
        AVFramePtr ref(av_frame_alloc());
        checkError(av_frame_ref(ref.get(), frame) < 0, "Failed to ref passthrough frame");
        // if FULL range we normalize to YUV420P (YUVJ420p is deprecated)
        // encoder context already carries color_range=AVCOL_RANGE_JPEG, so players can understand it's full range
        if (ref->format == AV_PIX_FMT_YUVJ420P)
            ref->format = AV_PIX_FMT_YUV420P;
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
// synchronous encode: used by the NVDEC+SW-encoder path (chroma already in hostFrame)
void encodeFrame(VideoSession* s, const int64_t pts) {
    AVFramePtr encFrame = buildEncFrame(s, pts);
    const int ret = avcodec_send_frame(s->outputEncoderCtx.get(), encFrame.get());
    checkError(ret < 0 && ret != AVERROR(EAGAIN), "Encoder send_frame failed");
    drainEncoderPackets(s);
}

// embed watermark in a NVDEC frame, two paths depending on the encoder:
// NVDEC + NVENC: full zero-copy, Y and UV stay in VRAM
// NVDEC + SW encoder: download Y+UV to hostFrame, CPU encodes
void embedWatermarkHWAccel(VideoSession* s, int& framesCount, const AVFrame* frame) {
    const auto stream = CudaStreamManager::getInstance().getComputeStream();
    const auto [height, width] = s->videoDims();
    const bool doEmbed = framesCount % s->settings.watermarkInterval == 0;

    if (doEmbed)
        cout << std::format(" [Embedding frame {}]\n", framesCount + 1);
    if (s->outputEncoderCtx->hw_frames_ctx) {
        // NVDEC + NVENC zero-copy
        CudaArray<uint8_t> yRowMajor(height, width, stream);
        if (doEmbed) {
            CudaArray<float> lumaFloat(height, width, stream);
            cuda_utils::launchPitchedToFloatKernel(frame->data[0], lumaFloat.data(), width, height, frame->linesize[0], stream);
            s->watermarkObj->makeWatermark(lumaFloat, lumaFloat, s->watermarkedFrame, MaskMethod::ME);
            cuda_utils::launchColMajorToRowMajorU8Kernel(s->watermarkedFrame.data(), yRowMajor.data(), width, height, 1, stream);
        } else {
            cudaMemcpy2DAsync(yRowMajor.data(), width, frame->data[0], frame->linesize[0], width, height, cudaMemcpyDeviceToDevice, stream);
        }
        encodeFrameGPU(s, yRowMajor, frame, frame->pts);
    } else {
        // NVDEC + SW encoder
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
        encodeFrame(s, frame->pts);
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
    cuda_utils::launchPitchedToFloatKernel(frame->data[0], lumaBuffer.data(), width, height, frame->linesize[0], stream);
    const float correlation = s->watermarkObj->detectWatermark(lumaBuffer, MaskMethod::ME);
    cout << "Correlation for frame: " << (framesCount + 1) << ": " << correlation << "\n";
    framesCount++;
}
#endif

// main frames loop: decodes packets, calls processFrame for each video frame,
// remuxes audio/subtitle packets directly when outputFormatCtx is set (embed mode)
template <bool needsFilter, typename Func>
int processFrames(VideoSession* s, Func&& processFrame) {
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
                    if constexpr (needsFilter)
                        filterFrame(frame, filteredFrame, s);
                    std::forward<Func>(processFrame)(frame.get(), framesCount);
                }
            }
        } else if (s->outputFormatCtx) {
            const int inIdx = packet->stream_index;
            if (inIdx < static_cast<int>(s->inputToOutputStreamMap.size())) {
                const int outIdx = s->inputToOutputStreamMap[inIdx];
                if (outIdx >= 0) {
                    av_packet_rescale_ts(packet.get(), s->inputFormatCtx->streams[inIdx]->time_base, s->outputFormatCtx->streams[outIdx]->time_base);
                    packet->stream_index = outIdx;
                    av_interleaved_write_frame(s->outputFormatCtx.get(), packet.get());
                }
            }
        }
        av_packet_unref(packet.get());
    }
    avcodec_send_packet(s->inputDecoderCtx.get(), nullptr);
    while (avcodec_receive_frame(s->inputDecoderCtx.get(), frame.get()) == 0) {
        if constexpr (needsFilter)
            filterFrame(frame, filteredFrame, s);
        std::forward<Func>(processFrame)(frame.get(), framesCount);
    }
    return framesCount;
}

} // namespace

// public API

int findVideoStream(const AVFormatContext* inputFormatCtx) {
    for (unsigned int i = 0; i < inputFormatCtx->nb_streams; i++)
        if (inputFormatCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO)
            return i;
    return -1;
}

AVCodecContextPtr openDecoder(const AVCodecParameters* inputCodecParams, const string& userHwDecoder, bool& useHwDecoder) {
#if defined(_USE_CUDA_)
    return openDecoderHWAccel(inputCodecParams, userHwDecoder, useHwDecoder);
#else
    return openSoftwareDecoder(inputCodecParams);
#endif
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
    // SW detect
    if (op == VideoMode::DETECT)
        return needsFilter ?
            processFrames<true>(s, [&](const AVFrame* frame, int& framesCount) { detectWatermark(s, framesCount, frame); }) :
            processFrames<false>(s, [&](const AVFrame* frame, int& framesCount) { detectWatermark(s, framesCount, frame); });

    // SW embed: start a dedicated encode thread so decode+watermark (main) and
    // avcodec_send_frame+write (background) run concurrently
    EncodeQueue queue;
    std::exception_ptr encErr;
    std::thread encThread(encodeWorker, s, std::ref(queue), std::ref(encErr));
    int result = 0;
    try {
        result = needsFilter ?
            processFrames<true>(s, [&](const AVFrame* frame, int& framesCount) { embedWatermark(s, framesCount, frame, queue); }) :
            processFrames<false>(s, [&](const AVFrame* frame, int& framesCount) { embedWatermark(s, framesCount, frame, queue); });
        queue.push(nullptr); // END: tell the encode thread to flush and exit
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
// clang-format on

void initOutputEncoder(VideoSession* s) {
    checkError(s->settings.encodeOutputPath.empty(), "No output path specified for video encode");

    auto [codecName, rawOpts] = parseEncodeOptions(s->settings.encodeOptions);
    AVDictionary* opts = rawOpts;
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

    // important to set 0 for maximum performance
    encCtx->thread_count = 0;

    const int height = s->videoStream->codecpar->height;
    const int width = s->videoStream->codecpar->width;
    encCtx->width = width;
    encCtx->height = height;
    // NVDEC+NVENC zero-copy: encoder receives CUDA NV12 frames directly from VRAM
    // all other paths: encoder receives YUV420P frames from hostFrame (CPU RAM)
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
        fc->initial_pool_size = 4;
        checkError(av_hwframe_ctx_init(framesRef) < 0, "Failed to init hw_frames_ctx for NVENC");
        encCtx->hw_frames_ctx = framesRef;
    }
#endif

    const int openRet = avcodec_open2(encCtx.get(), encoder, &opts);
    av_dict_free(&opts);
    checkError(openRet < 0, "Failed to open encoder: " + codecName);

    s->inputToOutputStreamMap.assign(s->inputFormatCtx->nb_streams, -1);
    AVStream* outVideoStream = avformat_new_stream(s->outputFormatCtx.get(), nullptr);
    checkError(!outVideoStream, "Failed to create output video stream");
    checkError(avcodec_parameters_from_context(outVideoStream->codecpar, encCtx.get()) < 0, "Failed to copy encoder parameters to output video stream");
    outVideoStream->time_base = encCtx->time_base;
    // fix codecpar->format when CUDA input was used
    if (useGpuPipeline)
        outVideoStream->codecpar->format = AV_PIX_FMT_YUV420P;
    // copy side data (rotation, etc) -> skip HDR metadata when we tonemapped to SDR
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

    // remux audio + subtitle streams as-is
    for (unsigned i = 0; i < s->inputFormatCtx->nb_streams; i++) {
        const AVStream* inSt = s->inputFormatCtx->streams[i];
        const AVMediaType type = inSt->codecpar->codec_type;
        if (type != AVMEDIA_TYPE_AUDIO && type != AVMEDIA_TYPE_SUBTITLE)
            continue;
        AVStream* outSt = avformat_new_stream(s->outputFormatCtx.get(), nullptr);
        if (!outSt)
            continue;
        if (avcodec_parameters_copy(outSt->codecpar, inSt->codecpar) < 0)
            continue;
        outSt->time_base = inSt->time_base;
        s->inputToOutputStreamMap[i] = outSt->index;
    }
    // open the output file and write the container header
    if (!(s->outputFormatCtx->oformat->flags & AVFMT_NOFILE))
        checkError(avio_open(&s->outputFormatCtx->pb, s->settings.encodeOutputPath.c_str(), AVIO_FLAG_WRITE) < 0, "Failed to open output file: " + s->settings.encodeOutputPath);
    s->outputFormatCtx->max_interleave_delta = 0; // same as -max_interleave_delta 0 in the ffmpeg cli
    AVDictionary* hdrOpts = nullptr;
    checkError(avformat_write_header(s->outputFormatCtx.get(), &hdrOpts) < 0, "Failed to write output container header");
    av_dict_free(&hdrOpts);

    s->outputEncoderCtx = std::move(encCtx);
    cout << info("Encoder: " + codecName + ": \"" + s->settings.encodeOutputPath + "\"\n\n");
}

void flushAndFinalize(VideoSession* s) {
    if (!s->outputEncoderCtx || !s->outputFormatCtx)
        return;
    avcodec_send_frame(s->outputEncoderCtx.get(), nullptr);
    drainEncoderPackets(s);
    av_write_trailer(s->outputFormatCtx.get());
}

} // namespace video_utils
