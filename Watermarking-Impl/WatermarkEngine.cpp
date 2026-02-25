#include "buffer.hpp"
#include "HostMemory.hpp"
#include "ImageFileBuffer.hpp"
#include "include/common_utils.hpp"
#include "include/WatermarkEngine.hpp"
#include "include/WatermarkTypes.hpp"
#include "utils.hpp"
#include "video_utils.hpp"
#include "VideoProcessingContext.hpp"
#include "WatermarkBase.hpp"
#include <cstdint>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

#if defined(_USE_GPU_)
#include <arrayfire.h>
#elif defined(_USE_EIGEN_)
#include "eigen_utils.hpp"
#include <omp.h>
#endif

extern "C" {
#include "libavcodec/codec_par.h"
#include "libavformat/avformat.h"
#include "libavutil/log.h"
}

using namespace video_utils;
using namespace CommonUtils;
using namespace InternalUtils;
using FILEPtr = std::unique_ptr<FILE, decltype(&_pclose)>;

namespace WatermarkEngine {

void initializeEnvironment(const int openclDevice) {
#if defined(_USE_OPENCL_)
    try {
        af::setDevice(openclDevice);
    } catch (...) {
        std::cout << "NOTE: Invalid OpenCL device, using default 0\n";
        af::setDevice(0);
    }
#endif
#if defined(_USE_GPU_)
    af::info();
    std::cout << "\n";
#else
#pragma omp parallel
    {}
    std::cout << "\nUsing " + std::to_string(omp_get_max_threads()) + " parallel threads for Watermark calculations.\n";
#endif
}

struct ImageSession {
    ImageFileBuffer imgBuffer;
    std::unique_ptr<WatermarkBase> watermarkObj;
    ImageOutputBuffer watermarkBuffer;
    ImageBuffer detectGrayBuffer;
};

void ImageSessionDeleter::operator()(ImageSession* s) const { delete s; }

ImageHandle initImage(const std::string& imagePath, const uint32_t watermarkSeed, const int p, const float psnr) {
    auto* session = new ImageSession();
    loadImage(session->imgBuffer, imagePath);
    auto& [rgb, img, alpha, rows, cols, isRGB] = session->imgBuffer;
    session->watermarkObj = createWatermarkObject(rows, cols, watermarkSeed, p, psnr);
#if defined(_USE_EIGEN_)
    session->watermarkBuffer = isRGB ? ImageOutputBuffer(eigen_utils::makeEigenRGBu8(rows, cols)) : ImageOutputBuffer(Gray8Buffer(rows, cols));
#endif
    return ImageHandle(session);
}

void embedImage(ImageSession* s, MaskMethod method) {
    s->watermarkObj->makeWatermark(s->imgBuffer.image, s->imgBuffer.rgbImage, s->watermarkBuffer, method);
#if defined(_USE_GPU_)
    s->watermarkBuffer.eval();
    af::sync();
#endif
}

float detectLoadedImage(const ImageSession* s, MaskMethod method) { return s->watermarkObj->detectWatermark(s->imgBuffer.image, method); }

void prepareDetectionImage(ImageSession* s, MaskMethod method) {
    const auto floatBuffer = castToFloat(s->watermarkBuffer);
    s->detectGrayBuffer = s->imgBuffer.isRGB ? ImageBuffer(InternalUtils::rgb2gray(floatBuffer)) : ImageBuffer(floatBuffer);
#if defined(_USE_GPU_)
    s->detectGrayBuffer.eval();
    af::sync();
#endif
}

float detectEmbeddedBuffer(const ImageSession* s, MaskMethod method) { return s->watermarkObj->detectWatermark(s->detectGrayBuffer, method); }

void saveImage(const ImageSession* s, const std::string& outPath, MaskMethod method) {
    std::string suffix = (method == MaskMethod::NVF) ? "W_NVF" : "W_ME";
    InternalUtils::saveImage(outPath, suffix, s->watermarkBuffer, s->imgBuffer.alphaChannel);
}

void VideoSessionDeleter::operator()(VideoSession* s) const { delete s; }

VideoHandle initVideo(const VideoSettings& settings) {
    av_log_set_level(AV_LOG_INFO);
    auto* session = new VideoSession();
    session->settings = settings;
    AVFormatContext* rawInputCtx = nullptr;
    checkError(avformat_open_input(&rawInputCtx, settings.videoFile.c_str(), nullptr, nullptr) < 0, "Failed to open video");
    session->inputFormatCtx.reset(rawInputCtx);
    avformat_find_stream_info(session->inputFormatCtx.get(), nullptr);
    session->videoStreamIndex = findVideoStream(session->inputFormatCtx.get());
    checkError(session->videoStreamIndex == -1, "No video stream found");
    session->videoStream = session->inputFormatCtx->streams[session->videoStreamIndex];
    session->useHwDecoder = false;
    session->inputDecoderCtx = openDecoder(session->videoStream->codecpar, settings.hwDecoder, session->useHwDecoder);
    checkError(!session->inputDecoderCtx.get(), "Could not open video decoder");
    const int height = session->videoStream->codecpar->height;
    const int width = session->videoStream->codecpar->width;
    session->watermarkObj = createWatermarkObject(height, width, settings.watermarkSeed, settings.p, settings.psnr);
    session->hostFrame = std::make_unique<HostMemory<uint8_t>>(session->useHwDecoder ? width * height * 3 / 2 : width * height);
    session->inputFrame = ImageBuffer({height, width});
    session->watermarkedFrame = ImageOutputBuffer({height, width});
    session->grayFrame = Gray8Buffer({height, width});
    return VideoHandle(session);
}

int embedVideo(VideoSession* s) {
    const bool needsFilter = initFilterGraph(s);

    std::ostringstream ffmpegCmd;
    ffmpegCmd << "ffmpeg -y -f rawvideo " << getPixFmt(s->videoStream) << "-s " << s->videoStream->codecpar->width << "x" << s->videoStream->codecpar->height << " -r " << getFrameRate(s->videoStream)
              << " -i - -i \"" << s->settings.videoFile << "\" " << s->settings.encodeOptions << " -c:s copy -c:a copy -map 1:s? -map 0:v -map 1:a? -max_interleave_delta 0 "
              << getStreamRotation(s->videoStream) << getColorRange(s->videoStream) << "\"" << s->settings.encodeOutputPath << "\"";
    std::cout << "\033[38;5;208m\nFFmpeg encode command: " << ffmpegCmd.str() << "\033[0m\n\n";

    FILEPtr ffmpegPipe(_popen(ffmpegCmd.str().c_str(), "wb"), _pclose);
    checkError(!ffmpegPipe.get(), "Error: Could not open FFmpeg pipe");
    return videoDispatcher(s, VideoMode::EMBED, needsFilter, ffmpegPipe.get());
}

int detectVideo(VideoSession* s) { return videoDispatcher(s, VideoMode::DETECT, false, nullptr); }

} // namespace WatermarkEngine