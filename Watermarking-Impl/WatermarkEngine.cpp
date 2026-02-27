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
#include <omp.h>
#include <optional>
#include <sstream>
#include <string>
#include <utility>

#if defined(_USE_GPU_)
#include <arrayfire.h>
#elif defined(_USE_EIGEN_)
#include "eigen_utils.hpp"
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

// definition of the Image session structs and their deleters
struct ImageSession {
    uint32_t seed;
    int p;
    float psnr;
    int currentRows = 0;
    int currentCols = 0;
    ImageFileBuffer imgBuffer;
    std::unique_ptr<WatermarkBase> watermarkObj;
    ImageOutputBuffer watermarkBuffer;
    ImageBuffer detectGrayBuffer;
};

struct PreloadedImage {
    ImageFileBuffer buffer;
};

struct ExportedImage {
    ImageOutputBuffer finalPixels;
    std::optional<Gray8BufferIO> alpha;
};

void ImageSessionDeleter::operator()(ImageSession* s) const { delete s; }

void PreloadedImageDeleter::operator()(PreloadedImage* p) const { delete p; }

void ExportedImageDeleter::operator()(ExportedImage* p) const { delete p; }

ExportHandle createReusableExportBuffer() { return ExportHandle(new ExportedImage()); }

// copy data (arrayfire is copy on write, and for eigen we copy into an already allocated buffer)
void exportForSave(const ImageSession* s, ExportedImage* p, MaskMethod method) {
    p->finalPixels = s->watermarkBuffer;
    p->alpha = s->imgBuffer.alphaChannel;
}

// same as saveImage, but used as a separate step to allow for asynchronous saving (in batched mode)
void flushToDiskAsync(ExportedImage* handle, const std::string& outPath, MaskMethod method) {
    std::string suffix = (method == MaskMethod::NVF) ? "W_NVF" : "W_ME";
    InternalUtils::saveImage(outPath, suffix, handle->finalPixels, handle->alpha);
}

// initialization, including OpenCL device setup and ArrayFire info display, as well as OpenMP thread pool initialization
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
#endif
#pragma omp parallel
    {}
    std::cout << "Using " + std::to_string(omp_get_max_threads()) + " parallel threads for Watermark calculations.\n";
}

// creates a new session for image processing, initialized with the given parameters (no memory allocations yet)
ImageHandle createImageSession(const uint32_t watermarkSeed, const int p, const float psnr) {
    auto* session = new ImageSession();
    session->seed = watermarkSeed;
    session->p = p;
    session->psnr = psnr;
    return ImageHandle(session);
}

// used for disk images preloading, useful for scenarios where multiple images need to be processed in parallel, as it allows the loading step to be done in parallel
PreloadedHandle preloadImageFromDisk(const std::string& imagePath) {
    auto* p = new PreloadedImage();
    p->buffer = InternalUtils::loadImage(imagePath);
    return PreloadedHandle(p);
}

// binds a disk preloaded image buffer into the watermark session. It also lazily initializes the watermark object and buffers based on the dimensions of the loaded image, which is useful for
// scenarios where multiple images of different dimensions need to be processed in parallel, as it avoids unnecessary allocations and initializations until the actual image data is available.
void bindPreloadedImage(ImageSession* s, PreloadedHandle preloadedData) {
    s->imgBuffer = std::move(preloadedData->buffer);
    auto& [rgb, img, alpha, rows, cols, isRGB] = s->imgBuffer;
    // lazy initialization of the watermark object and buffers, only if dimensions change or not initialized yet
    if (!s->watermarkObj || s->currentRows != rows || s->currentCols != cols) {
        // for GPU backends we need to clear the previous allocations to free GPU memory before creating a new one,
        // otherwise we might run out of memory (ArrayFire's memory manager is greedy)
#if defined(_USE_GPU_)
        if (s->watermarkObj)
            af::deviceGC();
#endif
        s->watermarkObj = createWatermarkObject(rows, cols, s->seed, s->p, s->psnr);
        s->currentRows = rows;
        s->currentCols = cols;
#if defined(_USE_EIGEN_)
        s->watermarkBuffer = isRGB ? ImageOutputBuffer(eigen_utils::makeEigenRGBu8(rows, cols)) : ImageOutputBuffer(Gray8Buffer(rows, cols));
#endif
    }
}

// combines the loading and binding steps, useful for single image processing without the need for preloading multiple images in parallel
void loadImage(ImageSession* session, const std::string& imagePath) { bindPreloadedImage(session, preloadImageFromDisk(imagePath)); }

// main function to embed the watermark into the loaded image, it calls the makeWatermark method of the watermark object,
// which implements the actual embedding algorithm based on the specified mask method (NVF or ME)
void embedImage(ImageSession* s, MaskMethod method) {
    s->watermarkObj->makeWatermark(s->imgBuffer.image, s->imgBuffer.rgbImage, s->watermarkBuffer, method);
#if defined(_USE_GPU_)
    s->watermarkBuffer.eval();
    af::sync();
#endif
}

// used as an intermediate step before detection, it prepares the image buffer in the correct format for the detection algorithm,
// which is always a float buffer (grayscale), regardless of the original image format (RGB or grayscale)
// used only when the input isn't already a float buffer, otherwise it is redundant
void prepareDetectionImage(ImageSession* s, MaskMethod method) {
    const auto floatBuffer = castToFloat(s->watermarkBuffer);
    s->detectGrayBuffer = s->imgBuffer.isRGB ? ImageBuffer(InternalUtils::rgb2gray(floatBuffer)) : ImageBuffer(floatBuffer);
#if defined(_USE_GPU_)
    s->detectGrayBuffer.eval();
    af::sync();
#endif
}

// main function to detect the watermark from the loaded image, it calls the detectWatermark method of the watermark object,
// which implements the actual detection algorithm based on the specified mask method (NVF or ME)
float detectLoadedImage(const ImageSession* s, MaskMethod method) { return s->watermarkObj->detectWatermark(s->imgBuffer.image, method); }

// main function to detect the watermark from the embedded buffer. This is useful only when we for example embed and then directly detect (benchmark)
// not used when we want to detect from the original loaded image, as in that case we need to prepare the detection buffer first (convert to float grayscale), which is done in prepareDetectionImage
float detectEmbeddedBuffer(const ImageSession* s, MaskMethod method) { return s->watermarkObj->detectWatermark(s->detectGrayBuffer, method); }

// saves the image to disk
void saveImage(const ImageSession* s, const std::string& outPath, MaskMethod method) {
    std::string suffix = (method == MaskMethod::NVF) ? "W_NVF" : "W_ME";
    InternalUtils::saveImage(outPath, suffix, s->watermarkBuffer, s->imgBuffer.alphaChannel);
}

// definition of the Video session struct and its deleter
void VideoSessionDeleter::operator()(VideoSession* s) const { delete s; }

// initializes the video session by opening the video file, finding the video stream, opening the decoder, and initializing the watermark object and processing buffers based on the video dimensions
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

// main function to embed the watermark into the video, it first initializes the filter graph if needed (for 10-bit to 8-bit conversion and HDR to SDR tonemapping),
// then constructs the FFmpeg command for encoding the output video with the watermark
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

// main function to detect the watermark from the video
int detectVideo(VideoSession* s) { return videoDispatcher(s, VideoMode::DETECT, false, nullptr); }

} // namespace WatermarkEngine