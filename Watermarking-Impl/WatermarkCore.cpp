#include "buffer.hpp"
#include "common_utils.hpp"
#include "HostMemory.hpp"
#include "ImageFileBuffer.hpp"
#include "include/WatermarkCore.hpp"
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
#include <string>
#include <utility>
#include <vector>

#if defined(_USE_CUDA_)
#include "CudaStreamManager.hpp"
#include "CudaArray.hpp"
#include <algorithm>
#include <cuda_runtime.h>
#elif defined(_USE_OPENCL_)
#include <algorithm>
#include "OclQueueManager.hpp"
#include "opencl_utils.hpp"
#elif defined(_USE_EIGEN_)
#include <cstring>
#include <intrin.h>
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

using std::string;

namespace WatermarkCore {

// definition of the Image session structs and their deleters
struct ImageSession {
    string watermarkPassword;
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

// copy data to the export buffer (D2D clone for GPU builds, shallow copy for Eigen)
void exportForSave(const ImageSession* s, ExportedImage* p, MaskMethod method) {
#if defined(_USE_GPU_)
    p->finalPixels = s->watermarkBuffer.clone();
#else
    p->finalPixels = s->watermarkBuffer;
#endif
    p->alpha = s->imgBuffer.alphaChannel;
}

// same as saveImage, but used as a separate step to allow for asynchronous saving (in batched mode)
void flushToDiskAsync(ExportedImage* handle, const string& outPath, MaskMethod method) {
    const string suffix = method == MaskMethod::NVF ? "W_NVF" : "W_ME";
    InternalUtils::saveImage(outPath, suffix, handle->finalPixels, handle->alpha);
}

// INTERNAL HELPERS
namespace {
#if defined(_USE_OPENCL_)
string getOCLDeviceName(int deviceIndex = -1) {
    const auto devices = OclQueueManager::enumerateDevices();
    if (deviceIndex < 0)
        deviceIndex = OclQueueManager::getInstance().getDeviceIndex();
    if (deviceIndex >= 0 && deviceIndex < static_cast<int>(devices.size()))
        return devices[deviceIndex];
    return "Unknown OpenCL Device";
}
#elif defined(_USE_CUDA_)
string getCUDADeviceName(int deviceIndex = -1) {
    if (deviceIndex < 0)
        cudaGetDevice(&deviceIndex);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceIndex);
    return string(prop.name);
}
#endif
// generic method taking the underlying ImageOutputBuffer directly
const uint8_t* extractPixelData(const ImageOutputBuffer& buffer, int& width, int& height, int& channels) {
    static std::vector<uint8_t> hostBuffer;
#if defined(_USE_GPU_)
    width = buffer.getCols();
    height = buffer.getRows();
    channels = buffer.getChannels();
    hostBuffer.resize(static_cast<size_t>(width) * height * channels);
    buffer.toHost(hostBuffer.data());
    return hostBuffer.data();
#else
    if (buffer.isRGB()) {
        const auto& rgb = buffer.getRGB();
        width = rgb[0].cols();
        height = rgb[0].rows();
        channels = 3;
        const size_t planeSize = width * height;
        hostBuffer.resize(planeSize * channels);
        std::memcpy(hostBuffer.data(), rgb[0].data(), planeSize);
        std::memcpy(hostBuffer.data() + planeSize, rgb[1].data(), planeSize);
        std::memcpy(hostBuffer.data() + 2 * planeSize, rgb[2].data(), planeSize);
        return hostBuffer.data();
    } else {
        const auto& gray = buffer.getGray();
        width = gray.cols();
        height = gray.rows();
        channels = 1;
        return gray.data();
    }
#endif
}
} // end anonymous namespace

// main function to get the data from the image session buffer (column-wise) directly, it also fills the width, height and channels parameters for the caller
const uint8_t* getSessionPixelData(const ImageSession* session, int& width, int& height, int& channels) { return extractPixelData(session->watermarkBuffer, width, height, channels); }

// initialization, including device setup, info display, and OpenMP thread pool initialization
bool initializeEnvironment(const int openclDevice) {
    bool deviceSetSuccess = true;
#if defined(_USE_OPENCL_)
    try {
        OclQueueManager::initialize(openclDevice);
    } catch (...) {
        std::cout << "NOTE: Invalid OpenCL device index, using default 0\n";
        OclQueueManager::initialize(0);
        deviceSetSuccess = false;
    }
    const auto& dev = OclQueueManager::getInstance().getDevice();
    std::cout << "OpenCL Device [" << OclQueueManager::getInstance().getDeviceIndex() << "]: " << dev.getInfo<CL_DEVICE_NAME>() << "\n\n";
#elif defined(_USE_CUDA_)
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    std::cout << "CUDA Device [" << device << "]: " << prop.name << " (Compute " << prop.major << "." << prop.minor << ")\n\n";
    CudaStreamManager::getInstance(); // lazy initalization of the CUDA stream manager (create streams and pool)
#endif
#pragma omp parallel
    {}
    std::cout << info("Using " + std::to_string(omp_get_max_threads()) + " parallel threads for Watermark calculations.\n");
    return deviceSetSuccess;
}

// helper function to get the device name (GPU or CPU)
string getDeviceName(const int deviceIndex) {
#if defined(_USE_CUDA_)
    return getCUDADeviceName(deviceIndex);
#elif defined(_USE_OPENCL_)
    return getOCLDeviceName(deviceIndex);
#elif defined(_USE_EIGEN_)
#ifdef _WIN32
    int CPUInfo[4] = {-1};
    char CPUBrandString[0x40] = {0};
    __cpuid(CPUInfo, 0x80000000);
    unsigned int nExIds = CPUInfo[0];
    for (unsigned int i = 0x80000000; i <= nExIds; ++i) {
        __cpuid(CPUInfo, i);
        if (i == 0x80000002)
            memcpy(CPUBrandString, CPUInfo, sizeof(CPUInfo));
        else if (i == 0x80000003)
            memcpy(CPUBrandString + 16, CPUInfo, sizeof(CPUInfo));
        else if (i == 0x80000004)
            memcpy(CPUBrandString + 32, CPUInfo, sizeof(CPUInfo));
    }
    string cpuName(CPUBrandString);
    cpuName.erase(cpuName.find_last_not_of(" \n\r\t\0") + 1);
    return cpuName.empty() ? "Unknown CPU" : cpuName;
#else
    return "Unknown CPU";
#endif
#endif
}

// get the list of available GPU devices
std::vector<string> getAvailableDevices() {
    std::vector<string> devices;
#if defined(_USE_CUDA_)
    int count = 0;
    cudaGetDeviceCount(&count);
    for (int i = 0; i < count; i++)
        devices.push_back(getCUDADeviceName(i));
#elif defined(_USE_OPENCL_)
    devices = OclQueueManager::enumerateDevices();
#elif defined(_USE_EIGEN_)
    devices.push_back(getDeviceName(0));
#endif
    return devices;
}

bool isOpenCLBackend() {
#if defined(_USE_OPENCL_)
    return true;
#else
    return false;
#endif
}

void buildOpenCLKernels() {
#if defined(_USE_OPENCL_)
    cl_utils::OpenCLKernelCache<3>::getProgram();
    cl_utils::OpenCLKernelCache<5>::getProgram();
    cl_utils::OpenCLKernelCache<7>::getProgram();
    cl_utils::OpenCLKernelCache<9>::getProgram();
    cl_utils::UtilityKernelCache::getProgram();
#endif
    // NO-OP else
}

void updateSessionParams(ImageSession* s, const int p, const float psnr) {
    if (s->p == p && s->psnr == psnr)
        return;
    s->p = p;
    s->psnr = psnr;
    s->watermarkObj = createWatermarkObject(s->currentRows, s->currentCols, s->watermarkPassword, s->p, s->psnr);
}

// creates a new session for image processing, initialized with the given parameters (no memory allocations yet)
ImageHandle createImageSession(const string& watermarkPassword, const int p, const float psnr) {
    auto* session = new ImageSession();
    session->watermarkPassword = watermarkPassword;
    session->p = p;
    session->psnr = psnr;
    return ImageHandle(session);
}

// used for disk images preloading, useful for scenarios where multiple images need to be processed in parallel, as it allows the loading step to be done in parallel
PreloadedHandle preloadImageFromDisk(const string& imagePath) {
    auto* p = new PreloadedImage();
    p->buffer = InternalUtils::loadImage(imagePath);
    return PreloadedHandle(p);
}

// used to get the current image dimensions
std::pair<int, int> getImageDims(const ImageSession* s) { return {s->currentRows, s->currentCols}; }

// binds a disk preloaded image buffer into the watermark session. It also lazily initializes the watermark object and buffers based on the dimensions of the loaded image, which is useful for
// scenarios where multiple images of different dimensions need to be processed in parallel, as it avoids unnecessary allocations and initializations until the actual image data is available.
void bindPreloadedImage(ImageSession* s, PreloadedHandle preloadedData) {
    s->imgBuffer = std::move(preloadedData->buffer);
    auto& [rgb, img, alpha, rows, cols, isRGB] = s->imgBuffer;
    // lazy initialization of the watermark object and buffers, only if dimensions change or not initialized yet
    if (!s->watermarkObj || s->currentRows != rows || s->currentCols != cols) {
        s->watermarkObj = createWatermarkObject(rows, cols, s->watermarkPassword, s->p, s->psnr);
        s->currentRows = rows;
        s->currentCols = cols;
#if defined(_USE_EIGEN_)
        s->watermarkBuffer = isRGB ? ImageOutputBuffer(eigen_utils::makeEigenRGBu8(rows, cols)) : ImageOutputBuffer(Gray8Buffer(rows, cols));
#endif
    }
}

// combines the loading and binding steps, useful for single image processing without the need for preloading multiple images in parallel
void loadImage(ImageSession* session, const string& imagePath) { bindPreloadedImage(session, preloadImageFromDisk(imagePath)); }

// main function to embed the watermark into the loaded image, it calls the makeWatermark method of the watermark object,
// which implements the actual embedding algorithm based on the specified mask method (NVF or ME)
void embedImage(ImageSession* s, MaskMethod method) {
#if defined(_USE_GPU_)
    const auto& inputImg = s->imgBuffer.isRGB ? s->imgBuffer.rgbImage : s->imgBuffer.image;
    s->watermarkObj->makeWatermark(s->imgBuffer.image, inputImg, s->watermarkBuffer, method);
#else
    s->watermarkObj->makeWatermark(s->imgBuffer.image, s->imgBuffer.rgbImage, s->watermarkBuffer, method);
#endif
}

void finish() {
#if defined(_USE_CUDA_)
    cudaStreamSynchronize(CudaStreamManager::getInstance().getComputeStream());
#elif defined(_USE_OPENCL_)
    OclQueueManager::getInstance().finish();
#endif
}

// used as an intermediate step before detection, it prepares the image buffer in the correct format for the detection algorithm,
// which is always a float buffer (grayscale), regardless of the original image format (RGB or grayscale)
// used only when the input isn't already a float buffer, otherwise it is redundant
void prepareDetectionImage(ImageSession* s, MaskMethod method) {
    s->detectGrayBuffer = InternalUtils::castToFloatGray(s->watermarkBuffer, s->imgBuffer.isRGB);
    // sync so benchmarks measure completion, not just the async upload
#if defined(_USE_CUDA_)
    cudaStreamSynchronize(CudaStreamManager::getInstance().getComputeStream());
#elif defined(_USE_OPENCL_)
    OclQueueManager::getInstance().finish();
#endif
}

// main function to detect the watermark from the loaded image, it calls the detectWatermark method of the watermark object,
// which implements the actual detection algorithm based on the specified mask method (NVF or ME)
float detectLoadedImage(const ImageSession* s, MaskMethod method) { return s->watermarkObj->detectWatermark(s->imgBuffer.image, method); }

// main function to detect the watermark from the embedded buffer. This is useful only when we for example embed and then directly detect (benchmark)
// not used when we want to detect from the original loaded image, as in that case we need to prepare the detection buffer first (convert to float grayscale), which is done in prepareDetectionImage
float detectEmbeddedBuffer(const ImageSession* s, MaskMethod method) { return s->watermarkObj->detectWatermark(s->detectGrayBuffer, method); }

// saves the image to disk
void saveImage(const ImageSession* s, const string& outPath, MaskMethod method) {
    const string suffix = method == MaskMethod::NVF ? "W_NVF" : "W_ME";
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
    session->inputDecoderCtx = openDecoder(session->videoStream->codecpar, settings.hwDecoder, session->useHwDecoder, session->videoStream->time_base);
    checkError(!session->inputDecoderCtx.get(), "Could not open video decoder");
    const int height = session->videoStream->codecpar->height;
    const int width = session->videoStream->codecpar->width;
    session->watermarkObj = createWatermarkObject(height, width, settings.watermarkPassword, settings.p, settings.psnr);
    session->hostFrame = std::make_unique<HostMemory<uint8_t>>(width * height * 3 / 2);
#if defined(_USE_EIGEN_)
    // video is always grayscale, initialize the output buffer to the gray variant (bad_variant_access fix)
    session->watermarkedFrame = ImageOutputBuffer(Gray8Buffer(height, width));
#endif
    return VideoHandle(session);
}

// embed the watermark into the video using libav encoding
// initializes the filter graph if needed (10-bit / HDR), opens the encoder and muxer,
// processes all frames (video watermarked, audio/subtitles remuxed), then finalises the container
int embedVideo(VideoSession* s) {
    const bool needsFilter = initFilterGraph(s);
    video_utils::initOutputEncoder(s);
    const int framesProcessed = videoDispatcher(s, VideoMode::EMBED, needsFilter);
    video_utils::flushAndFinalize(s);
    return framesProcessed;
}

// main function to detect the watermark from the video
int detectVideo(VideoSession* s) { return videoDispatcher(s, VideoMode::DETECT, false); }

} // namespace WatermarkCore