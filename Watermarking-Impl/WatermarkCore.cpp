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
#include <algorithm>
#include <cstdint>
#include <format>
#include <immintrin.h>
#include <iostream>
#include <memory>
#include <omp.h>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(_USE_CUDA_)
#include "CudaCheck.hpp"
#include "CudaStreamManager.hpp"
#include "CudaArray.hpp"
#include <cuda_runtime.h>
#elif defined(_USE_OPENCL_)
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
    bool currentIsRGB = false;
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

// Move the CPU output into an idle export buffer, GPU builds clone device data
void exportForSave(ImageSession* s, ExportedImage* p, MaskMethod method) {
#if defined(_USE_GPU_)
    p->finalPixels = s->watermarkBuffer.clone();
#else
    if (!p->finalPixels.matches(s->currentRows, s->currentCols, s->currentIsRGB))
        p->finalPixels = s->currentIsRGB ? ImageOutputBuffer(eigen_utils::makeEigenRGBu8(s->currentRows, s->currentCols)) : ImageOutputBuffer(Gray8Buffer(s->currentRows, s->currentCols));
    std::swap(p->finalPixels, s->watermarkBuffer);
#endif
    // Batch export consumes the alpha plane, the next image bind replaces the input buffer
    p->alpha = std::move(s->imgBuffer.alphaChannel);
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
        CUDA_CHECK(cudaGetDevice(&deviceIndex));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, deviceIndex));
    return string(prop.name);
}
#endif
// generic method taking the underlying ImageOutputBuffer directly
SessionPixelData extractPixelData(const ImageOutputBuffer& buffer) {
    SessionPixelData result;
#if defined(_USE_GPU_)
    result.width = buffer.getCols();
    result.height = buffer.getRows();
    result.channels = buffer.getChannels();
    result.pixels.resize(static_cast<size_t>(result.width) * result.height * result.channels);
    buffer.toHost(result.pixels.data());
#else
    if (buffer.isRGB()) {
        const auto& rgb = buffer.getRGB();
        result.width = static_cast<int>(rgb[0].cols());
        result.height = static_cast<int>(rgb[0].rows());
        result.channels = 3;
        const size_t planeSize = static_cast<size_t>(result.width) * result.height;
        result.pixels.resize(planeSize * result.channels);
        std::memcpy(result.pixels.data(), rgb[0].data(), planeSize);
        std::memcpy(result.pixels.data() + planeSize, rgb[1].data(), planeSize);
        std::memcpy(result.pixels.data() + 2 * planeSize, rgb[2].data(), planeSize);
    } else {
        const auto& gray = buffer.getGray();
        result.width = static_cast<int>(gray.cols());
        result.height = static_cast<int>(gray.rows());
        result.channels = 1;
        const size_t size = static_cast<size_t>(result.width) * result.height;
        result.pixels.resize(size);
        std::memcpy(result.pixels.data(), gray.data(), size);
    }
#endif
    return result;
}
} // end anonymous namespace

// main function to get the data from the image session buffer (column-wise) directly, it also fills the width, height and channels parameters for the caller
SessionPixelData getSessionPixelData(const ImageSession* session) { return extractPixelData(session->watermarkBuffer); }

namespace {
// transpose 8 column-major byte columns into eight row vectors
void transposePreviewBlock(const uint8_t* source, const size_t columnStride, __m128i rows[8]) {
    const __m128i c0 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(source + 0 * columnStride));
    const __m128i c1 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(source + 1 * columnStride));
    const __m128i c2 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(source + 2 * columnStride));
    const __m128i c3 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(source + 3 * columnStride));
    const __m128i c4 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(source + 4 * columnStride));
    const __m128i c5 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(source + 5 * columnStride));
    const __m128i c6 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(source + 6 * columnStride));
    const __m128i c7 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(source + 7 * columnStride));
    const __m128i p0 = _mm_unpacklo_epi8(c0, c1);
    const __m128i p1 = _mm_unpacklo_epi8(c2, c3);
    const __m128i p2 = _mm_unpacklo_epi8(c4, c5);
    const __m128i p3 = _mm_unpacklo_epi8(c6, c7);
    const __m128i q0 = _mm_unpacklo_epi16(p0, p1);
    const __m128i q1 = _mm_unpackhi_epi16(p0, p1);
    const __m128i q2 = _mm_unpacklo_epi16(p2, p3);
    const __m128i q3 = _mm_unpackhi_epi16(p2, p3);
    const __m128i r01 = _mm_unpacklo_epi32(q0, q2);
    const __m128i r23 = _mm_unpackhi_epi32(q0, q2);
    const __m128i r45 = _mm_unpacklo_epi32(q1, q3);
    const __m128i r67 = _mm_unpackhi_epi32(q1, q3);
    rows[0] = r01;
    rows[1] = _mm_srli_si128(r01, 8);
    rows[2] = r23;
    rows[3] = _mm_srli_si128(r23, 8);
    rows[4] = r45;
    rows[5] = _mm_srli_si128(r45, 8);
    rows[6] = r67;
    rows[7] = _mm_srli_si128(r67, 8);
}
} // namespace

void copySessionPixelsForPreview(const SessionPixelData& source, uint8_t* destination, const size_t bytesPerLine) {
    const int rows = source.height;
    const int cols = source.width;
    const int channels = source.channels;
    if (rows <= 0 || cols <= 0 || (channels != 1 && channels != 3) || !destination || bytesPerLine < static_cast<size_t>(cols) * channels ||
        source.pixels.size() < static_cast<size_t>(rows) * cols * channels)
        throw std::invalid_argument("Invalid session pixels for preview");

    // small row tiles keeps cache lines HOT (while we read each planar column in order)
    constexpr int tileRows = 32;
    const size_t planeSize = static_cast<size_t>(rows) * cols;
    const uint8_t* red = source.pixels.data();
    const uint8_t* green = channels == 3 ? red + planeSize : nullptr;
    const uint8_t* blue = channels == 3 ? green + planeSize : nullptr;
    const __m128i zero = _mm_setzero_si128();
    const __m128i rgbMask = _mm_setr_epi8(0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, -1, -1, -1, -1);
#pragma omp parallel for schedule(static)
    for (int rowBlock = 0; rowBlock < rows; rowBlock += tileRows) {
        const int rowEnd = std::min(rowBlock + tileRows, rows);
        int row = rowBlock;
        for (; row + 7 < rowEnd; row += 8) {
            int col = 0;
            for (; col + 7 < cols; col += 8) {
                const size_t sourceOffset = static_cast<size_t>(col) * rows + row;
                __m128i redRows[8];
                transposePreviewBlock(red + sourceOffset, rows, redRows);
                if (channels == 1) {
                    for (int lane = 0; lane < 8; ++lane)
                        _mm_storel_epi64(reinterpret_cast<__m128i*>(destination + static_cast<size_t>(row + lane) * bytesPerLine + col), redRows[lane]);
                } else {
                    __m128i greenRows[8];
                    __m128i blueRows[8];
                    transposePreviewBlock(green + sourceOffset, rows, greenRows);
                    transposePreviewBlock(blue + sourceOffset, rows, blueRows);
                    for (int lane = 0; lane < 8; ++lane) {
                        const __m128i rg = _mm_unpacklo_epi8(redRows[lane], greenRows[lane]);
                        const __m128i bz = _mm_unpacklo_epi8(blueRows[lane], zero);
                        const __m128i rgbaFirst = _mm_unpacklo_epi16(rg, bz);
                        const __m128i rgbaSecond = _mm_unpackhi_epi16(rg, bz);
                        const __m128i rgbFirst = _mm_shuffle_epi8(rgbaFirst, rgbMask);
                        const __m128i rgbSecond = _mm_shuffle_epi8(rgbaSecond, rgbMask);
                        uint8_t* pixel = destination + static_cast<size_t>(row + lane) * bytesPerLine + static_cast<size_t>(col) * 3;
                        _mm_storeu_si128(reinterpret_cast<__m128i*>(pixel), _mm_or_si128(rgbFirst, _mm_slli_si128(rgbSecond, 12)));
                        _mm_storel_epi64(reinterpret_cast<__m128i*>(pixel + 16), _mm_srli_si128(rgbSecond, 4));
                    }
                }
            }
            // tail columns that don't fit into 8x8 block scalar
            for (; col < cols; ++col)
                for (int lane = 0; lane < 8; ++lane) {
                    const size_t sourceIndex = static_cast<size_t>(col) * rows + row + lane;
                    uint8_t* pixel = destination + static_cast<size_t>(row + lane) * bytesPerLine + static_cast<size_t>(col) * channels;
                    pixel[0] = red[sourceIndex];
                    if (channels == 3) {
                        pixel[1] = green[sourceIndex];
                        pixel[2] = blue[sourceIndex];
                    }
                }
        }
        // tail rows that don't fit into 8x8 block should be scalar
        for (; row < rowEnd; ++row)
            for (int col = 0; col < cols; ++col) {
                const size_t sourceIndex = static_cast<size_t>(col) * rows + row;
                uint8_t* pixel = destination + static_cast<size_t>(row) * bytesPerLine + static_cast<size_t>(col) * channels;
                pixel[0] = red[sourceIndex];
                if (channels == 3) {
                    pixel[1] = green[sourceIndex];
                    pixel[2] = blue[sourceIndex];
                }
            }
    }
}

// initialization, including device setup, info display, and OpenMP thread pool initialization
bool initializeEnvironment(const int deviceIndex) {
    bool deviceSetSuccess = true;
#if defined(_USE_OPENCL_)
    try {
        OclQueueManager::initialize(deviceIndex);
    } catch (...) {
        std::cout << "NOTE: Invalid OpenCL device index, using default 0\n";
        OclQueueManager::initialize(0);
        deviceSetSuccess = false;
    }
    const auto& dev = OclQueueManager::getInstance().getDevice();
    std::cout << "OpenCL Device [" << OclQueueManager::getInstance().getDeviceIndex() << "]: " << dev.getInfo<CL_DEVICE_NAME>() << "\n\n";
#elif defined(_USE_CUDA_)
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    if (count == 0)
        throw std::runtime_error("No CUDA GPU devices found");
    const int device = deviceIndex >= 0 && deviceIndex < count ? deviceIndex : 0;
    if (device != deviceIndex) {
        std::cout << "NOTE: Invalid CUDA device index, using default 0\n";
        deviceSetSuccess = false;
    }
    CUDA_CHECK(cudaSetDevice(device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::cout << "CUDA Device [" << device << "]: " << prop.name << " (Compute " << prop.major << "." << prop.minor << ")\n\n";
    CudaStreamManager::getInstance().getComputeStream(); // initialize the selected device's stream and pool
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
    CUDA_CHECK(cudaGetDeviceCount(&count));
    for (int i = 0; i < count; i++)
        devices.push_back(getCUDADeviceName(i));
#elif defined(_USE_OPENCL_)
    devices = OclQueueManager::enumerateDevices();
#elif defined(_USE_EIGEN_)
    devices.push_back(getDeviceName(0));
#endif
    return devices;
}

int getCurrentDeviceIndex() {
#if defined(_USE_CUDA_)
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    return device;
#elif defined(_USE_OPENCL_)
    return OclQueueManager::getInstance().getDeviceIndex();
#else
    return 0;
#endif
}

bool isOpenCLBackend() {
#if defined(_USE_OPENCL_)
    return true;
#else
    return false;
#endif
}

string getBackendName() {
#if defined(_USE_CUDA_)
    return "cuda";
#elif defined(_USE_OPENCL_)
    return "opencl";
#elif defined(_USE_EIGEN_)
    return "eigen";
#else
    return "unknown";
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
    const bool pChanged = s->p != p;
    s->p = p;
    s->psnr = psnr;
    if (!s->watermarkObj)
        return;
    if (pChanged)
        s->watermarkObj = createWatermarkObject(s->currentRows, s->currentCols, s->watermarkPassword, s->p, s->psnr);
    else
        s->watermarkObj->updatePsnr(psnr);
}

// creates a new session for image processing, initialized with the given parameters (no memory allocations yet)
ImageHandle createImageSession(const string& watermarkPassword, const int p, const float psnr) {
    ImageHandle session(new ImageSession());
    session->watermarkPassword = watermarkPassword;
    session->p = p;
    session->psnr = psnr;
    return session;
}

// used for disk images preloading, useful for scenarios where multiple images need to be processed in parallel, as it allows the loading step to be done in parallel
PreloadedHandle preloadImageFromDisk(const string& imagePath, const int deviceIndex, const bool captureOriginal) {
#if defined(_USE_CUDA_)
    // Each prefetch thread must activate the GPU used by its owning session
    if (deviceIndex >= 0)
        CUDA_CHECK(cudaSetDevice(deviceIndex));
#endif
    PreloadedHandle p(new PreloadedImage());
    p->buffer = InternalUtils::loadImage(imagePath, captureOriginal);
    return p;
}

// used to get the current image dimensions
std::pair<int, int> getImageDims(const ImageSession* s) { return {s->currentRows, s->currentCols}; }
OriginalPixelData takeOriginalPixelData(ImageSession* s) {
    return {std::move(s->imgBuffer.originalPreview), static_cast<int>(s->imgBuffer.cols), static_cast<int>(s->imgBuffer.rows), s->imgBuffer.previewChannels};
}

// binds a disk preloaded image buffer into the watermark session. It also lazily initializes the watermark object and buffers based on the dimensions of the loaded image, which is useful for
// scenarios where multiple images of different dimensions need to be processed in parallel, as it avoids unnecessary allocations and initializations until the actual image data is available.
void bindPreloadedImage(ImageSession* s, PreloadedHandle preloadedData) {
    s->imgBuffer = std::move(preloadedData->buffer);
    const auto rows = s->imgBuffer.rows;
    const auto cols = s->imgBuffer.cols;
    const auto isRGB = s->imgBuffer.isRGB;
    // lazy initialization of the watermark object and buffers, only if dimensions change or not initialized yet
    if (!s->watermarkObj || s->currentRows != rows || s->currentCols != cols || s->currentIsRGB != isRGB) {
        s->watermarkObj = createWatermarkObject(rows, cols, s->watermarkPassword, s->p, s->psnr);
        s->currentRows = rows;
        s->currentCols = cols;
        s->currentIsRGB = isRGB;
#if defined(_USE_EIGEN_)
        s->watermarkBuffer = isRGB ? ImageOutputBuffer(eigen_utils::makeEigenRGBu8(rows, cols)) : ImageOutputBuffer(Gray8Buffer(rows, cols));
#endif
    }
}

// combines the loading and binding steps, useful for single image processing without the need for preloading multiple images in parallel
void loadImage(ImageSession* session, const string& imagePath, const bool captureOriginal) { bindPreloadedImage(session, preloadImageFromDisk(imagePath, -1, captureOriginal)); }

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
    CUDA_CHECK(cudaStreamSynchronize(CudaStreamManager::getInstance().getComputeStream()));
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
    CUDA_CHECK(cudaStreamSynchronize(CudaStreamManager::getInstance().getComputeStream()));
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

void saveImageExact(const ImageSession* s, const string& outPath) { InternalUtils::saveImage(outPath, "", s->watermarkBuffer, s->imgBuffer.alphaChannel); }

// definition of the Video session struct and its deleter
void VideoSessionDeleter::operator()(VideoSession* s) const { delete s; }

// initializes the video session by opening the video file, finding the video stream, opening the decoder, and initializing the watermark object and processing buffers based on the video dimensions
VideoHandle initVideo(const VideoSettings& settings) {
    av_log_set_level(AV_LOG_INFO);
    VideoHandle session(new VideoSession());
    session->settings = settings;
    AVFormatContext* rawInputCtx = nullptr;
    const int openResult = avformat_open_input(&rawInputCtx, settings.videoFile.c_str(), nullptr, nullptr);
    session->inputFormatCtx.reset(rawInputCtx);
    checkError(openResult < 0, "Failed to open video");
    video_utils::checkAv(avformat_find_stream_info(session->inputFormatCtx.get(), nullptr), "Failed to read stream info");
    session->videoStreamIndex = findVideoStream(session->inputFormatCtx.get());
    checkError(session->videoStreamIndex == -1, "No video stream found");
    session->videoStream = session->inputFormatCtx->streams[session->videoStreamIndex];
    session->useHwDecoder = false;
    session->inputDecoderCtx = openDecoder(session->videoStream->codecpar, settings.useHwDecoder, session->useHwDecoder, session->videoStream->time_base);
    checkError(!session->inputDecoderCtx.get(), "Could not open video decoder");
    const int height = session->videoStream->codecpar->height;
    const int width = session->videoStream->codecpar->width;
    checkError((width & 1) != 0 || (height & 1) != 0, std::format("YUV 4:2:0 video requires even dimensions; input is {}x{}.", width, height));
    session->watermarkObj = createWatermarkObject(height, width, settings.watermarkPassword, settings.p, settings.psnr);
#if !defined(_USE_EIGEN_)
    session->hostFrame = std::make_unique<HostMemory<uint8_t>>(width * height * 3 / 2);
#endif
#if defined(_USE_EIGEN_)
    // video is always grayscale, initialize the output buffer to the gray variant (bad_variant_access fix)
    session->watermarkedFrame = ImageOutputBuffer(Gray8Buffer(height, width));
#endif
    return session;
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
int detectVideo(VideoSession* s) {
    checkError(is10bit(s->inputDecoderCtx.get(), s->videoStream) || isHDR(s->inputDecoderCtx.get()),
        "Video watermark detection supports only 8-bit SDR YUV 4:2:0 input. 10-bit and HDR detection are not supported.");
    return videoDispatcher(s, VideoMode::DETECT, false);
}

void optimizeThreadsForVideoEmbedding() {
#if defined(_USE_EIGEN_)
    eigen_utils::setThreadsToPhysicalCores();
#endif
}

} // namespace WatermarkCore
