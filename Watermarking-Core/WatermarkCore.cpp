#include "AvUtil.hpp"
#include "buffer.hpp"
#include "CheckedSize.hpp"
#include "EncodeOptions.hpp"
#include "common_utils.hpp"
#include "ImageFileBuffer.hpp"
#include "include/WatermarkCore.hpp"
#include "utils.hpp"
#include "video_utils.hpp"
#include "VideoProcessingContext.hpp"
#include "WatermarkBase.hpp"
#include "WatermarkFactory.hpp"
#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <format>
#include <iostream>
#include <memory>
#include <omp.h>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#if defined(_USE_CUDA_)
#include "cuda_utils.hpp"
#include "CudaArray.hpp"
#include "CudaCheck.hpp"
#include "CudaStreamManager.hpp"
#include "HostMemory.hpp"
#include <cuda_runtime.h>
#elif defined(_USE_OPENCL_)
#include "HostMemory.hpp"
#include "OclQueueManager.hpp"
#include "opencl_utils.hpp"
#elif defined(_USE_EIGEN_)
#include "eigen_utils.hpp"
#include <cstring>
#include <intrin.h>
#endif

extern "C" {
#include "libavcodec/codec_par.h"
#include "libavformat/avformat.h"
#include "libavutil/log.h"
}

/*!
 *  \brief  Public C-style and C++ API implementation for image and video watermarking sessions
 *  \author Dimitris Karatzas
 */

using namespace video_utils;
using namespace CommonUtils;
using namespace InternalUtils;

using std::string;

namespace WatermarkCore {

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
    bool outputReady = false;
    bool detectionReady = false;
};

struct PreloadedImage {
    ImageFileBuffer buffer;
};

struct ExportedImage {
    ImageOutputBuffer finalPixels;
    std::vector<uint8_t> alpha;
};

namespace {
void checkOutput(const ImageSession* s);
}

void ImageSessionDeleter::operator()(ImageSession* s) const { delete s; }

void PreloadedImageDeleter::operator()(PreloadedImage* p) const { delete p; }

void ExportedImageDeleter::operator()(ExportedImage* p) const { delete p; }

ExportHandle createReusableExportBuffer() { return ExportHandle(new ExportedImage()); }

// move CPU output into idle export buffer, GPU builds clone device data
void exportForSave(ImageSession* s, ExportedImage* p) {
    checkOutput(s);
#if defined(_USE_GPU_)
    p->finalPixels = s->watermarkBuffer.clone();
#else
    if (!p->finalPixels.matches(s->currentRows, s->currentCols, s->currentIsRGB))
        p->finalPixels = s->currentIsRGB ? ImageOutputBuffer(eigen_utils::makeEigenRGBu8(s->currentRows, s->currentCols)) : ImageOutputBuffer(Gray8Buffer(s->currentRows, s->currentCols));
    std::swap(p->finalPixels, s->watermarkBuffer);
    s->outputReady = false;
#endif
    // batch export consumes alpha plane, next image bind replaces input buffer
    p->alpha = std::move(s->imgBuffer.alphaChannel);
    s->detectionReady = false;
}

namespace {
// appended to filename before extension
constexpr const char* kWatermarkedSuffix = "W_ME";

// async workers run serially to avoid nested OpenMP teams, restore setting when done
class SerialOpenMPScope {
  public:
    SerialOpenMPScope() : previous(omp_get_max_threads()) { omp_set_num_threads(1); }
    ~SerialOpenMPScope() { omp_set_num_threads(previous); }
    SerialOpenMPScope(const SerialOpenMPScope&) = delete;
    SerialOpenMPScope& operator=(const SerialOpenMPScope&) = delete;

  private:
    int previous;
};

// NaN and infinity would reach the pixels through the watermark strength
void checkPsnr(const float psnr) {
    if (!std::isfinite(psnr) || psnr <= 0.0f)
        throw std::invalid_argument("PSNR must be a finite number greater than 0");
}

void checkPredictionOrder(const int p) {
    if (p != 3 && p != 5 && p != 7 && p != 9)
        throw std::invalid_argument("Unsupported value for p. Allowed p values: 3, 5, 7, 9");
}

void checkLoaded(const ImageSession* s) {
    if (!s || !s->watermarkObj)
        throw std::logic_error("Load an image before processing it");
}

void checkOutput(const ImageSession* s) {
    checkLoaded(s);
    if (!s->outputReady)
        throw std::logic_error("Embed an image before reading or saving its output");
}

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

SessionPixelData getSessionPixelData(const ImageSession* session) {
    checkOutput(session);
    return extractPixelData(session->watermarkBuffer);
}

#if defined(_USE_EIGEN_)
namespace {
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

// column-major planes into row-major interleaved display buffer
void interleavePlanes(const uint8_t* red, const uint8_t* green, const uint8_t* blue, const int rows, const int cols, const int channels, uint8_t* destination, const size_t bytesPerLine) {
    // small row tiles keep cache lines hot while reading planar columns
    constexpr int tileRows = 32;
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
            // tail columns that do not fit 8x8 block
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
        // tail rows that do not fit 8x8 block
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
} // namespace
#endif

PreviewFormat getSessionPreviewFormat(const ImageSession* session) {
    checkOutput(session);
    const ImageOutputBuffer& buffer = session->watermarkBuffer;
    const bool hasAlpha = !session->imgBuffer.alphaChannel.empty();
#if defined(_USE_GPU_)
    return {buffer.getCols(), buffer.getRows(), hasAlpha ? 4 : buffer.getChannels()};
#else
    if (buffer.isRGB())
        return {static_cast<int>(buffer.getRGB()[0].cols()), static_cast<int>(buffer.getRGB()[0].rows()), hasAlpha ? 4 : 3};
    return {static_cast<int>(buffer.getGray().cols()), static_cast<int>(buffer.getGray().rows()), 1};
#endif
}

// opaque GPU previews interleave on device, RGBA previews also combine host alpha plane
void copySessionPreview(const ImageSession* session, uint8_t* destination, const size_t bytesPerLine) {
    const auto [width, height, channels] = getSessionPreviewFormat(session);
    if (width <= 0 || height <= 0 || (channels != 1 && channels != 3 && channels != 4) || !destination || bytesPerLine < static_cast<size_t>(width) * channels)
        throw std::invalid_argument("Invalid session pixels for preview");
    InternalUtils::checkedProduct(bytesPerLine, static_cast<size_t>(height));
    if (channels == 4) {
        const auto pixels = getSessionPixelData(session);
        const size_t planeSize = static_cast<size_t>(width) * height;
        if (pixels.channels != 3 || session->imgBuffer.alphaChannel.size() != planeSize)
            throw std::logic_error("Invalid alpha plane for preview");
        for (int row = 0; row < height; ++row)
            for (int col = 0; col < width; ++col) {
                const size_t source = static_cast<size_t>(col) * height + row;
                uint8_t* pixel = destination + static_cast<size_t>(row) * bytesPerLine + static_cast<size_t>(col) * 4;
                for (int channel = 0; channel < 3; ++channel)
                    pixel[channel] = pixels.pixels[static_cast<size_t>(channel) * planeSize + source];
                pixel[3] = session->imgBuffer.alphaChannel[static_cast<size_t>(row) * width + col];
            }
        return;
    }
    const ImageOutputBuffer& buffer = session->watermarkBuffer;
#if defined(_USE_CUDA_)
    CUDA_CHECK(cudaSetDevice(buffer.getDeviceIndex()));
    const cudaStream_t stream = CudaStreamManager::getInstance().getComputeStream();
    CudaArray<uint8_t> interleaved(height, width, channels, stream);
    cuda_utils::launchColMajorToInterleavedU8Kernel(buffer.data(), interleaved.data(), width, height, channels, stream);
    interleaved.toHostPitched(destination, width * channels, bytesPerLine);
#elif defined(_USE_OPENCL_)
    const OclQueueManager::Binding binding(buffer.getOwner());
    auto& mgr = OclQueueManager::getInstance();
    OclArray<uint8_t> interleaved(height, width, channels, mgr.getQueueRaw());
    cl_utils::launchColMajorToInterleavedU8(buffer.clBuffer(), interleaved.clBuffer(), width, height, channels, mgr.getQueue());
    interleaved.toHostPitched(destination, width * channels, bytesPerLine);
#else
    if (channels == 3) {
        const auto& rgb = buffer.getRGB();
        interleavePlanes(rgb[0].data(), rgb[1].data(), rgb[2].data(), height, width, channels, destination, bytesPerLine);
    } else {
        interleavePlanes(buffer.getGray().data(), nullptr, nullptr, height, width, channels, destination, bytesPerLine);
    }
#endif
}

bool initializeEnvironment(const int deviceIndex) {
    bool deviceSetSuccess = true;
#if defined(_USE_OPENCL_)
    const auto devices = OclQueueManager::enumerateDevices();
    const int selectedDevice = deviceIndex >= 0 && deviceIndex < static_cast<int>(devices.size()) ? deviceIndex : 0;
    if (selectedDevice != deviceIndex) {
        std::cout << "NOTE: Invalid OpenCL device index, using default 0\n";
        deviceSetSuccess = false;
    }
    OclQueueManager::initialize(selectedDevice);
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
    CudaStreamManager::getInstance().getComputeStream(); // init device stream and pool
#endif
#if defined(_USE_EIGEN_)
#pragma omp parallel
    {}
    std::cout << info("Using " + std::to_string(omp_get_max_threads()) + " parallel threads for Watermark calculations.\n");
#endif
    return deviceSetSuccess;
}

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
}

void updateSessionParams(ImageSession* s, const int p, const float psnr) {
#if defined(_USE_OPENCL_)
    const OclQueueManager::Binding binding(s->imgBuffer.image.getOwner());
#endif
    checkPredictionOrder(p);
    checkPsnr(psnr);
    if (s->p == p && s->psnr == psnr)
        return;
    // construct before commit, failure leaves old engine usable
    if (s->watermarkObj && s->p != p)
        s->watermarkObj = createWatermarkObject(s->currentRows, s->currentCols, s->watermarkPassword, p, psnr);
    else if (s->watermarkObj)
        s->watermarkObj->updatePsnr(psnr);
    s->p = p;
    s->psnr = psnr;
    s->detectGrayBuffer = ImageBuffer();
    s->detectionReady = false;
}

ImageHandle createImageSession(const string& watermarkPassword, const int p, const float psnr) {
    checkPredictionOrder(p);
    checkPsnr(psnr);
    ImageHandle session(new ImageSession());
    session->watermarkPassword = watermarkPassword;
    session->p = p;
    session->psnr = psnr;
    return session;
}

PreloadedHandle preloadImageFromDisk(const string& imagePath, const int deviceIndex, const bool captureOriginal) {
#if defined(_USE_CUDA_)
    // each prefetch thread must activate owning session GPU
    if (deviceIndex >= 0)
        CUDA_CHECK(cudaSetDevice(deviceIndex));
#elif defined(_USE_OPENCL_)
    if (deviceIndex >= 0)
        OclQueueManager::initialize(deviceIndex);
#endif
    PreloadedHandle p(new PreloadedImage());
    p->buffer = InternalUtils::loadImage(imagePath, captureOriginal);
    return p;
}

ImagePrefetcher::ImagePrefetcher(std::vector<std::filesystem::path> files, const int deviceIndex, const size_t depth) : files(std::move(files)), deviceIndex(deviceIndex) {
    const size_t loads = depth > 0 ? depth : static_cast<size_t>(batchLoadThreads());
    for (size_t i = 0; i < loads; i++)
        launch();
}

PreloadedHandle ImagePrefetcher::next() {
    if (pending.empty())
        throw std::logic_error("No images left to load");
    std::future<PreloadedHandle> image = std::move(pending.front());
    pending.pop_front();
    launch();
    return image.get();
}

ImageSaver::ImageSaver(const size_t maxSaves) : ImageSaver(maxSaves, 0) {}

ImageSaver::ImageSaver(const size_t maxSaves, const size_t maxInFlightBytes) : maxInFlightBytes(maxInFlightBytes) {
    const size_t count = maxSaves > 0 ? maxSaves : std::max(std::thread::hardware_concurrency(), 1u);
    saveSlots.resize(count);
    for (size_t i = 0; i < count; i++) {
        saveSlots[i].buffer = createReusableExportBuffer();
        freeSlots.push_back(count - 1 - i);
    }
}

// save tasks must end before members are destroyed
ImageSaver::~ImageSaver() {
    for (auto& slot : saveSlots)
        if (slot.task.valid())
            slot.task.wait();
}

void ImageSaver::save(ImageSession* session, const string& outPath, const size_t id) { saveImpl(session, outPath, id, false); }

void ImageSaver::saveExact(ImageSession* session, const string& outPath, const size_t id) { saveImpl(session, outPath, id, true); }

void ImageSaver::saveImpl(ImageSession* session, const string& outPath, const size_t id, const bool exactPath) {
    const size_t pixels = InternalUtils::checkedProduct(static_cast<size_t>(session->currentRows), static_cast<size_t>(session->currentCols));
    const size_t bytes = InternalUtils::checkedProduct(pixels, session->currentIsRGB ? 9u : 3u) + session->imgBuffer.alphaChannel.size();
    std::unique_lock lock(mutex);
    // One oversized image can proceed alone, otherwise include export, conversion and host pixels in the budget
    slotFreed.wait(lock, [&] { return !freeSlots.empty() && (!maxInFlightBytes || !inFlightBytes || (bytes <= maxInFlightBytes && inFlightBytes <= maxInFlightBytes - bytes)); });
    const size_t index = freeSlots.back();
    freeSlots.pop_back();
    inFlightBytes += bytes;
    lock.unlock();
    Slot& slot = saveSlots[index];
    try {
        // join thread once slot finishes
        if (slot.task.valid())
            slot.task.get();
        exportForSave(session, slot.buffer.get());
        slot.task = std::async(std::launch::async, [this, index, id, outPath, exactPath, bytes] {
            string error;
            try {
                if (exactPath) {
                    const SerialOpenMPScope serial;
                    const auto& output = *saveSlots[index].buffer;
                    InternalUtils::saveImage(outPath, "", output.finalPixels, output.alpha);
                } else {
                    flushToDiskAsync(saveSlots[index].buffer.get(), outPath);
                }
            } catch (const std::exception& e) { error = *e.what() ? e.what() : "Unknown save error"; } catch (...) {
                error = "Unknown save error";
            }
            // Idle slots must not retain GPU buffers outside the active save budget
            if (maxInFlightBytes)
                *saveSlots[index].buffer = {};
            {
                std::lock_guard finishedLock(mutex);
                finished.push_back(SaveResult{id, std::move(error)});
                freeSlots.push_back(index);
                inFlightBytes -= bytes;
            }
            slotFreed.notify_all();
        });
    } catch (...) {
        if (maxInFlightBytes)
            *slot.buffer = {};
        {
            std::lock_guard freeLock(mutex);
            freeSlots.push_back(index);
            inFlightBytes -= bytes;
        }
        slotFreed.notify_all();
        throw;
    }
}

std::vector<SaveResult> ImageSaver::takeFinished() {
    std::lock_guard lock(mutex);
    return std::exchange(finished, {});
}

std::vector<SaveResult> ImageSaver::finish() {
    std::unique_lock lock(mutex);
    slotFreed.wait(lock, [&] { return freeSlots.size() == saveSlots.size(); });
    return std::exchange(finished, {});
}

void ImagePrefetcher::launch() {
    if (launched < files.size()) {
        pending.push_back(std::async(std::launch::async, [path = files[launched].string(), device = deviceIndex] {
            const SerialOpenMPScope serial;
            return preloadImageFromDisk(path, device, false);
        }));
        launched++;
    }
}

std::pair<int, int> getImageDims(const ImageSession* s) { return {s->currentRows, s->currentCols}; }
OriginalPixelData takeOriginalPixelData(ImageSession* s) {
    return {std::move(s->imgBuffer.originalPreview), static_cast<int>(s->imgBuffer.cols), static_cast<int>(s->imgBuffer.rows), s->imgBuffer.previewChannels};
}

void bindPreloadedImage(ImageSession* s, PreloadedHandle preloadedData) {
    if (!preloadedData)
        throw std::invalid_argument("No preloaded image supplied");
#if defined(_USE_OPENCL_)
    const OclQueueManager::Binding binding(preloadedData->buffer.image.getOwner());
    const bool deviceChanged = !s->imgBuffer.image.empty() && s->imgBuffer.image.getOwner() != preloadedData->buffer.image.getOwner();
#else
    const bool deviceChanged = false;
#endif
    const auto rows = preloadedData->buffer.rows;
    const auto cols = preloadedData->buffer.cols;
    const auto isRGB = preloadedData->buffer.isRGB;
    // lazy init when dimensions change or on first frame
    if (!s->watermarkObj || s->currentRows != rows || s->currentCols != cols || s->currentIsRGB != isRGB || deviceChanged) {
        auto engine = createWatermarkObject(rows, cols, s->watermarkPassword, s->p, s->psnr);
#if defined(_USE_EIGEN_)
        auto output = isRGB ? ImageOutputBuffer(eigen_utils::makeEigenRGBu8(rows, cols)) : ImageOutputBuffer(Gray8Buffer(rows, cols));
        s->watermarkBuffer = std::move(output);
#endif
        s->watermarkObj = std::move(engine);
#if defined(_USE_OPENCL_)
        if (deviceChanged)
            s->watermarkBuffer = {};
#endif
        s->currentRows = rows;
        s->currentCols = cols;
        s->currentIsRGB = isRGB;
    }
    s->imgBuffer = std::move(preloadedData->buffer);
    s->detectGrayBuffer = ImageBuffer();
    s->outputReady = false;
    s->detectionReady = false;
}

void loadImage(ImageSession* session, const string& imagePath, const bool captureOriginal) { bindPreloadedImage(session, preloadImageFromDisk(imagePath, -1, captureOriginal)); }

void embedImage(ImageSession* s) {
    checkLoaded(s);
    s->outputReady = false;
    s->detectGrayBuffer = ImageBuffer();
    s->detectionReady = false;
    if (s->imgBuffer.isRGB)
        s->watermarkObj->makeWatermark(s->imgBuffer.image, s->imgBuffer.rgbImage, s->watermarkBuffer);
    else
        s->watermarkObj->makeWatermark(s->imgBuffer.image, s->watermarkBuffer, WatermarkBase::Layout::ColMajor);
    s->outputReady = true;
}

void finish() {
#if defined(_USE_CUDA_)
    CUDA_CHECK(cudaStreamSynchronize(CudaStreamManager::getInstance().getComputeStream()));
#elif defined(_USE_OPENCL_)
    OclQueueManager::getInstance().finish();
#endif
}

void prepareDetectionImage(ImageSession* s) {
    checkOutput(s);
#if defined(_USE_OPENCL_)
    const OclQueueManager::Binding binding(s->watermarkBuffer.getOwner());
#endif
    s->detectionReady = false;
    s->detectGrayBuffer = InternalUtils::castToFloatGray(s->watermarkBuffer, s->imgBuffer.isRGB);
    // sync device to measure full completion instead of async upload
#if defined(_USE_CUDA_)
    CUDA_CHECK(cudaStreamSynchronize(CudaStreamManager::getInstance().getComputeStream()));
#elif defined(_USE_OPENCL_)
    OclQueueManager::getInstance().finish();
#endif
    s->detectionReady = true;
}

float detectLoadedImage(const ImageSession* s) {
    checkLoaded(s);
    return s->watermarkObj->detectWatermark(s->imgBuffer.image);
}

float detectEmbeddedBuffer(const ImageSession* s) {
    checkLoaded(s);
    if (!s->detectionReady)
        throw std::logic_error("Prepare the detection image before detecting from the embedded buffer");
    return s->watermarkObj->detectWatermark(s->detectGrayBuffer);
}

void saveImage(const ImageSession* s, const string& outPath) {
    checkOutput(s);
    InternalUtils::saveImage(outPath, kWatermarkedSuffix, s->watermarkBuffer, s->imgBuffer.alphaChannel);
}

void flushToDiskAsync(ExportedImage* handle, const string& outPath) {
    const SerialOpenMPScope serial;
    InternalUtils::saveImage(outPath, kWatermarkedSuffix, handle->finalPixels, handle->alpha);
}

void saveImageExact(const ImageSession* s, const string& outPath) {
    checkOutput(s);
    InternalUtils::saveImage(outPath, "", s->watermarkBuffer, s->imgBuffer.alphaChannel);
}

void VideoSessionDeleter::operator()(VideoSession* s) const { delete s; }

std::string defaultVideoEncoderOptions(const bool hardware, [[maybe_unused]] const int deviceIndex) {
    if (!hardware)
        return "-c:v libx265 -preset medium -crf 20";
#if defined(_USE_OPENCL_)
    return hardwareEncoderDefaults(OclQueueManager::deviceVendorId(deviceIndex));
#elif defined(_USE_CUDA_)
    return hardwareEncoderDefaults(0x10de);
#else
    throw std::invalid_argument("Eigen uses CPU video encoding");
#endif
}

void validateVideoEncoderOptions(const std::string& options, const bool hardware) {
#if defined(_USE_EIGEN_)
    if (hardware)
        throw std::invalid_argument("Eigen uses CPU encoding, select a GPU build for hardware encoding");
#endif
    if (hardware && options.empty())
        return;
    const auto parsed = parseEncodeOptions(options);
    try {
        findVideoEncoder(parsed.codecName, hardware);
    } catch (const std::runtime_error& error) { throw std::invalid_argument(error.what()); }
}

static VideoHandle openVideo(const VideoSettings& settings) {
    checkPsnr(settings.psnr);
#if defined(_USE_EIGEN_)
    if (settings.useHwEncoder)
        throw std::invalid_argument("Eigen uses CPU video decoding and encoding, select the CUDA or OpenCL build for hardware encoding");
#endif
    if (settings.watermarkInterval < 1)
        throw std::invalid_argument("Video watermark interval must be at least 1");
    av_log_set_level(AV_LOG_INFO);
    VideoHandle session(new VideoSession());
    session->settings = settings;
#if defined(_USE_OPENCL_)
    session->openclDevice = OclQueueManager::current();
#endif
#if defined(_USE_GPU_)
    session->deviceIndex = getCurrentDeviceIndex();
#endif
    checkCancelled(session.get());
    reportProgress(session.get(), VideoState::Setup);
    AVFormatContext* rawInputCtx = avformat_alloc_context();
    checkError(!rawInputCtx, "Failed to allocate video input");
    rawInputCtx->interrupt_callback = {interruptVideo, session.get()};
    const int openResult = avformat_open_input(&rawInputCtx, settings.videoFile.c_str(), nullptr, nullptr);
    session->inputFormatCtx.reset(rawInputCtx);
    checkCancelled(session.get());
    checkError(openResult < 0, "Failed to open video");
    video_utils::checkAv(avformat_find_stream_info(session->inputFormatCtx.get(), nullptr), "Failed to read stream info");
    session->videoStreamIndex = findVideoStream(session->inputFormatCtx.get());
    checkAv(session->videoStreamIndex, "No video stream found");
    session->videoStream = session->inputFormatCtx->streams[session->videoStreamIndex];
    session->useHwDecoder = false;
    session->inputDecoderCtx = openDecoder(session.get());
    checkCancelled(session.get());
    checkError(!session->inputDecoderCtx.get(), "Could not open video decoder");
    if (!session->useHwDecoder)
        session->decodedFormat = session->inputDecoderCtx->pix_fmt;
    session->frameRate = av_guess_frame_rate(session->inputFormatCtx.get(), session->inputFormatCtx->streams[session->videoStreamIndex], nullptr);
    session->nominalFrameDuration = nominalFrameDuration(session->frameRate, session->videoStream->time_base);
    // dimensions must fit chroma subsampling, checked with encoder, detection reads only luma
    const int height = session->videoStream->codecpar->height;
    const int width = session->videoStream->codecpar->width;
    checkCancelled(session.get());
    session->watermarkObj = createWatermarkObject(height, width, settings.watermarkPassword, settings.p, settings.psnr);
    reportProgress(session.get(), VideoState::Setup);
    return session;
}

VideoHandle initVideo(const VideoSettings& settings) {
    try {
        return openVideo(settings);
    } catch (const VideoCancelled&) {
        if (settings.observer) {
            VideoProgress progress;
            progress.state = VideoState::Cancelled;
            settings.observer->onProgress(progress);
        }
        throw;
    } catch (...) {
        if (settings.observer) {
            VideoProgress progress;
            progress.state = VideoState::Failed;
            settings.observer->onProgress(progress);
        }
        throw;
    }
}

// failed embed discards temporary file and preserves destination
int embedVideo(VideoSession* s) { return embedVideo(s, true); }

int embedVideo(VideoSession* s, const bool replaceExistingOutput) {
    try {
#if defined(_USE_OPENCL_)
        const OclQueueManager::Binding binding(s->openclDevice);
#elif defined(_USE_CUDA_)
        CUDA_CHECK(cudaSetDevice(s->deviceIndex));
#endif
        checkCancelled(s);
        video_utils::initOutputEncoder(s);
        reportProgress(s, VideoState::Processing);
        const bool needsFilter = initFilterGraph(s);
        const int framesProcessed = videoDispatcher(s, VideoMode::EMBED, needsFilter);
        checkError(framesProcessed <= 0, "No video frames were decoded");
        reportProgress(s, VideoState::Finalizing);
        video_utils::flushAndFinalize(s, replaceExistingOutput);
        reportProgress(s, VideoState::Completed);
        return framesProcessed;
    } catch (const VideoCancelled&) {
        video_utils::discardOutput(s);
        reportProgress(s, VideoState::Cancelled);
        throw;
    } catch (...) {
        reportProgress(s, VideoState::Failed);
        video_utils::discardOutput(s);
        throw;
    }
}

int detectVideo(VideoSession* s) {
    try {
#if defined(_USE_OPENCL_)
        const OclQueueManager::Binding binding(s->openclDevice);
#elif defined(_USE_CUDA_)
        CUDA_CHECK(cudaSetDevice(s->deviceIndex));
#endif
        checkCancelled(s);
        checkError(is10bit(s->inputDecoderCtx.get(), s->videoStream) || isHDR(s->inputDecoderCtx.get()),
            "Video watermark detection supports only 8-bit SDR input. High-bit-depth and HDR detection are not supported.");
        reportProgress(s, VideoState::Processing);
        const int frames = videoDispatcher(s, VideoMode::DETECT, initFilterGraph(s));
        checkError(frames <= 0, "No video frames were decoded");
        checkCancelled(s);
        reportProgress(s, VideoState::Completed);
        return frames;
    } catch (const VideoCancelled&) {
        reportProgress(s, VideoState::Cancelled);
        throw;
    } catch (...) {
        reportProgress(s, VideoState::Failed);
        throw;
    }
}

void optimizeThreadsForVideoEmbedding() {
#if defined(_USE_EIGEN_)
    eigen_utils::setThreadsToPhysicalCores();
#endif
}

bool hasSupportedImageExtension(const std::filesystem::path& path) {
    static constexpr std::array<std::string_view, 7> validExts{".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"};
    std::string ext = path.extension().string();
    std::ranges::transform(ext, ext.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return std::ranges::find(validExts, ext) != validExts.end();
}

std::vector<std::filesystem::path> getValidImageFiles(const std::filesystem::path& inputDir) {
    std::vector<std::filesystem::path> validFiles;
    if (!std::filesystem::exists(inputDir) || !std::filesystem::is_directory(inputDir))
        return validFiles;
    for (const auto& entry : std::filesystem::directory_iterator(inputDir)) {
        if (!entry.is_regular_file())
            continue;
        if (hasSupportedImageExtension(entry.path()))
            validFiles.push_back(entry.path());
    }
    std::sort(validFiles.begin(), validFiles.end());
    return validFiles;
}

} // namespace WatermarkCore
