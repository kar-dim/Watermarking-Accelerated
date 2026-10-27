#pragma once

#include "VideoObserver.hpp"

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <future>
#include <memory>
#include <mutex>
#include <stop_token>
#include <string>
#include <utility>
#include <vector>

/*!
 *  \brief  Main interface for Watermarking operations, including image loading, embedding, detection, and video processing
 *  \author Dimitris Karatzas
 */
namespace WatermarkCore {
struct ImageSession;
struct VideoSession;
struct PreloadedImage;
struct ExportedImage;
struct SessionPixelData {
    std::vector<uint8_t> pixels;
    int width = 0;
    int height = 0;
    int channels = 0;
};
struct PreviewFormat {
    int width = 0;
    int height = 0;
    int channels = 0;
};
struct OriginalPixelData {
    std::vector<uint8_t> pixels;
    int width = 0;
    int height = 0;
    int channels = 0;
};
// clang-format off
struct VideoSessionDeleter { void operator()(VideoSession* s) const; };
struct ImageSessionDeleter { void operator()(ImageSession* s) const; };
struct PreloadedImageDeleter { void operator()(PreloadedImage* p) const; };
struct ExportedImageDeleter { void operator()(ExportedImage* p) const;};
// clang-format on
using VideoHandle = std::unique_ptr<VideoSession, VideoSessionDeleter>;
using ImageHandle = std::unique_ptr<ImageSession, ImageSessionDeleter>;
using PreloadedHandle = std::unique_ptr<PreloadedImage, PreloadedImageDeleter>;
using ExportHandle = std::unique_ptr<ExportedImage, ExportedImageDeleter>;

// Device selection is local to the calling thread, and by that allocations RETAIN their owning device
bool initializeEnvironment(const int deviceIndex = 0);
void updateSessionParams(ImageSession* session, const int p, const float psnr);
bool isOpenCLBackend();
std::string getBackendName();
void buildOpenCLKernels();
std::string getDeviceName(const int deviceIndex = -1);
std::vector<std::string> getAvailableDevices();
int getCurrentDeviceIndex();

bool hasSupportedImageExtension(const std::filesystem::path& path);
std::vector<std::filesystem::path> getValidImageFiles(const std::filesystem::path& inputDir);
ImageHandle createImageSession(const std::string& watermarkPassword, const int p, const float psnr);
PreloadedHandle preloadImageFromDisk(const std::string& imagePath, int deviceIndex = -1, bool captureOriginal = false);
void loadImage(ImageSession* session, const std::string& imagePath, bool captureOriginal = false);
void bindPreloadedImage(ImageSession* session, PreloadedHandle preloadedData);
std::pair<int, int> getImageDims(const ImageSession* s);
OriginalPixelData takeOriginalPixelData(ImageSession* session);
void embedImage(ImageSession* session);
void finish();
void prepareDetectionImage(ImageSession* session);
float detectLoadedImage(const ImageSession* session);
float detectEmbeddedBuffer(const ImageSession* session);
void saveImage(const ImageSession* session, const std::string& outPath);
void saveImageExact(const ImageSession* session, const std::string& outPath);

// loads images in batch order on up to depth threads, 0 uses CPU core count
class ImagePrefetcher {
  public:
    ImagePrefetcher(std::vector<std::filesystem::path> files, int deviceIndex, size_t depth = 0);
    ImagePrefetcher(const ImagePrefetcher&) = delete;
    ImagePrefetcher& operator=(const ImagePrefetcher&) = delete;
    // load error rethrown on calling thread, subsequent loads continue
    PreloadedHandle next();

  private:
    std::vector<std::filesystem::path> files;
    int deviceIndex;
    size_t launched = 0;
    // futures join before destruction
    std::deque<std::future<PreloadedHandle>> pending;
    void launch();
};

struct SaveResult {
    size_t id;
    std::string error;
};

// background save queue on up to maxSaves threads, 0 uses CPU core count
class ImageSaver {
  public:
    explicit ImageSaver(size_t maxSaves = 0);
    ImageSaver(size_t maxSaves, size_t maxInFlightBytes);
    // wait for in-flight saves
    ~ImageSaver();
    ImageSaver(const ImageSaver&) = delete;
    ImageSaver& operator=(const ImageSaver&) = delete;
    void save(ImageSession* session, const std::string& outPath, size_t id);
    void saveExact(ImageSession* session, const std::string& outPath, size_t id);
    // finished saves since last call in completion order
    std::vector<SaveResult> takeFinished();
    // wait for all saves and return finished results
    std::vector<SaveResult> finish();

  private:
    void saveImpl(ImageSession* session, const std::string& outPath, size_t id, bool exactPath);
    struct Slot {
        ExportHandle buffer;
        std::future<void> task;
    };
    std::vector<Slot> saveSlots;
    std::vector<size_t> freeSlots;
    std::vector<SaveResult> finished;
    std::mutex mutex;
    std::condition_variable slotFreed;
    size_t inFlightBytes = 0;
    size_t maxInFlightBytes = 0;
};
ExportHandle createReusableExportBuffer();
// on Eigen transfers output to export buffer, call embedImage again before session reuse
// transfers alpha ownership, bind new image before exporting again
void exportForSave(ImageSession* session, ExportedImage* reusableBuffer);
void flushToDiskAsync(ExportedImage* handle, const std::string& outPath);
SessionPixelData getSessionPixelData(const ImageSession* session);
// size and channels (1, 3 or 4 with alpha) for copySessionPreview
PreviewFormat getSessionPreviewFormat(const ImageSession* session);
// copies output into row-major interleaved display buffer without touching padding
void copySessionPreview(const ImageSession* session, uint8_t* destination, size_t bytesPerLine);
void optimizeThreadsForVideoEmbedding();

struct VideoSourceInfo {
    int width = 0;
    int height = 0;
    int bitDepth = 0;
    double framesPerSecond = 0.0;
    double durationSeconds = -1.0;
    std::int64_t frames = -1;
    std::int64_t bytes = -1;
    std::string codec;
    std::string pixelFormat;
    std::string dynamicRange;
    unsigned audioTracks = 0;
    unsigned subtitleTracks = 0;
    unsigned attachments = 0;
    unsigned chapters = 0;
};

VideoSourceInfo probeVideoSource(const std::string& path, std::stop_token stop = {});
void validateVideoOutputPath(const std::string& path);

struct VideoSettings {
    std::string videoFile;
    std::string watermarkPassword;
    int p;
    float psnr;
    int watermarkInterval;
    bool useHwDecoder;
    bool useHwEncoder;
    // empty string selects GPU default hardware encoder when useHwEncoder is true
    std::string encodeOptions;
    std::string encodeOutputPath;
    VideoObserver* observer = nullptr;
    std::stop_token stopToken;
    int cpuThreads = 0;
};

// codec type only, device support is checked when encoding starts
void validateVideoEncoderOptions(const std::string& options, bool hardware);
// querying another device does not switch the active compute context
std::string defaultVideoEncoderOptions(bool hardware, int deviceIndex = 0);
VideoHandle initVideo(const VideoSettings& settings);
int embedVideo(VideoSession* session);
int embedVideo(VideoSession* session, bool replaceExistingOutput);
int detectVideo(VideoSession* session);
} // namespace WatermarkCore
