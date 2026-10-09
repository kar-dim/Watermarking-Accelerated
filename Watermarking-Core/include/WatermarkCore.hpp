#pragma once

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

/*!
 *  \brief  Main interface for Watermarking operations, including image loading, embedding, detection, and video processing.
 *  \author Dimitris Karatzas
 */
namespace WatermarkCore {
// forward declarations of session structs and their deleters for RAII management
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

// OpenCL device switches require releasing all live image/session/export buffers first
// environment and params functions
bool initializeEnvironment(const int deviceIndex = 0);
void updateSessionParams(ImageSession* session, const int p, const float psnr);
bool isOpenCLBackend();
std::string getBackendName();
void buildOpenCLKernels();
std::string getDeviceName(const int deviceIndex = -1);
std::vector<std::string> getAvailableDevices();
int getCurrentDeviceIndex();

// image processing functions
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

// loads the images of a batch in their order on up to "depth" background threads. Depth 0 picks it from the CPU thread count
class ImagePrefetcher {
  public:
    ImagePrefetcher(std::vector<std::filesystem::path> files, int deviceIndex, size_t depth = 0);
    ImagePrefetcher(const ImagePrefetcher&) = delete;
    ImagePrefetcher& operator=(const ImagePrefetcher&) = delete;
    // the next image, its load error is rethrown here (the loads of the images after it keep running)
    PreloadedHandle next();

  private:
    std::vector<std::filesystem::path> files;
    int deviceIndex;
    size_t launched = 0;
    // the futures join on destruction, before the files they read are released
    std::deque<std::future<PreloadedHandle>> pending;
    void launch();
};

struct SaveResult {
    size_t id;
    std::string error;
};

// saves embedded images in the background on up to "maxSaves" threads, each with its own export buffer. maxSaves 0 picks it from the CPU thread count
class ImageSaver {
  public:
    explicit ImageSaver(size_t maxSaves = 0);
    // waits for the saves still running
    ~ImageSaver();
    ImageSaver(const ImageSaver&) = delete;
    ImageSaver& operator=(const ImageSaver&) = delete;
    // waits while every slot is busy, then exports the session output and writes it to outPath in the background
    void save(ImageSession* session, const std::string& outPath, size_t id);
    // the saves that finished since the last call, in the order they finished
    std::vector<SaveResult> takeFinished();
    // waits for all saves, then returns the ones that finished since the last call
    std::vector<SaveResult> finish();

  private:
    struct Slot {
        ExportHandle buffer;
        std::future<void> task;
    };
    std::vector<Slot> saveSlots;
    std::vector<size_t> freeSlots;
    std::vector<SaveResult> finished;
    std::mutex mutex;
    std::condition_variable slotFreed;
};
ExportHandle createReusableExportBuffer();
// on the Eigen backend this transfers the session output to the export buffer
// call embedImage again before reading, detecting, or saving the session output
// this also transfers alpha ownership, bind a new image before exporting again
void exportForSave(ImageSession* session, ExportedImage* reusableBuffer);
void flushToDiskAsync(ExportedImage* handle, const std::string& outPath);
SessionPixelData getSessionPixelData(const ImageSession* session);
// size and channels (1, 3 or 4 with preserved alpha) of the embedded output for copySessionPreview
PreviewFormat getSessionPreviewFormat(const ImageSession* session);
// copy the session output into a row-major interleaved display buffer (rows start every bytesPerLine, the padding is not touched)
void copySessionPreview(const ImageSession* session, uint8_t* destination, size_t bytesPerLine);
void optimizeThreadsForVideoEmbedding();

// video processing functions
struct VideoSettings {
    std::string videoFile;
    std::string watermarkPassword;
    int p;
    float psnr;
    int watermarkInterval;
    bool useHwDecoder;
    bool useHwEncoder;
    std::string encodeOptions;
    std::string encodeOutputPath;
};

VideoHandle initVideo(const VideoSettings& settings);
int embedVideo(VideoSession* session);
int detectVideo(VideoSession* session);
} // namespace WatermarkCore
