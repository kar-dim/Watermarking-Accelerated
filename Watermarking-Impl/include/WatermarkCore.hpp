#pragma once

#include "WatermarkTypes.hpp"
#include <cstddef>
#include <cstdint>
#include <memory>
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
ImageHandle createImageSession(const std::string& watermarkPassword, const int p, const float psnr);
PreloadedHandle preloadImageFromDisk(const std::string& imagePath, int deviceIndex = -1, bool captureOriginal = false);
void loadImage(ImageSession* session, const std::string& imagePath, bool captureOriginal = false);
void bindPreloadedImage(ImageSession* session, PreloadedHandle preloadedData);
std::pair<int, int> getImageDims(const ImageSession* s);
OriginalPixelData takeOriginalPixelData(ImageSession* session);
void embedImage(ImageSession* session, MaskMethod method);
void finish();
void prepareDetectionImage(ImageSession* session, MaskMethod method);
float detectLoadedImage(const ImageSession* session, MaskMethod method);
float detectEmbeddedBuffer(const ImageSession* session, MaskMethod method);
void saveImage(const ImageSession* session, const std::string& outPath, MaskMethod method);
void saveImageExact(const ImageSession* session, const std::string& outPath);
ExportHandle createReusableExportBuffer();
// on the Eigen backend this transfers the session output to the export buffer
// call embedImage again before reading, detecting, or saving the session output
// this also transfers alpha ownership, bind a new image before exporting again
void exportForSave(ImageSession* session, ExportedImage* reusableBuffer, MaskMethod method);
void flushToDiskAsync(ExportedImage* handle, const std::string& outPath, MaskMethod method);
SessionPixelData getSessionPixelData(const ImageSession* session);
// copy the column-major planar session pixels into a row-major display buffer
void copySessionPixelsForPreview(const SessionPixelData& source, uint8_t* destination, size_t bytesPerLine);
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
