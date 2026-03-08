#pragma once

#include "WatermarkTypes.hpp"
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
bool initializeEnvironment(const int openclDevice = 0);
void updateSessionParams(ImageSession* session, const int p, const float psnr);
bool isOpenCLBackend();
std::string getDeviceName(const int deviceIndex = -1);
std::vector<std::string> getAvailableDevices();

// image processing functions
ImageHandle createImageSession(const std::string& watermarkPassword, const int p, const float psnr);
PreloadedHandle preloadImageFromDisk(const std::string& imagePath);
void loadImage(ImageSession* session, const std::string& imagePath);
void bindPreloadedImage(ImageSession* session, PreloadedHandle preloadedData);
std::pair<int, int> getImageDims(const ImageSession* s);
void embedImage(ImageSession* session, MaskMethod method);
void prepareDetectionImage(ImageSession* session, MaskMethod method);
float detectLoadedImage(const ImageSession* session, MaskMethod method);
float detectEmbeddedBuffer(const ImageSession* session, MaskMethod method);
void saveImage(const ImageSession* session, const std::string& outPath, MaskMethod method);
ExportHandle createReusableExportBuffer();
void exportForSave(const ImageSession* session, ExportedImage* reusableBuffer, MaskMethod method);
void flushToDiskAsync(ExportedImage* handle, const std::string& outPath, MaskMethod method);
const uint8_t* getSessionPixelData(const ImageSession* session, int& width, int& height, int& channels);

// video processing functions
struct VideoSettings {
    std::string videoFile;
    std::string watermarkPassword;
    int p;
    float psnr;
    int watermarkInterval;
    std::string hwDecoder;
    bool useHwEncoder;
    std::string encodeOptions;
    std::string encodeOutputPath;
};

VideoHandle initVideo(const VideoSettings& settings);
int embedVideo(VideoSession* session);
int detectVideo(VideoSession* session);
} // namespace WatermarkCore