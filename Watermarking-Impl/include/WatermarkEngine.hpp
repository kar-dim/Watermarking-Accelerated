#pragma once

#include "WatermarkTypes.hpp"
#include <cstdint>
#include <memory>
#include <string>

/*!
 *  \brief  Main interface for Watermarking operations, including image loading, embedding, detection, and video processing.
 *  \author Dimitris Karatzas
 */
namespace WatermarkEngine {
// forward declarations of session structs and their deleters for RAII management
struct ImageSession;
struct VideoSession;
struct VideoSessionDeleter {
    void operator()(VideoSession* s) const;
};
struct ImageSessionDeleter {
    void operator()(ImageSession* s) const;
};
using VideoHandle = std::unique_ptr<VideoSession, VideoSessionDeleter>;
using ImageHandle = std::unique_ptr<ImageSession, ImageSessionDeleter>;

// initialize the environment (set OpenCL device, initialize ArrayFire/OpenMP etc)
void initializeEnvironment(const int openclDevice = 0);

// image processing functions
ImageHandle createImageSession(const uint32_t watermarkSeed, const int p, const float psnr);
void loadImage(ImageSession* session, const std::string& imagePath);
void embedImage(ImageSession* session, MaskMethod method);
void prepareDetectionImage(ImageSession* session, MaskMethod method);
float detectLoadedImage(const ImageSession* session, MaskMethod method);
float detectEmbeddedBuffer(const ImageSession* session, MaskMethod method);
void saveImage(const ImageSession* session, const std::string& outPath, MaskMethod method);

// video processing functions
struct VideoSettings {
    std::string videoFile;
    uint32_t watermarkSeed;
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
} // namespace WatermarkEngine