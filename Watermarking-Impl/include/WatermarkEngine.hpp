#pragma once

#include "WatermarkTypes.hpp"
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
ImageHandle initImage(const std::string& imagePath, const std::string& watermarkDataPath, const int p, const float psnr);
void embedImage(ImageSession* session, MaskMethod method);
void prepareDetectionImage(ImageSession* session, MaskMethod method);
float detectLoadedImage(const ImageSession* session, MaskMethod method);
float detectEmbeddedBuffer(const ImageSession* session, MaskMethod method);
void saveImage(const ImageSession* session, const std::string& outPath, MaskMethod method);

// video processing functions
struct VideoSettings {
    std::string videoFile;
    std::string watermarkDataPath;
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