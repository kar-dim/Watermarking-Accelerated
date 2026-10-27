#pragma once

/*!
 * \brief Video progress, detection results and live preview callbacks
 * \author Dimitris Karatzas
 */

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace WatermarkCore {
enum class VideoState { Setup, Processing, Finalizing, Completed, Cancelled, Failed };

struct VideoProgress {
    VideoState state = VideoState::Setup;
    std::uint64_t frames = 0;
    std::uint64_t selectedFrames = 0;
    double fraction = -1.0;
    double elapsedSeconds = 0.0;
    double fps = 0.0;
    double remainingSeconds = -1.0;
    std::string source;
    std::string pipeline;
    std::array<double, 4> displayTransform{1.0, 0.0, 0.0, 1.0};
};

struct VideoDetection {
    std::uint64_t frame = 0;
    double seconds = -1.0;
    float correlation = 0.0f;
};

class VideoCancelled : public std::runtime_error {
  public:
    VideoCancelled() : std::runtime_error("Video processing cancelled") {}
};

// callbacks run on processing threads, callers own the observer until the session stops
class VideoObserver {
  public:
    virtual ~VideoObserver() = default;
    virtual void onProgress(const VideoProgress&) {}
    virtual void onDetection(const VideoDetection&) {}
    virtual void onLog(const std::string&) {}
    virtual bool wantsPreview() const { return false; }
    virtual void previewSize(int& width, int& height) const {
        width = 960;
        height = 540;
    }
    // RGBA pixels are valid only during this callback
    virtual void onPreview(const std::uint8_t*, int, int, int) {}
};
} // namespace WatermarkCore
