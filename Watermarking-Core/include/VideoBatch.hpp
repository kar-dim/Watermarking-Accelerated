#pragma once

#include "WatermarkCore.hpp"
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <stop_token>
#include <string>
#include <vector>

/*!
 * \brief Video batches on selected devices with progress, cancellation and safe outputs
 * \author Dimitris Karatzas
 */

namespace WatermarkCore {
enum class VideoBatchState { Pending, Probing, Processing, Completed, Failed, Skipped, Cancelled, Retrying };

struct VideoBatchJob {
    std::uint64_t id = 0;
    std::string inputPath;
    std::string outputPath;
};

struct VideoBatchSettings {
    VideoSettings video{};
    int deviceIndex = 0;
    bool detect = false;
    std::vector<int> deviceIndices;
    std::map<int, std::string> hardwareOptions;
    bool accelerateFinal = true;
};

struct VideoBatchUpdate {
    std::uint64_t id = 0;
    VideoBatchState state = VideoBatchState::Pending;
    VideoProgress progress;
    std::uint64_t detections = 0;
    double meanCorrelation = 0.0;
    std::string detail;
    std::uint64_t attempt = 0;
    std::uint64_t revision = 0;
    int deviceIndex = -1;
    std::vector<int> activeDevices;
};

struct VideoBatchSummary {
    std::size_t succeeded = 0;
    std::size_t failed = 0;
    std::size_t skipped = 0;
    std::size_t cancelled = 0;
    double elapsedSeconds = 0.0;
    std::vector<VideoBatchUpdate> results;
};

// Callbacks are serialized across devices and may run on different worker threads
// callback error cancels the batch and returns to the caller after workers stop
class VideoBatchObserver {
  public:
    virtual ~VideoBatchObserver() = default;
    virtual void onUpdate(const VideoBatchUpdate&) {}
    virtual void onLog(std::uint64_t, const std::string&) {}
};

class VideoBatchScheduler;
class VideoBatchControl {
  public:
    void cancel() { stop_.request_stop(); }
    bool cancelled() const { return stop_.stop_requested(); }
    bool skipCurrent(std::uint64_t expectedId = 0);

  private:
    friend VideoBatchSummary runVideoBatch(const VideoBatchSettings&, const std::vector<VideoBatchJob>&, VideoBatchControl&, VideoBatchObserver*);
    std::mutex mutex_;
    std::stop_source stop_;
    std::shared_ptr<VideoBatchScheduler> scheduler_;
};

const char* videoBatchStateName(VideoBatchState state);
std::vector<std::filesystem::path> getValidVideoFiles(const std::filesystem::path& directory);
std::vector<VideoBatchJob> prepareVideoBatch(const std::vector<std::string>& inputs, const std::string& outputDirectory, bool detect, const std::string& extension = "mkv");
VideoBatchSummary runVideoBatch(const VideoBatchSettings& settings, const std::vector<VideoBatchJob>& jobs, VideoBatchControl& control, VideoBatchObserver* observer = nullptr);
} // namespace WatermarkCore
