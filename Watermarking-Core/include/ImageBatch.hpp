#pragma once

#include "WatermarkCore.hpp"
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stop_token>
#include <string>
#include <vector>

/*!
 * \brief Whole-image batch processing on selected devices with bounded prefetching and saving
 * \author Dimitris Karatzas
 */
namespace WatermarkCore {
enum class ImageBatchState { Queued, Loading, Processing, Saving, Completed, Failed, Cancelled };

struct ImageBatchJob {
    std::uint64_t id = 0;
    std::string inputPath;
    std::string outputPath;
};

struct ImageBatchSettings {
    std::string password;
    int p = 3;
    float psnr = 40.0f;
    int deviceIndex = 0;
    std::vector<int> deviceIndices;
    bool detect = false;
};

struct ImageBatchUpdate {
    std::uint64_t id = 0;
    std::uint64_t revision = 0;
    ImageBatchState state = ImageBatchState::Queued;
    int deviceIndex = -1;
    std::optional<float> correlation;
    std::string detail;
};

struct ImageBatchSummary {
    std::size_t succeeded = 0, failed = 0, cancelled = 0;
    double elapsedSeconds = 0;
    std::vector<ImageBatchUpdate> results;
};

// Callbacks are serialized across devices and may run on different worker threads
// callback error cancels the batch and returns to the caller after workers stop
class ImageBatchObserver {
  public:
    virtual ~ImageBatchObserver() = default;
    virtual void onUpdate(const ImageBatchUpdate&) {}
};

class ImageBatchControl {
  public:
    void cancel() { stop_.request_stop(); }
    bool cancelled() const { return stop_.stop_requested(); }
    std::stop_token token() const { return stop_.get_token(); }

  private:
    std::stop_source stop_;
};

const char* imageBatchStateName(ImageBatchState state);
std::vector<ImageBatchJob> prepareImageBatch(const std::vector<std::string>& inputs, const std::string& outputDirectory, bool detect, const std::string& suffix = {});
ImageBatchSummary runImageBatch(const ImageBatchSettings& settings, const std::vector<ImageBatchJob>& jobs, ImageBatchControl& control, ImageBatchObserver* observer = nullptr);
} // namespace WatermarkCore
