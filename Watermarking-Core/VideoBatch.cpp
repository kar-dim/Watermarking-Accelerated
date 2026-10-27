#include "include/VideoBatch.hpp"
#include "AvUtil.hpp"
#include "BatchUtils.hpp"
#include "OutputFile.hpp"
#include "VideoBatchScheduler.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <locale>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <utility>
#include <thread>
#include <windows.h>

#if defined(_USE_OPENCL_)
#include "OclQueueManager.hpp"
#elif defined(_USE_CUDA_)
#include "CudaCheck.hpp"
#include "CudaStreamManager.hpp"
#include <cuda_runtime.h>
#endif

/*!
 * \brief Selected-device video pipelines and per-file detection reports
 * \author Dimitris Karatzas
 */

namespace WatermarkCore {
namespace {
namespace fs = std::filesystem;
using namespace BatchUtils;

class JobObserver : public VideoObserver {
  public:
    JobObserver(const VideoBatchJob& job, std::function<void(const VideoBatchUpdate&)> progress, std::function<void(const std::string&)> log, std::ostream* csv)
        : progress_(std::move(progress)), log_(std::move(log)), csv_(csv) {
        update.id = job.id;
        update.state = VideoBatchState::Probing;
    }
    void notify() const { progress_(update); }
    void onProgress(const VideoProgress& progress) override {
        update.progress = progress;
        if (progress.state == VideoState::Processing || progress.state == VideoState::Finalizing)
            update.state = VideoBatchState::Processing;
        // Keep file progress below 100% until the scheduler publishes a winning output
        if (progress.state == VideoState::Completed)
            update.progress.fraction = .999;
        notify();
    }
    void onDetection(const VideoDetection& result) override {
        *csv_ << result.frame << ',';
        if (result.seconds >= 0.0)
            *csv_ << std::fixed << std::setprecision(6) << result.seconds;
        *csv_ << ',' << std::defaultfloat << std::setprecision(std::numeric_limits<float>::max_digits10) << result.correlation << '\n';
        if (!*csv_)
            throw std::runtime_error("Could not write the batch detection report");
        correlationSum_ += result.correlation;
        ++update.detections;
        update.meanCorrelation = correlationSum_ / update.detections;
    }
    void onLog(const std::string& text) override { log_(text); }
    bool wantsPreview() const override { return false; }
    VideoBatchUpdate update;

  private:
    std::function<void(const VideoBatchUpdate&)> progress_;
    std::function<void(const std::string&)> log_;
    std::ostream* csv_;
    double correlationSum_ = 0.0;
};
} // namespace

const char* videoBatchStateName(const VideoBatchState state) {
    switch (state) {
    case VideoBatchState::Pending: return "Pending";
    case VideoBatchState::Probing: return "Opening";
    case VideoBatchState::Processing: return "Processing";
    case VideoBatchState::Completed: return "Done";
    case VideoBatchState::Failed: return "Failed";
    case VideoBatchState::Skipped: return "Skipped";
    case VideoBatchState::Cancelled: return "Cancelled";
    case VideoBatchState::Retrying: return "Retrying";
    }
    return "Unknown";
}

bool VideoBatchControl::skipCurrent(const std::uint64_t expectedId) {
    std::shared_ptr<VideoBatchScheduler> scheduler;
    {
        const std::lock_guard lock(mutex_);
        scheduler = scheduler_;
    }
    return scheduler && scheduler->skip(expectedId);
}

std::vector<fs::path> getValidVideoFiles(const fs::path& directory) {
    constexpr std::wstring_view extensions[]{L".mkv", L".mp4", L".mov", L".avi", L".webm", L".m4v", L".ts", L".mts", L".m2ts", L".mpg", L".mpeg", L".wmv", L".vob", L".mxf", L".flv", L".ogv", L".nut",
        L".y4m", L".h264", L".hevc", L".av1"};
    if (!fs::is_directory(directory))
        throw std::invalid_argument("The video batch input must be a folder");
    std::vector<fs::path> files;
    for (const auto& entry : fs::directory_iterator(directory)) {
        if (!entry.is_regular_file())
            continue;
        const std::wstring extension = entry.path().extension().native();
        if (std::ranges::any_of(extensions, [&](const auto candidate) {
                return CompareStringOrdinal(extension.c_str(), static_cast<int>(extension.size()), candidate.data(), static_cast<int>(candidate.size()), TRUE) == CSTR_EQUAL;
            }))
            files.push_back(entry.path());
    }
    std::sort(files.begin(), files.end(), PathLess{});
    return files;
}

std::vector<VideoBatchJob> prepareVideoBatch(const std::vector<std::string>& inputs, const std::string& outputDirectory, const bool detect, const std::string& extension) {
    if (inputs.empty())
        throw std::invalid_argument("Choose at least one input video");
    if (outputDirectory.empty())
        throw std::invalid_argument("Choose a batch output folder");
    std::string suffix = extension;
    if (!suffix.empty() && suffix.front() == '.')
        suffix.erase(0, 1);
    if (!detect)
        validateVideoOutputPath("output." + suffix);
    const fs::path output = normalized(video_utils::pathFromUtf8(outputDirectory));
    fs::create_directories(output);
    if (!fs::is_directory(output))
        throw std::invalid_argument("The batch output path must be a folder");
    std::vector<fs::path> uniqueInputs;
    std::set<fs::path, PathLess> reserved;
    for (const auto& input : inputs) {
        if (input.empty())
            throw std::invalid_argument("An input video path is empty");
        const fs::path path = fs::absolute(video_utils::pathFromUtf8(input)).lexically_normal();
        if (reserved.insert(normalized(path)).second)
            uniqueInputs.push_back(path);
    }
    std::map<fs::path, std::uint64_t, PathLess> sequences;
    std::vector<VideoBatchJob> jobs;
    for (const auto& input : uniqueInputs) {
        const fs::path base = output / (input.stem().native() + (detect ? L"_detections" : L"_watermarked"));
        auto& sequence = sequences[base];
        for (;;) {
            if (sequence == std::numeric_limits<std::uint64_t>::max())
                throw std::runtime_error("Could not reserve another batch output name");
            ++sequence;
            const fs::path candidate = base.native() + (sequence == 1 ? L"" : L"_" + std::to_wstring(sequence)) + (detect ? L".csv" : L"." + std::wstring(suffix.begin(), suffix.end()));
            std::error_code error;
            const fs::file_status status = fs::symlink_status(candidate, error);
            if (error && error != std::errc::no_such_file_or_directory)
                throw std::system_error(error, "Could not inspect a batch output path");
            if (fs::exists(status) || reserved.contains(candidate))
                continue;
            reserved.insert(candidate);
            jobs.push_back({jobs.size() + 1, utf8(input), utf8(candidate)});
            break;
        }
    }
    return jobs;
}

VideoBatchSummary runVideoBatch(const VideoBatchSettings& settings, const std::vector<VideoBatchJob>& jobs, VideoBatchControl& control, VideoBatchObserver* observer) {
    if (jobs.empty())
        throw std::invalid_argument("The video batch is empty");
    if (settings.video.watermarkPassword.empty())
        throw std::invalid_argument("Enter a watermark password");
    if (settings.video.p != 3 && settings.video.p != 5 && settings.video.p != 7 && settings.video.p != 9)
        throw std::invalid_argument("The prediction window must be 3, 5, 7 or 9");
    if (!std::isfinite(settings.video.psnr) || settings.video.psnr <= 0 || settings.video.watermarkInterval < 1 || settings.video.cpuThreads < 0)
        throw std::invalid_argument("PSNR and the watermark interval must be positive, CPU threads cannot be negative");
    if (!settings.detect) {
        validateVideoEncoderOptions(settings.video.encodeOptions, settings.video.useHwEncoder);
        for (const auto& [device, options] : settings.hardwareOptions)
            validateVideoEncoderOptions(options, true);
    }
    std::vector<int> selected = settings.deviceIndices.empty() ? std::vector<int>{settings.deviceIndex} : settings.deviceIndices;
    std::set<std::uint64_t> ids;
    std::map<std::uint64_t, const VideoBatchJob*> byId;
    for (const auto& job : jobs) {
        if (!job.id || !ids.insert(job.id).second || job.inputPath.empty() || job.outputPath.empty())
            throw std::invalid_argument("The batch contains an invalid job");
        byId.emplace(job.id, &job);
    }
    std::set<fs::path, PathLess> inputs, outputs;
    for (const auto& job : jobs)
        inputs.insert(normalized(fs::u8path(job.inputPath)));
    for (const auto& job : jobs) {
        if (!settings.detect)
            validateVideoOutputPath(job.outputPath);
        const auto path = normalized(fs::u8path(job.outputPath));
        if (inputs.contains(path) || !outputs.insert(path).second)
            throw std::invalid_argument("Video batch outputs must be distinct from every input and other output");
    }
    const auto started = std::chrono::steady_clock::now();
    auto scheduler = std::make_shared<VideoBatchScheduler>(jobs, selected, settings.accelerateFinal);
    std::mutex reportMutex;
    std::exception_ptr observerError;
    std::map<std::uint64_t, std::uint64_t> reported;
    const auto report = [&] {
        if (!observer)
            return;
        // User callbacks share one lock and run outside the scheduler lock
        const std::lock_guard lock(reportMutex);
        if (observerError)
            return;
        for (const auto& view : scheduler->snapshot())
            if (reported[view.id] < view.revision) {
                reported[view.id] = view.revision;
                try {
                    observer->onUpdate(view);
                } catch (...) {
                    observerError = std::current_exception();
                    control.cancel();
                    return;
                }
            }
    };
    const auto log = [&](const std::uint64_t id, const int device, const std::string& text) {
        if (!observer)
            return;
        const std::lock_guard lock(reportMutex);
        if (observerError)
            return;
        try {
            observer->onLog(id, "[device " + std::to_string(device) + "] " + text);
        } catch (...) {
            observerError = std::current_exception();
            control.cancel();
        }
    };
    {
        const std::lock_guard lock(control.mutex_);
        if (control.scheduler_)
            throw std::logic_error("The batch controller is already in use");
        control.scheduler_ = scheduler;
    }
    struct Unbind {
        std::mutex& mutex;
        std::shared_ptr<VideoBatchScheduler>& bound;
        ~Unbind() {
            const std::lock_guard lock(mutex);
            bound.reset();
        }
    } unbind{control.mutex_, control.scheduler_};
    std::stop_callback cancel(control.stop_.get_token(), [&] { scheduler->cancel(); });

    struct Device {
        int index;
#if defined(_USE_OPENCL_)
        cl_device_id handle;
#elif defined(_USE_CUDA_)
        cudaUUID_t uuid;
#endif
    };
    std::vector<Device> devices;
    if (!control.cancelled()) {
        const auto names = getAvailableDevices();
#if defined(_USE_EIGEN_)
        if (selected.size() != 1)
            throw std::invalid_argument("Eigen video batches use one CPU worker");
#endif
        for (const int device : selected) {
            if (device < 0 || device >= static_cast<int>(names.size()))
                throw std::invalid_argument("The selected batch device is unavailable");
#if defined(_USE_OPENCL_)
            devices.push_back({device, OclQueueManager::deviceHandle(device)});
#elif defined(_USE_CUDA_)
            cudaDeviceProp properties;
            CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
            devices.push_back({device, properties.uuid});
#else
            devices.push_back({device});
#endif
        }
    }
    report();
    // Decode, filters and encode share CPU time across all selected GPU workers
    const int cpuThreads = settings.video.cpuThreads > 0 ? settings.video.cpuThreads
                           : devices.size() > 1          ? std::max(1, static_cast<int>(std::thread::hardware_concurrency()) / static_cast<int>(devices.size() * 3))
                                                         : 0;
    std::vector<std::jthread> workers;
    try {
        for (const Device device : devices) {
            workers.emplace_back([&, device] {
                try {
                    if (control.cancelled())
                        return;
#if defined(_USE_OPENCL_)
                    OclQueueManager::initialize(device.index, device.handle);
#elif defined(_USE_CUDA_)
                    cudaDeviceProp current;
                    CUDA_CHECK(cudaGetDeviceProperties(&current, device.index));
                    if (std::memcmp(current.uuid.bytes, device.uuid.bytes, sizeof(current.uuid.bytes)) != 0)
                        throw std::runtime_error("The selected CUDA adapter changed before initialization");
                    CUDA_CHECK(cudaSetDevice(device.index));
                    CudaStreamManager::getInstance().getComputeStream();
#else
                    initializeEnvironment(0);
                    if (!settings.detect)
                        optimizeThreadsForVideoEmbedding();
#endif
                    while (const auto claim = scheduler->claim(device.index)) {
                        const auto& job = *byId.at(claim->id);
                        // Each attempt owns a separate file, only the scheduler can publish the winner
                        OutputFile staged;
                        std::ofstream csv;
                        bool retryable = false;
                        JobObserver bridge(
                            job,
                            [&](const VideoBatchUpdate& update) {
                                scheduler->progress(*claim, update);
                                report();
                            },
                            [&](const std::string& text) { log(job.id, device.index, text); }, settings.detect ? &csv : nullptr);
                        try {
                            bridge.notify();
                            if (claim->stop.stop_requested())
                                throw VideoCancelled();
                            const auto source = probeVideoSource(job.inputPath, claim->stop);
                            staged.prepare(job.outputPath, true);
                            if (settings.detect) {
                                csv.open(video_utils::pathFromUtf8(staged.path()), std::ios::binary | std::ios::trunc);
                                csv.imbue(std::locale::classic());
                                if (!csv)
                                    throw std::runtime_error("Could not open the batch detection report");
                                csv << "frame,time_seconds,correlation\n";
                            }
                            VideoSettings video = settings.video;
                            video.videoFile = job.inputPath;
                            video.encodeOutputPath = staged.path();
                            video.observer = &bridge;
                            video.stopToken = claim->stop;
                            video.cpuThreads = cpuThreads;
                            if (video.useHwEncoder && settings.hardwareOptions.contains(device.index))
                                video.encodeOptions = settings.hardwareOptions.at(device.index);
                            bridge.update.detail = std::to_string(source.width) + " x " + std::to_string(source.height) + " " + source.codec;
                            retryable = true;
                            int frames = 0;
                            {
                                auto session = initVideo(video);
                                frames = settings.detect ? detectVideo(session.get()) : embedVideo(session.get());
                            }
                            retryable = false;
                            if (settings.detect) {
                                csv.flush();
                                csv.close();
                                if (!csv)
                                    throw std::runtime_error("Could not finish the batch detection report");
                            }
                            bridge.update.progress.frames = frames;
                            bridge.update.detail = job.outputPath;
                            scheduler->complete(*claim, bridge.update, [&] { staged.commit(false); });
                        } catch (const std::exception& error) {
                            log(job.id, device.index, error.what());
                            scheduler->fail(*claim, error.what(), retryable, lostDevice(error));
                        }
                        report();
                    }
                } catch (const std::exception& error) {
                    scheduler->retire(device.index, error.what());
                    log(0, device.index, error.what());
                    report();
                } catch (...) {
                    scheduler->retire(device.index, "Video batch worker failed");
                    report();
                }
            });
        }
    } catch (...) {
        scheduler->cancel();
        for (auto& worker : workers)
            worker.join();
        throw;
    }
    for (auto& worker : workers)
        worker.join();
    report();
    if (observerError)
        std::rethrow_exception(observerError);
    VideoBatchSummary summary;
    summary.results = scheduler->snapshot();
    for (const auto& result : summary.results) {
        switch (result.state) {
        case VideoBatchState::Completed: ++summary.succeeded; break;
        case VideoBatchState::Failed: ++summary.failed; break;
        case VideoBatchState::Skipped: ++summary.skipped; break;
        case VideoBatchState::Cancelled: ++summary.cancelled; break;
        default: break;
        }
    }
    summary.elapsedSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    return summary;
}
} // namespace WatermarkCore
