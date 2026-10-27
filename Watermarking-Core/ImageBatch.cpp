#include "BatchUtils.hpp"
#include "include/ImageBatch.hpp"
#include "OutputFile.hpp"
#include "utils.hpp"
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <future>
#include <latch>
#include <map>
#include <mutex>
#include <omp.h>
#include <optional>
#include <semaphore>
#include <set>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#if defined(_USE_OPENCL_)
#include "OclQueueManager.hpp"
#elif defined(_USE_CUDA_)
#include "CudaCheck.hpp"
#include "CudaStreamManager.hpp"
#include <cuda_runtime.h>
#endif

/*!
 * \brief Shared image queue, device-local pipelines and publication of completed image files
 * \author Dimitris Karatzas
 */
namespace WatermarkCore {
namespace {
namespace fs = std::filesystem;
using namespace BatchUtils;

bool terminal(const ImageBatchState state) { return state == ImageBatchState::Completed || state == ImageBatchState::Failed || state == ImageBatchState::Cancelled; }

class Queue {
  public:
    Queue(const std::vector<ImageBatchJob>& jobs, const std::vector<int>& devices, ImageBatchControl& control, ImageBatchObserver* observer)
        : control(control), observer(observer), rows(jobs.size()), owners(jobs.size(), -1), tried(jobs.size()), reported(jobs.size()) {
        std::set<std::uint64_t> ids;
        for (std::size_t i = 0; i < jobs.size(); ++i) {
            if (!jobs[i].id || !ids.insert(jobs[i].id).second)
                throw std::invalid_argument("Image batch job IDs must be unique and nonzero");
            rows[i].id = jobs[i].id;
            rows[i].revision = 1;
            pending.push_back(i);
        }
        for (const int device : devices)
            alive.emplace(device, true);
    }

    std::optional<std::size_t> claim(const int device) {
        std::optional<std::size_t> id;
        {
            const std::lock_guard lock(mutex);
            if (control.cancelled() || !alive.at(device))
                return {};
            const auto found = std::ranges::find_if(pending, [&](const auto index) { return !tried[index].contains(device); });
            if (found == pending.end())
                return {};
            id = *found;
            pending.erase(found);
            owners[*id] = device;
            tried[*id].insert(device);
            ++active;
            auto& row = rows[*id];
            row.state = ImageBatchState::Loading;
            row.deviceIndex = device;
            row.detail.clear();
            ++row.revision;
        }
        notify(*id);
        return id;
    }

    bool wait(const int device) {
        std::unique_lock lock(mutex);
        changed.wait(lock, [&] { return control.cancelled() || !alive.at(device) || !active || std::ranges::any_of(pending, [&](const auto index) { return !tried[index].contains(device); }); });
        return !control.cancelled() && alive.at(device) && !pending.empty();
    }

    void update(const std::size_t index, const ImageBatchState state, const std::string& detail = {}, const std::optional<float> correlation = {}) {
        {
            const std::lock_guard lock(mutex);
            auto& row = rows[index];
            if (terminal(row.state))
                return;
            row.state = state;
            row.detail = detail;
            row.correlation = correlation;
            ++row.revision;
            if (terminal(state) && owners[index] >= 0) {
                owners[index] = -1;
                --active;
            }
        }
        changed.notify_all();
        notify(index);
    }

    void retire(const int device, const std::string& error) {
        std::vector<std::size_t> updates;
        {
            const std::lock_guard lock(mutex);
            alive.at(device) = false;
            for (std::size_t i = 0; i < rows.size(); ++i) {
                if (owners[i] != device || terminal(rows[i].state))
                    continue;
                owners[i] = -1;
                --active;
                auto& row = rows[i];
                row.detail = error;
                row.correlation.reset();
                if (control.cancelled())
                    row.state = ImageBatchState::Cancelled;
                else if (std::ranges::any_of(alive, [&](const auto& item) { return item.second && !tried[i].contains(item.first); })) {
                    row.state = ImageBatchState::Queued;
                    row.deviceIndex = -1;
                    pending.push_back(i);
                } else
                    row.state = ImageBatchState::Failed;
                ++row.revision;
                updates.push_back(i);
            }
            if (!error.empty())
                deviceErrors.push_back(error);
        }
        changed.notify_all();
        for (const auto i : updates)
            notify(i);
    }

    void settle() {
        for (std::size_t i = 0; i < rows.size(); ++i)
            update(i, control.cancelled() ? ImageBatchState::Cancelled : ImageBatchState::Failed,
                control.cancelled()    ? "Batch cancelled"
                : deviceErrors.empty() ? "No selected device could process this image"
                                       : deviceErrors.front());
    }

    std::vector<ImageBatchUpdate> snapshot() const {
        const std::lock_guard lock(mutex);
        return rows;
    }

    void wake() { changed.notify_all(); }

    void rethrowObserverError() {
        if (observerError)
            std::rethrow_exception(observerError);
    }

  private:
    void notify(const std::size_t index) {
        if (!observer)
            return;
        const std::lock_guard reportLock(reportMutex);
        if (observerError)
            return;
        ImageBatchUpdate value;
        {
            // Read the row after taking the callback lock to avoid sending an older state last
            const std::lock_guard lock(mutex);
            if (rows[index].revision <= reported[index])
                return;
            value = rows[index];
            reported[index] = value.revision;
        }
        try {
            observer->onUpdate(value);
        } catch (...) {
            observerError = std::current_exception();
            control.cancel();
            changed.notify_all();
        }
    }

    ImageBatchControl& control;
    ImageBatchObserver* observer;
    mutable std::mutex mutex;
    std::mutex reportMutex;
    std::condition_variable changed;
    std::deque<std::size_t> pending;
    std::vector<ImageBatchUpdate> rows;
    std::vector<int> owners;
    std::vector<std::set<int>> tried;
    std::vector<std::uint64_t> reported;
    std::map<int, bool> alive;
    std::vector<std::string> deviceErrors;
    std::size_t active = 0;
    std::exception_ptr observerError;
};

} // namespace

const char* imageBatchStateName(const ImageBatchState state) {
    switch (state) {
    case ImageBatchState::Queued: return "Queued";
    case ImageBatchState::Loading: return "Loading";
    case ImageBatchState::Processing: return "Processing";
    case ImageBatchState::Saving: return "Saving";
    case ImageBatchState::Completed: return "Done";
    case ImageBatchState::Failed: return "Failed";
    case ImageBatchState::Cancelled: return "Cancelled";
    }
    return "Unknown";
}

std::vector<ImageBatchJob> prepareImageBatch(const std::vector<std::string>& inputs, const std::string& outputDirectory, const bool detect, const std::string& suffix) {
    if (!detect && outputDirectory.empty())
        throw std::invalid_argument("Choose an image batch output folder");
    const auto folder = detect ? fs::path{} : normalized(fs::u8path(outputDirectory));
    std::set<fs::path, PathLess> reserved, seen;
    std::vector<fs::path> paths;
    for (const auto& input : inputs) {
        if (input.empty())
            throw std::invalid_argument("An image input path is empty");
        const auto path = normalized(fs::u8path(input));
        reserved.insert(path);
        if (seen.insert(path).second)
            paths.push_back(path);
    }
    std::vector<ImageBatchJob> jobs;
    for (const auto& input : paths) {
        std::string output;
        if (!detect) {
            const auto stem = input.stem().native() + fs::u8path(suffix).native();
            auto candidate = folder / (stem + input.extension().native());
            for (std::size_t number = 2; reserved.contains(candidate) || fs::exists(candidate); ++number)
                candidate = folder / (stem + L"_" + std::to_wstring(number) + input.extension().native());
            reserved.insert(candidate);
            output = utf8(candidate);
        }
        jobs.push_back({jobs.size() + 1, utf8(input), std::move(output)});
    }
    return jobs;
}
ImageBatchSummary runImageBatch(const ImageBatchSettings& settings, const std::vector<ImageBatchJob>& jobs, ImageBatchControl& control, ImageBatchObserver* observer) {
    const auto started = std::chrono::steady_clock::now();
    if (settings.password.empty())
        throw std::invalid_argument("Enter a watermark password");
    auto checkParameters = createImageSession(settings.password, settings.p, settings.psnr);
    checkParameters.reset();
    const auto selected = settings.deviceIndices.empty() ? std::vector<int>{settings.deviceIndex} : settings.deviceIndices;
    const auto names = getAvailableDevices();
    std::set<int> unique;
    for (const int device : selected) {
        if (device < 0 || device >= static_cast<int>(names.size()))
            throw std::invalid_argument("The selected image batch device is unavailable");
        if (!unique.insert(device).second)
            throw std::invalid_argument("Each image batch device must be selected once");
    }
#if defined(_USE_EIGEN_)
    if (selected.size() != 1)
        throw std::invalid_argument("Eigen image batches use one CPU worker");
#endif
    std::set<fs::path, PathLess> inputs, outputs;
    for (const auto& job : jobs) {
        if (job.inputPath.empty())
            throw std::invalid_argument("An image input path is empty");
        inputs.insert(normalized(fs::u8path(job.inputPath)));
    }
    for (const auto& job : jobs) {
        if (!settings.detect) {
            if (job.outputPath.empty())
                throw std::invalid_argument("An image output path is empty");
            const auto path = normalized(fs::u8path(job.outputPath));
            if (inputs.contains(path) || !outputs.insert(path).second)
                throw std::invalid_argument("Image batch outputs must be distinct from every input and other output");
        }
    }
    Queue queue(jobs, selected, control, observer);
    std::stop_callback wake(control.token(), [&] { queue.wake(); });
    struct Device {
        int index;
#if defined(_USE_OPENCL_)
        cl_device_id identity;
#elif defined(_USE_CUDA_)
        cudaUUID_t identity;
#endif
    };
    std::vector<Device> devices;
    if (!control.cancelled() && !jobs.empty()) {
        for (const int index : selected) {
#if defined(_USE_OPENCL_)
            devices.push_back({index, OclQueueManager::deviceHandle(index)});
#elif defined(_USE_CUDA_)
            cudaDeviceProp properties{};
            CUDA_CHECK(cudaGetDeviceProperties(&properties, index));
            devices.push_back({index, properties.uuid});
#else
            devices.push_back({index});
#endif
        }
    }
    const auto workerCount = std::max<std::size_t>(1, devices.size());
    const auto loadBudget = std::max<std::size_t>(InternalUtils::batchLoadThreads(), workerCount);
    const auto saveBudget = std::max<std::size_t>(omp_get_max_threads(), workerCount);
    const auto depth = std::max<std::size_t>(1, loadBudget / workerCount);
    const auto saveSlots = std::max<std::size_t>(1, std::min(jobs.size(), saveBudget / workerCount));
    std::counting_semaphore<> loadSlots(static_cast<std::ptrdiff_t>(loadBudget + 1));
    std::latch initialized(static_cast<std::ptrdiff_t>(devices.size()));
    std::vector<std::jthread> workers;
    try {
        for (const auto device : devices) {
            workers.emplace_back([&, device] {
                struct Arrival {
                    std::latch& initialized;
                    bool arrived = false;
                    void wait() {
                        arrived = true;
                        initialized.count_down();
                        initialized.wait();
                    }
                    ~Arrival() {
                        if (!arrived)
                            initialized.count_down();
                    }
                } arrival{initialized};
                std::string retirement;
                try {
                    if (control.cancelled()) {
                        queue.retire(device.index, {});
                        return;
                    }
                    std::size_t saveMemory = 0;
#if defined(_USE_OPENCL_)
                    OclQueueManager::initialize(device.index, device.identity);
                    const auto owner = OclQueueManager::current();
                    if (workerCount > 1)
                        saveMemory = std::max<std::size_t>(1, std::min<std::size_t>(512ull << 20, owner->getDevice().getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() / 8));
#elif defined(_USE_CUDA_)
                    cudaDeviceProp properties{};
                    CUDA_CHECK(cudaGetDeviceProperties(&properties, device.index));
                    if (std::memcmp(properties.uuid.bytes, device.identity.bytes, sizeof(properties.uuid.bytes)))
                        throw std::runtime_error("The selected CUDA adapter changed before image batch initialization");
                    CUDA_CHECK(cudaSetDevice(device.index));
                    CudaStreamManager::getInstance().getComputeStream();
                    if (workerCount > 1) {
                        std::size_t free = 0, total = 0;
                        CUDA_CHECK(cudaMemGetInfo(&free, &total));
                        saveMemory = std::max<std::size_t>(1, std::min<std::size_t>(512ull << 20, free / 8));
                    }
#else
                    initializeEnvironment(0);
#endif
                    auto session = createImageSession(settings.password, settings.p, settings.psnr);
                    // The saver must join before staged files are destroyed during error cleanup
                    std::map<std::size_t, std::unique_ptr<OutputFile>> staged;
                    std::optional<ImageSaver> saver;
                    if (!settings.detect)
                        saver.emplace(saveSlots, saveMemory);
                    // A fast GPU must not consume a short batch while another selected GPU is still starting
                    arrival.wait();
                    const auto reportSaves = [&](const auto& results) {
                        for (const auto& result : results) {
                            auto found = staged.find(result.id);
                            if (found == staged.end())
                                throw std::logic_error("An image save has no owning batch job");
                            try {
                                if (!result.error.empty())
                                    throw std::runtime_error(result.error);
                                found->second->commit(false);
                                queue.update(result.id, ImageBatchState::Completed, jobs[result.id].outputPath);
                            } catch (const std::exception& error) { queue.update(result.id, ImageBatchState::Failed, error.what()); }
                            staged.erase(found);
                        }
                    };
                    struct Pending {
                        std::size_t id;
                        std::future<PreloadedHandle> loaded;
                    };
                    std::deque<Pending> pending;
                    const auto load = [&, device
#if defined(_USE_OPENCL_)
                                          ,
                                          owner
#endif
                    ](const std::size_t id) {
                        loadSlots.acquire();
                        struct Release {
                            decltype(loadSlots)& slots;
                            int threads = omp_get_max_threads();
                            ~Release() {
                                omp_set_num_threads(threads);
                                slots.release();
                            }
                        } release{loadSlots};
                        // Parallel loads already share the CPU budget, avoid nested OpenMP teams
                        omp_set_num_threads(1);
                        if (control.cancelled())
                            return PreloadedHandle{};
#if defined(_USE_OPENCL_)
                        const OclQueueManager::Binding binding(owner);
#elif defined(_USE_CUDA_)
                        CUDA_CHECK(cudaSetDevice(device.index));
#endif
                        return preloadImageFromDisk(jobs[id].inputPath, -1, false);
                    };
                    const auto launch = [&] {
                        const auto id = queue.claim(device.index);
                        if (!id)
                            return false;
                        try {
                            pending.push_back({*id, std::async(std::launch::async, load, *id)});
                        } catch (...) {
                            queue.update(*id, ImageBatchState::Failed, "Could not start image loading");
                            throw;
                        }
                        return true;
                    };
                    for (;;) {
                        while (pending.size() < depth && launch()) {}
                        if (pending.empty()) {
                            if (saver)
                                reportSaves(saver->finish());
                            if (!queue.wait(device.index))
                                break;
                            continue;
                        }
                        if (control.cancelled())
                            break;
                        auto next = std::move(pending.front());
                        pending.pop_front();
                        launch();
                        try {
                            auto image = next.loaded.get();
                            if (!image || control.cancelled()) {
                                queue.update(next.id, ImageBatchState::Cancelled, "Batch cancelled");
                                continue;
                            }
                            bindPreloadedImage(session.get(), std::move(image));
                            queue.update(next.id, ImageBatchState::Processing);
                            if (settings.detect) {
                                const auto correlation = detectLoadedImage(session.get());
                                queue.update(next.id, ImageBatchState::Completed, {}, correlation);
                            } else {
                                auto output = std::make_unique<OutputFile>();
                                fs::create_directories(fs::absolute(fs::u8path(jobs[next.id].outputPath)).parent_path());
                                output->prepare(jobs[next.id].outputPath, true);
                                embedImage(session.get());
                                const auto path = output->path();
                                staged.emplace(next.id, std::move(output));
                                queue.update(next.id, ImageBatchState::Saving);
                                try {
                                    saver->saveExact(session.get(), path, next.id);
                                } catch (...) {
                                    staged.erase(next.id);
                                    throw;
                                }
                                reportSaves(saver->takeFinished());
                            }
                        } catch (const std::exception& error) {
                            if (lostDevice(error)) {
                                retirement = error.what();
                                break;
                            }
                            queue.update(next.id, ImageBatchState::Failed, error.what());
                        }
                    }
                    // A retired worker releases all loads and saves before another GPU can retry its unfinished jobs
                    pending.clear();
                    if (saver)
                        reportSaves(saver->finish());
                } catch (const std::exception& error) { retirement = error.what(); } catch (...) {
                    retirement = "Image batch worker failed";
                }
                queue.retire(device.index, retirement);
            });
        }
    } catch (...) {
        control.cancel();
        initialized.count_down(static_cast<std::ptrdiff_t>(devices.size() - workers.size()));
        workers.clear();
        throw;
    }
    workers.clear();
    queue.settle();
    queue.rethrowObserverError();
    ImageBatchSummary result;
    result.results = queue.snapshot();
    for (const auto& row : result.results) {
        result.succeeded += row.state == ImageBatchState::Completed;
        result.failed += row.state == ImageBatchState::Failed;
        result.cancelled += row.state == ImageBatchState::Cancelled;
    }
    result.elapsedSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    return result;
}
} // namespace WatermarkCore
