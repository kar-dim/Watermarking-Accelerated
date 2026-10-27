#pragma once
#include "OclMemPool.hpp"
#include "OclVideoInterop.hpp"
#include "opencl_init.h"
#include <cstdint>
#include <atomic>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <unordered_map>
#include <vector>

/*!
 *  \brief Persistent OpenCL context, queue, pool and video interop for each device
 *  \author Dimitris Karatzas
 */
class OclQueueManager {
    cl::Platform platform;
    cl::Device device;
    std::shared_ptr<OclVideoInterop> videoInterop;
    cl::Context ctx;
    cl::CommandQueue queue;
    OclMemPool pool;
    int deviceIndex = 0;
    uint64_t contextGeneration = 0;
    size_t liveAllocations = 0;
    std::recursive_mutex stateMutex;

    OclQueueManager() = default;

    static inline std::mutex registryMutex;
    static inline std::unordered_map<cl_device_id, std::shared_ptr<OclQueueManager>> devices;
    static inline std::atomic<uint64_t> nextGeneration{0};
    // Each worker selects its own device without changing another worker's context
    static inline thread_local std::shared_ptr<OclQueueManager> selected;

    static std::vector<std::pair<cl::Platform, cl::Device>> enumerateGpuDevices() {
        std::vector<std::pair<cl::Platform, cl::Device>> gpus;
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        for (auto& plat : platforms) {
            std::vector<cl::Device> devices;
            plat.getDevices(CL_DEVICE_TYPE_GPU, &devices);
            for (auto& dev : devices)
                gpus.emplace_back(plat, dev);
        }
        return gpus;
    }

  public:
    static void initialize(int deviceIndex = 0, cl_device_id expected = nullptr) {
        const auto gpus = enumerateGpuDevices();
        if (gpus.empty())
            throw std::runtime_error("No OpenCL GPU devices found.");
        if (deviceIndex < 0 || deviceIndex >= static_cast<int>(gpus.size()))
            throw std::runtime_error("OpenCL device index out of range: " + std::to_string(deviceIndex));
        // Adapter indices can change between selection and worker startup
        if (expected && gpus[deviceIndex].second() != expected)
            throw std::runtime_error("The selected OpenCL adapter changed before initialization");
        std::lock_guard lock(registryMutex);
        const auto found = devices.find(gpus[deviceIndex].second());
        if (found != devices.end()) {
            selected = found->second;
            return;
        }
        auto next = std::shared_ptr<OclQueueManager>(new OclQueueManager());
        auto& mgr = *next;
        auto nextInterop = OclVideoInterop::create(gpus[deviceIndex].first(), gpus[deviceIndex].second());
        cl::Context nextContext = nextInterop ? cl::Context(nextInterop->context(), true) : cl::Context(gpus[deviceIndex].second);
        cl::CommandQueue nextQueue(nextContext, gpus[deviceIndex].second, 0);
        mgr.contextGeneration = ++nextGeneration;
        mgr.deviceIndex = deviceIndex;
        mgr.platform = gpus[deviceIndex].first;
        mgr.device = gpus[deviceIndex].second;
        mgr.ctx = std::move(nextContext);
        mgr.queue = std::move(nextQueue);
        mgr.videoInterop = std::move(nextInterop);
        mgr.pool.setCapacity(mgr.device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>());
        devices.emplace(mgr.device(), next);
        selected = std::move(next);
    }

    static OclQueueManager& getInstance() {
        if (!selected)
            throw std::runtime_error("OclQueueManager not initialized, call initialize() first.");
        return *selected;
    }

    static std::shared_ptr<OclQueueManager> current() { return selected; }
    static void bind(std::shared_ptr<OclQueueManager> owner) { selected = std::move(owner); }
    class Binding {
      public:
        explicit Binding(std::shared_ptr<OclQueueManager> owner) : previous(current()) {
            if (owner)
                bind(std::move(owner));
        }
        ~Binding() { bind(std::move(previous)); }
        Binding(const Binding&) = delete;
        Binding& operator=(const Binding&) = delete;

      private:
        std::shared_ptr<OclQueueManager> previous;
    };
    static std::shared_ptr<OclQueueManager> forQueue(const cl_command_queue queue) {
        if (selected && selected->queue() == queue)
            return selected;
        std::lock_guard lock(registryMutex);
        for (const auto& [device, owner] : devices)
            if (owner->queue() == queue)
                return owner;
        throw std::invalid_argument("The OpenCL queue has no owning device");
    }
    static cl_device_id deviceHandle(const int index) {
        const auto gpus = enumerateGpuDevices();
        if (index < 0 || index >= static_cast<int>(gpus.size()))
            throw std::out_of_range("OpenCL device index out of range");
        return gpus[index].second();
    }

    cl_command_queue getQueueRaw() const { return queue.get(); }
    cl_context getContextRaw() const { return ctx.get(); }
    cl::CommandQueue& getQueue() { return queue; }
    const cl::CommandQueue& getQueue() const { return queue; }
    cl::Context& getContext() { return ctx; }
    const cl::Context& getContext() const { return ctx; }
    cl::Device& getDevice() { return device; }
    const cl::Device& getDevice() const { return device; }
    int getDeviceIndex() const { return deviceIndex; }
    uint64_t getContextGeneration() const { return contextGeneration; }
    OclMemPool& getPool() { return pool; }
    const std::shared_ptr<OclVideoInterop>& getVideoInterop() const { return videoInterop; }
    // program compilation reads context and device properties, hold one selection
    std::unique_lock<std::recursive_mutex> lockContext() { return std::unique_lock(stateMutex); }

    cl_mem acquireBuffer(const size_t bytes, const cl_command_queue ownerQueue) {
        std::lock_guard lock(stateMutex);
        if (ownerQueue != queue.get())
            throw std::invalid_argument("OpenCL buffer queue does not belong to the selected context");
        cl_mem buffer = pool.acquire(bytes, ctx.get());
        ++liveAllocations;
        return buffer;
    }

    void releaseBuffer(const size_t bytes, cl_mem buffer) noexcept {
        std::lock_guard lock(stateMutex);
        pool.release(bytes, buffer);
        --liveAllocations;
    }

    static std::vector<std::string> enumerateDevices() {
        std::vector<std::string> names;
        for (auto& [plat, dev] : enumerateGpuDevices())
            names.push_back(dev.getInfo<CL_DEVICE_NAME>());
        return names;
    }

    static cl_uint deviceVendorId(const int index) {
        const auto devices = enumerateGpuDevices();
        if (index < 0 || index >= static_cast<int>(devices.size()))
            throw std::out_of_range("OpenCL device index out of range: " + std::to_string(index));
        return devices[index].second.getInfo<CL_DEVICE_VENDOR_ID>();
    }

    static void finish() {
        if (selected)
            selected->queue.finish();
    }
};
