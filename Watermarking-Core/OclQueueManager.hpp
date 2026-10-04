#pragma once
#include "OclMemPool.hpp"
#include "opencl_init.h"
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

/*!
 *  \brief  Singleton that owns the OpenCL context, (in-order) command queue, device selection,
 *          and the shared memory pool. Cleanup (finish queue, drain pool, bump generation)
 *          happens automatically when switching devices.
 *  \author Dimitris Karatzas
 */
class OclQueueManager {
    cl::Platform platform;
    cl::Device device;
    cl::Context ctx;
    cl::CommandQueue queue;
    OclMemPool pool;
    int deviceIndex = 0;
    uint32_t contextGeneration = 0;
    size_t liveAllocations = 0;
    std::recursive_mutex stateMutex;

    OclQueueManager() = default;

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
    static void initialize(int deviceIndex = 0) {
        auto& mgr = instance();
        std::lock_guard lock(mgr.stateMutex);
        if (mgr.queue.get() && mgr.deviceIndex == deviceIndex)
            return;
        const auto gpus = enumerateGpuDevices();
        if (gpus.empty())
            throw std::runtime_error("No OpenCL GPU devices found.");
        if (deviceIndex < 0 || deviceIndex >= static_cast<int>(gpus.size()))
            throw std::runtime_error("OpenCL device index out of range: " + std::to_string(deviceIndex));
        if (mgr.liveAllocations != 0)
            throw std::runtime_error("Release all OpenCL image, session and export buffers before switching devices");
        // Build the replacement first, initialization failure preserves the old context
        cl::Context nextContext(gpus[deviceIndex].second);
        cl::CommandQueue nextQueue(nextContext, gpus[deviceIndex].second, 0);
        if (mgr.queue.get())
            mgr.queue.finish();
        mgr.pool.reset();
        mgr.contextGeneration++;
        mgr.deviceIndex = deviceIndex;
        mgr.platform = gpus[deviceIndex].first;
        mgr.device = gpus[deviceIndex].second;
        mgr.ctx = std::move(nextContext);
        mgr.queue = std::move(nextQueue);
        mgr.pool.setCapacity(mgr.device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>());
    }

    static OclQueueManager& getInstance() {
        auto& mgr = instance();
        std::lock_guard lock(mgr.stateMutex);
        if (!mgr.queue.get())
            throw std::runtime_error("OclQueueManager not initialized, call initialize() first.");
        return mgr;
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
    uint32_t getContextGeneration() const { return contextGeneration; }
    OclMemPool& getPool() { return pool; }
    // Program compilation reads several context/device properties, we hold one selection
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

    static void finish() {
        if (instance().queue.get())
            instance().queue.finish();
    }

  private:
    static OclQueueManager& instance() {
        static OclQueueManager mgr;
        return mgr;
    }
};
