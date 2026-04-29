#pragma once
#include "OclMemPool.hpp"
#include "opencl_init.h"
#include <cstdint>
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
        if (mgr.queue.get() && mgr.deviceIndex == deviceIndex)
            return;

        // finish all work and release pooled buffers from the old context
        if (mgr.queue.get())
            mgr.queue.finish();
        mgr.pool.reset();
        mgr.contextGeneration++;

        const auto gpus = enumerateGpuDevices();
        if (gpus.empty())
            throw std::runtime_error("No OpenCL GPU devices found.");
        if (deviceIndex < 0 || deviceIndex >= static_cast<int>(gpus.size()))
            throw std::runtime_error("OpenCL device index out of range: " + std::to_string(deviceIndex));
        mgr.deviceIndex = deviceIndex;
        mgr.platform = gpus[deviceIndex].first;
        mgr.device = gpus[deviceIndex].second;
        mgr.ctx = cl::Context(mgr.device);
        mgr.queue = cl::CommandQueue(mgr.ctx, mgr.device, 0);
    }

    static OclQueueManager& getInstance() {
        auto& mgr = instance();
        if (!mgr.queue.get())
            throw std::runtime_error("OclQueueManager not initialized — call initialize() first.");
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
