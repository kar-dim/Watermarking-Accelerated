#pragma once
#include "CudaCheck.hpp"
#include "CudaMemPool.hpp"
#include <cuda_runtime.h>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

/*!
 *  \brief Owns one CUDA stream and memory pool per device, and the per device objects that are reused instead of created again for
 *         each watermark object (one per image size): pinned result floats, graph capture streams and instantiated graphs
 *  \author Dimitris Karatzas
 */
class CudaStreamManager {
  public:
    static CudaStreamManager& getInstance() {
        static CudaStreamManager instance;
        return instance;
    }

    cudaStream_t getComputeStream() { return currentResources().computeStream; }
    CudaMemPool& getPool() { return currentResources().pool; }

    // pinned host float that kernels write a result to, kept for reuse when released
    float* acquirePinnedFloat() {
        auto& resources = currentResources();
        if (float* value = resources.reused.takePinnedFloat())
            return value;
        float* value = nullptr;
        CUDA_CHECK(cudaHostAlloc(&value, sizeof(float), cudaHostAllocMapped));
        return value;
    }
    void releasePinnedFloat(const int device, float* value) noexcept {
        if (value && !reuse(device, [&](ReusedObjects& reused) { reused.pinnedFloats.push_back(value); }))
            cudaFreeHost(value);
    }

    // non-blocking stream to capture a CUDA graph on, held for one capture only
    cudaStream_t acquireCaptureStream() {
        auto& resources = currentResources();
        if (cudaStream_t stream = resources.reused.takeCaptureStream())
            return stream;
        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        return stream;
    }
    void releaseCaptureStream(const int device, cudaStream_t stream) noexcept {
        if (stream && !reuse(device, [&](ReusedObjects& reused) { reused.captureStreams.push_back(stream); }))
            cudaStreamDestroy(stream);
    }

    cudaGraphExec_t acquireGraph(const int kind) { return currentResources().reused.takeGraph(kind); }

    void releaseGraph(const int device, const int kind, cudaGraphExec_t graph) noexcept {
        if (graph && !reuse(device, [&](ReusedObjects& reused) { reused.graphs.emplace(kind, graph); }))
            cudaGraphExecDestroy(graph);
    }

    CudaStreamManager(const CudaStreamManager&) = delete;
    CudaStreamManager& operator=(const CudaStreamManager&) = delete;
    CudaStreamManager(CudaStreamManager&&) = delete;
    CudaStreamManager& operator=(CudaStreamManager&&) = delete;

  private:
    struct ReusedObjects {
        std::mutex mutex;
        std::vector<float*> pinnedFloats;
        std::vector<cudaStream_t> captureStreams;
        std::unordered_multimap<int, cudaGraphExec_t> graphs;

        template <typename T>
        T takeLast(std::vector<T>& objects) {
            std::lock_guard lock(mutex);
            if (objects.empty())
                return nullptr;
            T object = objects.back();
            objects.pop_back();
            return object;
        }
        float* takePinnedFloat() { return takeLast(pinnedFloats); }

        cudaStream_t takeCaptureStream() { return takeLast(captureStreams); }

        cudaGraphExec_t takeGraph(const int kind) {
            std::lock_guard lock(mutex);
            const auto it = graphs.find(kind);
            if (it == graphs.end())
                return nullptr;
            cudaGraphExec_t graph = it->second;
            graphs.erase(it);
            return graph;
        }

        ~ReusedObjects() {
            for (float* value : pinnedFloats)
                cudaFreeHost(value);
            for (cudaStream_t stream : captureStreams)
                cudaStreamDestroy(stream);
            for (auto& [kind, graph] : graphs)
                cudaGraphExecDestroy(graph);
        }
    };

    struct DeviceResources {
        cudaStream_t computeStream = nullptr;
        CudaMemPool pool;
        ReusedObjects reused;

        DeviceResources() {
            CUDA_CHECK(cudaStreamCreate(&computeStream));
            try {
                size_t freeMem = 0;
                size_t totalMem = 0;
                CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));
                pool.setCapacity(totalMem);
            } catch (...) {
                cudaStreamDestroy(computeStream);
                throw;
            }
        }

        ~DeviceResources() {
            pool.reset(computeStream);
            cudaStreamDestroy(computeStream);
        }
    };

    std::mutex mutex_;
    std::unordered_map<int, std::unique_ptr<DeviceResources>> devices_;

    CudaStreamManager() = default;
    ~CudaStreamManager() {
        for (auto& [device, resources] : devices_) {
            cudaSetDevice(device);
            resources.reset();
        }
    }

    DeviceResources& currentResources() {
        int device = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        std::lock_guard lock(mutex_);
        auto& resources = devices_[device];
        if (!resources)
            resources = std::make_unique<DeviceResources>();
        return *resources;
    }

    // keeps a released object of "device" for reuse (any thread can release it), false when it could not be kept (the caller frees it)
    template <typename Keep>
    bool reuse(const int device, Keep&& keep) noexcept {
        try {
            ReusedObjects* reused = nullptr;
            {
                std::lock_guard lock(mutex_);
                const auto it = devices_.find(device);
                if (it == devices_.end() || !it->second)
                    return false;
                reused = &it->second->reused;
            }
            std::lock_guard lock(reused->mutex);
            keep(*reused);
            return true;
        } catch (...) { return false; }
    }
};
