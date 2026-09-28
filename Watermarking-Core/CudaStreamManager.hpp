#pragma once
#include "CudaCheck.hpp"
#include "CudaMemPool.hpp"
#include <cuda_runtime.h>
#include <memory>
#include <mutex>
#include <unordered_map>

/*!
 *  \brief Owns one CUDA stream and memory pool per device
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

    CudaStreamManager(const CudaStreamManager&) = delete;
    CudaStreamManager& operator=(const CudaStreamManager&) = delete;
    CudaStreamManager(CudaStreamManager&&) = delete;
    CudaStreamManager& operator=(CudaStreamManager&&) = delete;

  private:
    struct DeviceResources {
        cudaStream_t computeStream = nullptr;
        CudaMemPool pool;

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
};
