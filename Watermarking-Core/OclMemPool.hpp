#pragma once
#include "CheckedSize.hpp"
#include "opencl_init.h"
#include <algorithm>
#include <cstddef>
#include <mutex>
#include <unordered_map>

/*!
 *  \brief  Free-list OpenCL buffer pool owned by one device context
 *          Allocations round to powers of 2 for bucket reuse, capped by free VRAM
 *  \author Dimitris Karatzas
 */
class OclMemPool {
  private:
    std::unordered_multimap<size_t, cl_mem> memList;
    size_t pooledBytes = 0;
    size_t maxPoolBytes = 0;
    std::mutex mtx;

    void clearIdle() {
        for (auto& [sz, m] : memList)
            clReleaseMemObject(m);
        memList.clear();
        pooledBytes = 0;
    }

  public:
    OclMemPool() = default;
    ~OclMemPool() { reset(); }

    OclMemPool(const OclMemPool&) = delete;
    OclMemPool& operator=(const OclMemPool&) = delete;

    void setCapacity(const size_t vramBytes) { maxPoolBytes = std::clamp<size_t>(vramBytes / 8, 1, 512ull << 20); }

    cl_mem acquire(const size_t bytes, cl_context ctx) {
        const size_t rounded = InternalUtils::checkedPowerOfTwo(bytes);
        std::lock_guard lock(mtx);
        auto it = memList.find(rounded);
        if (it != memList.end()) {
            cl_mem m = it->second;
            pooledBytes -= rounded;
            memList.erase(it);
            return m;
        }
        cl_int err;
        cl_mem m = clCreateBuffer(ctx, CL_MEM_READ_WRITE, rounded, nullptr, &err);
        if ((err == CL_MEM_OBJECT_ALLOCATION_FAILURE || err == CL_OUT_OF_RESOURCES) && !memList.empty()) {
            clearIdle();
            m = clCreateBuffer(ctx, CL_MEM_READ_WRITE, rounded, nullptr, &err);
        }
        if (err != CL_SUCCESS)
            throw cl_utils::DriverError(err, "clCreateBuffer");
        return m;
    }

    void release(const size_t bytes, cl_mem m) noexcept {
        if (!m)
            return;
        const size_t rounded = std::bit_ceil(bytes);
        std::lock_guard lock(mtx);
        if (maxPoolBytes > 0 && (rounded > maxPoolBytes || pooledBytes > maxPoolBytes - rounded)) {
            clReleaseMemObject(m);
            return;
        }
        try {
            memList.emplace(rounded, m);
            pooledBytes += rounded;
        } catch (...) { clReleaseMemObject(m); }
    }

    void reset() {
        std::lock_guard lock(mtx);
        clearIdle();
    }
};
