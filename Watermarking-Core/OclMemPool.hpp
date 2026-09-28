#pragma once
#include "opencl_init.h"
#include <cstddef>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

/*!
 *  \brief  Free-list memory pool for OpenCL cl_mem buffers. Allocations are rounded up to the next
 *          power of 2 so that images with similar dimensions are put in the same "bucket",
 *          a maximum capacity cap, set based on available VRAM prevents overflow to RAM.
 *          Owned by OclQueueManager. reset() is called automatically on device/context switch
 *  \author Dimitris Karatzas
 */
class OclMemPool {
  private:
    std::unordered_multimap<size_t, cl_mem> memList;
    size_t pooledBytes = 0;
    size_t maxPoolBytes = 0;
    std::mutex mtx;

    static size_t roundUpPow2(size_t v) {
        if (v <= 1)
            return 1;
        v--;
        v |= v >> 1;
        v |= v >> 2;
        v |= v >> 4;
        v |= v >> 8;
        v |= v >> 16;
        v |= v >> 32;
        return v + 1;
    }

  public:
    OclMemPool() = default;
    ~OclMemPool() { reset(); }

    OclMemPool(const OclMemPool&) = delete;
    OclMemPool& operator=(const OclMemPool&) = delete;

    void setCapacity(const size_t vramBytes) { maxPoolBytes = static_cast<size_t>(vramBytes * 0.95); }

    cl_mem acquire(const size_t bytes, cl_context ctx) {
        const size_t rounded = roundUpPow2(bytes);
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
        if (err != CL_SUCCESS)
            throw std::runtime_error("clCreateBuffer failed: " + std::to_string(err));
        return m;
    }

    void release(const size_t bytes, cl_mem m) {
        if (!m)
            return;
        const size_t rounded = roundUpPow2(bytes);
        std::lock_guard lock(mtx);
        if (maxPoolBytes > 0 && pooledBytes + rounded > maxPoolBytes) {
            clReleaseMemObject(m);
            return;
        }
        pooledBytes += rounded;
        memList.emplace(rounded, m);
    }

    void reset() {
        std::lock_guard lock(mtx);
        for (auto& [sz, m] : memList)
            clReleaseMemObject(m);
        memList.clear();
        pooledBytes = 0;
    }
};
