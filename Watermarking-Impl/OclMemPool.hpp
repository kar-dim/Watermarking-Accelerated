#pragma once
#include "opencl_init.h"
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

/*!
 *  \brief  Free-list memory pool for OpenCL cl_mem buffers, byte size is the key,
 *          owned by OclQueueManager and reset() is called automatically on device/context switch
 *  \author Dimitris Karatzas
 */
class OclMemPool {
  private:
    std::unordered_multimap<size_t, cl_mem> memList;
    std::mutex mtx;

  public:
    OclMemPool() = default;
    ~OclMemPool() { reset(); }

    OclMemPool(const OclMemPool&) = delete;
    OclMemPool& operator=(const OclMemPool&) = delete;

    cl_mem acquire(size_t bytes, cl_context ctx) {
        std::lock_guard lock(mtx);
        auto it = memList.find(bytes);
        if (it != memList.end()) {
            cl_mem m = it->second;
            memList.erase(it);
            return m;
        }
        cl_int err;
        cl_mem m = clCreateBuffer(ctx, CL_MEM_READ_WRITE, bytes, nullptr, &err);
        if (err != CL_SUCCESS)
            throw std::runtime_error("clCreateBuffer failed: " + std::to_string(err));
        return m;
    }

    void release(size_t bytes, cl_mem m) {
        if (!m)
            return;
        std::lock_guard lock(mtx);
        memList.emplace(bytes, m);
    }

    void reset() {
        std::lock_guard lock(mtx);
        for (auto& [sz, m] : memList)
            clReleaseMemObject(m);
        memList.clear();
    }
};
