#pragma once
#if defined(_USE_CUDA_)
#include "../CudaCheck.hpp"
#include <cuda_runtime.h>
#elif defined(_USE_OPENCL_)
#include "opencl_init.h"
#include "OclQueueManager.hpp"
#elif defined(_USE_EIGEN_)
#include <memory>
#endif

/*!
 *  \brief  Used only for video, holds pinned host memory for fast GPU<->CPU transfers, or simple heap memory for CPU implementation.
 *  \author Dimitris Karatzas
 */
template <typename T>
class HostMemory {
  public:
    HostMemory(const size_t size) {
#if defined(_USE_CUDA_)
        CUDA_CHECK(cudaHostAlloc(&ptr, size * sizeof(T), cudaHostAllocDefault));
#elif defined(_USE_OPENCL_)
        queue = OclQueueManager::getInstance().getQueue();
        pinnedBuffer = cl::Buffer(OclQueueManager::getInstance().getContext(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, size * sizeof(T));
        ptr = static_cast<T*>(queue.enqueueMapBuffer(pinnedBuffer, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, size * sizeof(T)));
#elif defined(_USE_EIGEN_)
        pinnedBuffer = std::make_unique<T[]>(size);
        ptr = pinnedBuffer.get();
#endif
    }
    ~HostMemory() {
#if defined(_USE_CUDA_)
        if (ptr)
            cudaFreeHost(ptr);
#elif defined(_USE_OPENCL_)
        if (ptr)
            queue.enqueueUnmapMemObject(pinnedBuffer, ptr);
#endif
    }

    // prevent copying to avoid double free or dangling pointer issues
    HostMemory(const HostMemory&) = delete;
    HostMemory& operator=(const HostMemory&) = delete;

    T* get() { return ptr; }

  private:
    T* ptr = nullptr;

#if defined(_USE_OPENCL_)
    cl::CommandQueue queue;
    cl::Buffer pinnedBuffer;
#elif defined(_USE_EIGEN_)
    std::unique_ptr<T[]> pinnedBuffer;
#endif
};
