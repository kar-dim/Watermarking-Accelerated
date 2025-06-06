#pragma once
#if defined(_USE_CUDA_)
#include <cuda_runtime.h>
#elif defined(_USE_OPENCL_)
#include "opencl_init.h"
#endif

template <typename T>
class HostMemory {
public:
    HostMemory(const size_t size) 
    {
#if defined(_USE_CUDA_)
        cudaHostAlloc(&ptr, size * sizeof(T), cudaHostAllocDefault);
#elif defined(_USE_OPENCL_)
        pinnedBuffer = cl::Buffer(cl::Context(afcl::getContext(false)), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, size * sizeof(T));
        ptr = static_cast<T*>(queue.enqueueMapBuffer(pinnedBuffer, CL_TRUE, CL_MAP_WRITE, 0, size * sizeof(T)));
#elif defined(_USE_EIGEN_)
        ptr = new T[size];
#endif
    }
    ~HostMemory()
    {
#if defined(_USE_CUDA_)
        if (ptr) cudaFreeHost(ptr);
#elif defined(_USE_OPENCL_)
        if (ptr) queue.enqueueUnmapMemObject(pinnedBuffer, ptr);
#elif defined(_USE_EIGEN_)
        delete[] ptr;
#endif
    }

    T* get() { return ptr; }

private:
    T* ptr = nullptr;

#if defined(_USE_OPENCL_)
    cl::Buffer pinnedBuffer;
    cl::CommandQueue queue{ afcl::getQueue(false) };
#endif

};