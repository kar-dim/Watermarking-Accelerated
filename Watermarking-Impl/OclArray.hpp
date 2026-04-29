#pragma once
#include "OclQueueManager.hpp"
#include "opencl_init.h"

/*!
 *  \brief  GPU buffer class for OpenCL, equivalent to CUDA GpuArray<T>
 *          Uses OclQueueManager's shared memory pool for buffer (re)use.
 *  \author Dimitris Karatzas
 */
template <typename T>
class OclArray {
  private:
    cl_mem mem = nullptr;
    unsigned int rows = 0;
    unsigned int cols = 0;
    unsigned int channels = 1;
    cl_command_queue queue = nullptr;

    void alloc() {
        if (size() > 0) {
            auto& mgr = OclQueueManager::getInstance();
            mem = mgr.getPool().acquire(bytes(), mgr.getContextRaw());
        }
    }

    void freeArray() {
        if (mem) {
            OclQueueManager::getInstance().getPool().release(bytes(), mem);
            mem = nullptr;
        }
    }

  public:
    OclArray() = default;

    explicit OclArray(unsigned int count, cl_command_queue queue) : rows(count), cols(1), queue(queue) { alloc(); }

    OclArray(unsigned int rows, unsigned int cols, cl_command_queue queue) : rows(rows), cols(cols), queue(queue) { alloc(); }

    OclArray(unsigned int rows, unsigned int cols, unsigned int channels, cl_command_queue queue) : rows(rows), cols(cols), channels(channels), queue(queue) { alloc(); }

    // constructors that accept pointer data, pass CL_TRUE wait until copy is finished before returning
    OclArray(unsigned int rows, unsigned int cols, const T* hostData, cl_command_queue queue) : rows(rows), cols(cols), queue(queue) {
        alloc();
        if (mem)
            clEnqueueWriteBuffer(queue, mem, CL_TRUE, 0, bytes(), hostData, 0, nullptr, nullptr);
    }

    OclArray(unsigned int rows, unsigned int cols, unsigned int channels, const T* hostData, cl_command_queue queue) : rows(rows), cols(cols), channels(channels), queue(queue) {
        alloc();
        if (mem)
            clEnqueueWriteBuffer(queue, mem, CL_TRUE, 0, bytes(), hostData, 0, nullptr, nullptr);
    }

    ~OclArray() { freeArray(); }

    OclArray(const OclArray&) = delete;
    OclArray& operator=(const OclArray&) = delete;

    OclArray(OclArray&& o) noexcept : mem(o.mem), rows(o.rows), cols(o.cols), channels(o.channels), queue(o.queue) {
        o.mem = nullptr;
        o.rows = o.cols = 0;
        o.channels = 1;
    }

    OclArray& operator=(OclArray&& o) noexcept {
        if (this != &o) {
            freeArray();
            mem = o.mem;
            rows = o.rows;
            cols = o.cols;
            channels = o.channels;
            queue = o.queue;
            o.mem = nullptr;
            o.rows = o.cols = 0;
            o.channels = 1;
        }
        return *this;
    }

    cl_mem data() { return mem; }
    cl_mem data() const { return mem; }
    unsigned int getRows() const { return rows; }
    unsigned int getCols() const { return cols; }
    unsigned int getChannels() const { return channels; }
    unsigned int size() const { return rows * cols * channels; }
    size_t bytes() const { return static_cast<size_t>(size()) * sizeof(T); }
    bool empty() const { return mem == nullptr; }
    cl_command_queue getQueue() const { return queue; }

    cl::Buffer clBuffer() const { return cl::Buffer(mem, true); }

    void fillZero() {
        if (mem) {
            T zero{};
            clEnqueueFillBuffer(queue, mem, &zero, sizeof(T), 0, bytes(), 0, nullptr, nullptr);
        }
    }

    T scalar() const {
        T val{};
        if (mem)
            clEnqueueReadBuffer(queue, mem, CL_TRUE, 0, sizeof(T), &val, 0, nullptr, nullptr);
        return val;
    }

    void toHost(T* dst) const {
        if (mem)
            clEnqueueReadBuffer(queue, mem, CL_TRUE, 0, bytes(), dst, 0, nullptr, nullptr);
    }

    void toHostAsync(T* dst) const {
        if (mem)
            clEnqueueReadBuffer(queue, mem, CL_FALSE, 0, bytes(), dst, 0, nullptr, nullptr);
    }

    static OclArray zeros(unsigned int count, cl_command_queue queue) {
        OclArray arr(count, queue);
        arr.fillZero();
        return arr;
    }

    static OclArray zeros(unsigned int rows, unsigned int cols, cl_command_queue queue) {
        OclArray arr(rows, cols, queue);
        arr.fillZero();
        return arr;
    }

    static OclArray zeros(unsigned int rows, unsigned int cols, unsigned int channels, cl_command_queue queue) {
        OclArray arr(rows, cols, channels, queue);
        arr.fillZero();
        return arr;
    }

    OclArray clone() const {
        OclArray copy(rows, cols, channels, queue);
        if (mem && copy.mem)
            clEnqueueCopyBuffer(queue, mem, copy.mem, 0, 0, bytes(), 0, nullptr, nullptr);
        return copy;
    }
};
