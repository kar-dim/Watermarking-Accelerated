#pragma once
#include "CheckedSize.hpp"
#include "OclQueueManager.hpp"
#include "opencl_init.h"
#include <memory>
#include <stdexcept>
#include <utility>

/*!
 *  \brief  GPU buffer class for OpenCL, equivalent to CUDA CudaArray<T>
 *          Uses OclQueueManager's shared memory pool for buffer (re)use.
 *  \author Dimitris Karatzas
 */
template <typename T>
class OclArray {
  private:
    cl_mem mem = nullptr;
    int rows = 0;
    int cols = 0;
    int channels = 1;
    cl_command_queue queue = nullptr;
    std::shared_ptr<OclQueueManager> owner;

    static void checkCl(const cl_int status, const char* operation) {
        if (status != CL_SUCCESS)
            throw cl_utils::DriverError(status, operation);
    }

    void alloc() {
        if (size() > 0) {
            owner = OclQueueManager::forQueue(queue);
            mem = owner->acquireBuffer(bytes(), queue);
        }
    }

    void freeArray() {
        if (mem) {
            // The caller's thread may now have a different GPU selected!
            owner->releaseBuffer(bytes(), mem);
            mem = nullptr;
        }
    }

  public:
    OclArray() = default;

    explicit OclArray(const int count, cl_command_queue queue) : rows(count), cols(1), queue(queue) { alloc(); }

    OclArray(const int rows, const int cols, cl_command_queue queue) : rows(rows), cols(cols), queue(queue) { alloc(); }

    OclArray(const int rows, const int cols, const int channels, cl_command_queue queue) : rows(rows), cols(cols), channels(channels), queue(queue) { alloc(); }

    // constructors that accept pointer data, pass CL_TRUE wait until copy is finished before returning
    OclArray(const int rows, const int cols, const T* hostData, cl_command_queue queue) : rows(rows), cols(cols), queue(queue) {
        alloc();
        try {
            if (mem)
                checkCl(clEnqueueWriteBuffer(queue, mem, CL_TRUE, 0, bytes(), hostData, 0, nullptr, nullptr), "clEnqueueWriteBuffer");
        } catch (...) {
            freeArray();
            throw;
        }
    }

    OclArray(const int rows, const int cols, const int channels, const T* hostData, cl_command_queue queue) : rows(rows), cols(cols), channels(channels), queue(queue) {
        alloc();
        try {
            if (mem)
                checkCl(clEnqueueWriteBuffer(queue, mem, CL_TRUE, 0, bytes(), hostData, 0, nullptr, nullptr), "clEnqueueWriteBuffer");
        } catch (...) {
            freeArray();
            throw;
        }
    }

    ~OclArray() { freeArray(); }

    OclArray(const OclArray&) = delete;
    OclArray& operator=(const OclArray&) = delete;

    OclArray(OclArray&& o) noexcept : mem(o.mem), rows(o.rows), cols(o.cols), channels(o.channels), queue(o.queue), owner(std::move(o.owner)) {
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
            owner = std::move(o.owner);
            o.mem = nullptr;
            o.rows = o.cols = 0;
            o.channels = 1;
        }
        return *this;
    }

    cl_mem data() { return mem; }
    cl_mem data() const { return mem; }
    int getRows() const { return rows; }
    int getCols() const { return cols; }
    int getChannels() const { return channels; }
    int size() const { return InternalUtils::checkedElements(rows, cols, channels); }
    size_t bytes() const { return InternalUtils::checkedProduct(static_cast<size_t>(size()), sizeof(T)); }
    bool empty() const { return mem == nullptr; }
    cl_command_queue getQueue() const { return queue; }
    const std::shared_ptr<OclQueueManager>& getOwner() const { return owner; }

    cl::Buffer clBuffer() const { return cl::Buffer(mem, true); }

    void fillZero() {
        if (mem) {
            T zero{};
            checkCl(clEnqueueFillBuffer(queue, mem, &zero, sizeof(T), 0, bytes(), 0, nullptr, nullptr), "clEnqueueFillBuffer");
        }
    }

    T scalar() const {
        T val{};
        if (mem)
            checkCl(clEnqueueReadBuffer(queue, mem, CL_TRUE, 0, sizeof(T), &val, 0, nullptr, nullptr), "clEnqueueReadBuffer");
        return val;
    }

    void toHost(T* dst) const {
        if (mem)
            checkCl(clEnqueueReadBuffer(queue, mem, CL_TRUE, 0, bytes(), dst, 0, nullptr, nullptr), "clEnqueueReadBuffer");
    }

    void toHostAsync(T* dst) const {
        if (mem)
            checkCl(clEnqueueReadBuffer(queue, mem, CL_FALSE, 0, bytes(), dst, 0, nullptr, nullptr), "clEnqueueReadBuffer");
    }

    // if the destination needs a pitch, we can use cudaMemcpy2DAsync to copy the data row by row, with the specified pitch for the destination
    void toHostPitched(T* dst, const int rowElements, const size_t dstPitchBytes) const {
        if (rowElements <= 0 || size() % rowElements != 0 || dstPitchBytes < InternalUtils::checkedProduct(static_cast<size_t>(rowElements), sizeof(T)))
            throw std::invalid_argument("Invalid pitched buffer layout");
        InternalUtils::checkedProduct(dstPitchBytes, static_cast<size_t>(size() / rowElements));
        if (mem) {
            const size_t rowBytes = static_cast<size_t>(rowElements) * sizeof(T);
            const size_t origin[3] = {0, 0, 0};
            const size_t region[3] = {rowBytes, static_cast<size_t>(size() / rowElements), 1};
            checkCl(clEnqueueReadBufferRect(queue, mem, CL_TRUE, origin, origin, region, rowBytes, 0, dstPitchBytes, 0, dst, 0, nullptr, nullptr), "clEnqueueReadBufferRect");
        }
    }

    static OclArray zeros(const int count, cl_command_queue queue) {
        OclArray arr(count, queue);
        arr.fillZero();
        return arr;
    }

    static OclArray zeros(const int rows, const int cols, cl_command_queue queue) {
        OclArray arr(rows, cols, queue);
        arr.fillZero();
        return arr;
    }

    static OclArray zeros(const int rows, const int cols, const int channels, cl_command_queue queue) {
        OclArray arr(rows, cols, channels, queue);
        arr.fillZero();
        return arr;
    }

    OclArray clone() const {
        OclArray copy(rows, cols, channels, queue);
        if (mem && copy.mem)
            checkCl(clEnqueueCopyBuffer(queue, mem, copy.mem, 0, 0, bytes(), 0, nullptr, nullptr), "clEnqueueCopyBuffer");
        return copy;
    }
};
