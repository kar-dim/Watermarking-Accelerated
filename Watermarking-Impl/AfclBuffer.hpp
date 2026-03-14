#pragma once
#include "opencl_init.h"
#include <arrayfire.h>
#include <concepts>
#include <utility>

/*!
 *  \brief  Holds an arrayfire array, handles the cl::Buffer instantiation from it,
            and handles the arrayfire cl_mem pointer cleanup boilerplate
 *  \author Dimitris Karatzas
 */
class AfclBuffer {
  private:
    const af::array* pArr = nullptr;
    af::array tempArr; // only used if we create a new array
    cl_mem* rawMem = nullptr;
    cl::Buffer clBuffer;

  public:
    // ctors
    explicit AfclBuffer(const af::array& a) : pArr(&a), rawMem(a.device<cl_mem>()), clBuffer{*rawMem, true} {}
    AfclBuffer(const dim_t size, af::dtype type) : tempArr(size, type), pArr(&tempArr), rawMem(tempArr.device<cl_mem>()), clBuffer{*rawMem, true} {}
    AfclBuffer(const dim_t rows, const dim_t cols, af::dtype type) : tempArr(rows, cols, type), pArr(&tempArr), rawMem(tempArr.device<cl_mem>()), clBuffer{*rawMem, true} {}
    explicit AfclBuffer(const af::dim4& dims, af::dtype type) : tempArr(dims, type), pArr(&tempArr), rawMem(tempArr.device<cl_mem>()), clBuffer{*rawMem, true} {}

    // destructor
    ~AfclBuffer() { unlock(); }

    // explicit destroy
    void unlock() {
        if (rawMem) {
            if (pArr)
                pArr->unlock();
            delete rawMem;
            rawMem = nullptr;
            clBuffer = cl::Buffer();
        }
    }

    // disable copy
    AfclBuffer(const AfclBuffer&) = delete;
    AfclBuffer& operator=(const AfclBuffer&) = delete;

    // allow move + move an rvalue arrayfire array
    AfclBuffer(AfclBuffer&& other) noexcept : tempArr(std::move(other.tempArr)), rawMem(other.rawMem), clBuffer(std::move(other.clBuffer)) {
        pArr = other.pArr == &other.tempArr ? &tempArr : other.pArr;
        other.rawMem = nullptr;
        other.pArr = nullptr;
    }
    explicit AfclBuffer(af::array&& a) : tempArr(std::move(a)), pArr(&tempArr), rawMem(tempArr.device<cl_mem>()), clBuffer{*rawMem, true} {}

    // get the underlying cl::Buffer to be passed to OpenCL kernels
    const cl::Buffer& get() const { return clBuffer; }

    // get the underlying arrayfire array
    const af::array& getArray() const { return *pArr; }

    // unlock and get the array's value back to host (for 1-element arrays)
    template <typename T>
    T scalar() {
        unlock();
        return pArr->scalar<T>();
    }

    // static helper to unlock multiple arrays
    template <std::same_as<AfclBuffer>... Args>
    static void unlockArrays(Args&... arrays) {
        (arrays.unlock(), ...);
    }
};