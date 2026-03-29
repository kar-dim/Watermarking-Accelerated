#pragma once
#include "opencl_init.h"
#include <algorithm>
#include <stdexcept>
#include <string>

/*!
 *  \brief  Helper utility functions related to OpenCL.
 *  \author Dimitris Karatzas
 */
namespace cl_utils {
class KernelBuilder {
  private:
    cl::Kernel kernel;
    int argsCounter;

  public:
    KernelBuilder(const cl::Program& program, const char* name);

    /*! \brief setArg overload taking a POD type */
    template <typename... T>
    KernelBuilder& args(const T&... values) {
        (kernel.setArg<T>(argsCounter++, values), ...);
        return *this;
    }

    /*! \brief build the cl::Kernel object */
    cl::Kernel build() const;
};

// helper method to build opencl kernels from source
cl::Program buildKernels(const int p);

// cache for reusing opencl kernels (static/global opencl program for each p)
template <int p>
struct OpenCLKernelCache {
    static cl::Program getProgram() {
        static cl::Program program = buildKernels(p);
        return program;
    }
};

// calculate the maximum power of two work group size for a device
unsigned int maxPow2WorkGroupSize(const cl::Device& device);

// helper method to calculate the number of local groups needed for a specific number of elements and local size, with a maximum of 2560 blocks (used for grid-stride kernels only)
inline int calculateLocalGroupsNumber(const int N, const int localSize) { return std::min((N + localSize - 1) / localSize, 2560); }

// helper method to calculate kernel grid size for persistent threads (Grid Stride Loop)
unsigned int gridSizeMeCalculate(const cl::Device& device, const cl::Kernel& kernel);

// helper method to execute an OpenCL kernel and throw detailed error on failure
template <typename Func>
auto executeKernel(const Func& kernelFunc, const std::string& context) {
    try {
        return kernelFunc();
    } catch (const cl::Error& ex) { throw std::runtime_error("OpenCL Error in " + context + ": " + std::string(ex.what()) + " Error code: " + std::to_string(ex.err()) + "\n"); }
}
} // namespace cl_utils