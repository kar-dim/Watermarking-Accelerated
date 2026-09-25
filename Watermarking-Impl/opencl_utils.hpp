#pragma once
#include "OclQueueManager.hpp"
#include "opencl_init.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>

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

// sets all arguments of a (cached) kernel in order
template <typename... T>
void setArgs(cl::Kernel& kernel, const T&... values) {
    cl_uint index = 0;
    (kernel.setArg(index++, values), ...);
}

// helper method to build watermark opencl kernels from source (WINDOW_SIZE=p), workgroup size (WG_SIZE) is probed: the largest power of
// 2 <= 256 the device supports for which every kernel of the program fits (register / local memory limits), halved to 64
cl::Program buildKernels(const int p);

// probed workgroup size
int workGroupSize(const cl::Program& program);

// helper method to build utility opencl kernels from source (no WINDOW_SIZE dependency)
cl::Program buildUtilityKernels();

// WATERMARK_OPENCL_FORCE_PORTABLE_REDUCTIONS=1 disables optimized reductions
bool forcePortableReductionsRequested();

enum class ReductionMode { Portable, WorkGroupCollective, Subgroup };

// inspect the selected reduction implementation in a built watermark program
ReductionMode reductionMode(const cl::Program& program);

// cache for reusing opencl kernels (static/global opencl program for each p) for each device
// automatically invalidates when OclQueueManager's context generation changes (device switch)
template <int p>
struct OpenCLKernelCache {
    static cl::Program getProgram() {
        static std::unordered_map<int, cl::Program> programs;
        static uint32_t cachedGeneration = 0;
        auto& mgr = OclQueueManager::getInstance();
        const uint32_t currentGen = mgr.getContextGeneration();
        if (cachedGeneration != currentGen) {
            programs.clear();
            cachedGeneration = currentGen;
        }
        const int deviceId = mgr.getDeviceIndex();
        const int cacheKey = (deviceId << 1) | static_cast<int>(forcePortableReductionsRequested());
        if (programs.find(cacheKey) == programs.end())
            programs[cacheKey] = buildKernels(p);
        return programs[cacheKey];
    }
};

// cache for utility kernels (transpose, etc) it is per device, not per p
struct UtilityKernelCache {
    static cl::Program getProgram() {
        static std::unordered_map<int, cl::Program> programs;
        static uint32_t cachedGeneration = 0;
        auto& mgr = OclQueueManager::getInstance();
        const uint32_t currentGen = mgr.getContextGeneration();
        if (cachedGeneration != currentGen) {
            programs.clear();
            cachedGeneration = currentGen;
        }
        const int deviceId = mgr.getDeviceIndex();
        if (programs.find(deviceId) == programs.end())
            programs[deviceId] = buildUtilityKernels();
        return programs[deviceId];
    }
};

// calculate the maximum power of two work group size for a device
unsigned int maxPow2WorkGroupSize(const cl::Device& device);

// local-memory bytes needed by a reduction kernel for the requested launch shape
// (one slot per subgroup on the optimized path, one per work-item on the fallback)
std::size_t reductionScratchBytes(const cl::Program& program, const char* kernelName, const cl::NDRange& localRange, std::size_t valuesPerReduction);

// row-major planar uchar RGB -> column-major planar uchar RGB + column-major float luma, one tiled transpose
void launchRowMajorRgbToColMajor(const cl::Buffer& src, const cl::Buffer& rgbDst, const cl::Buffer& grayDst, const int width, const int height, cl::CommandQueue& queue);

// uint8 col-major to float col-major grayscale on GPU, with optional RGB weighting
void launchU8ToFloatGray(const cl::Buffer& input, const cl::Buffer& output, const int planeSize, const int numChannels, cl::CommandQueue& queue);

// coalesced tiled transpose: column-major uchar -> row-major uchar on GPU, multichannel via z-dimension
void launchColMajorToRowMajorU8(const cl::Buffer& src, const cl::Buffer& dst, const int width, const int height, const int channels, cl::CommandQueue& queue);

// coalesced tiled transpose: row-major uchar (with pitch) -> column-major float on GPU (port of CUDA pitchedToFloat kernel)
void launchPitchedToFloat(const cl::Buffer& src, const cl::Buffer& dst, const int width, const int height, const int pitch, cl::CommandQueue& queue);

// helper method to calculate the number of local groups needed for a specific number of elements and local size, with a maximum of 2560 blocks (used for grid-stride kernels only)
inline int calculateLocalGroupsNumber(const int N, const int localSize) { return std::min((N + localSize - 1) / localSize, 2560); }

// rounds n up to the nearest multiple of blockSize (used to compute global NDRange sizes for 2D tiled kernels)
inline int roundUp(const int n, const int blockSize) { return ((n + blockSize - 1) / blockSize) * blockSize; }

// helper method to execute an OpenCL kernel and throw detailed error on failure
template <typename Func>
auto executeKernel(const Func& kernelFunc, const std::string& context) {
    try {
        return kernelFunc();
    } catch (const cl::Error& ex) { throw std::runtime_error("OpenCL Error in " + context + ": " + std::string(ex.what()) + " Error code: " + std::to_string(ex.err()) + "\n"); }
}
} // namespace cl_utils