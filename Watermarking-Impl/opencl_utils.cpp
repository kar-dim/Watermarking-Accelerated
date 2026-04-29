#include "kernels/kernels.hpp"
#include "kernels/utility_kernels.hpp"
#include "OclQueueManager.hpp"
#include "opencl_utils.hpp"
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

using std::cout;
using std::string;

namespace cl_utils {
KernelBuilder::KernelBuilder(const cl::Program& program, const char* name) : kernel(program, name), argsCounter(0) {}

cl::Kernel KernelBuilder::build() const { return kernel; }

cl::Program buildKernels(const int p) {
    auto& mgr = OclQueueManager::getInstance();
    cl::Context context = mgr.getContext();
    cl::Device device = mgr.getDevice();
    cl::Program program;
    try {
        program = cl::Program(context, kernels);
        program.build(device, ("-cl-unsafe-math-optimizations -DWINDOW_SIZE=" + std::to_string(p)).c_str());
        return program;
    } catch (const cl::Error& e) {
        cout << "Could not build a kernel, Reason: " << e.what() << "\n\n";
        if (program.get() != NULL && program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device) != CL_BUILD_SUCCESS)
            cout << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << "\n";
    } catch (const std::exception& ex) { cout << ex.what() << "\n"; }
    throw std::runtime_error("Failed to build OpenCL kernels. Check the error messages above for details.");
}

cl::Program buildUtilityKernels() {
    auto& mgr = OclQueueManager::getInstance();
    cl::Context context = mgr.getContext();
    cl::Device device = mgr.getDevice();
    cl::Program program;
    try {
        program = cl::Program(context, utilityKernels);
        program.build(device, "-cl-unsafe-math-optimizations");
        return program;
    } catch (const cl::Error& e) {
        cout << "Could not build utility kernels, Reason: " << e.what() << "\n\n";
        if (program.get() != NULL && program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device) != CL_BUILD_SUCCESS)
            cout << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << "\n";
    } catch (const std::exception& ex) { cout << ex.what() << "\n"; }
    throw std::runtime_error("Failed to build OpenCL utility kernels. Check the error messages above for details.");
}

unsigned int maxPow2WorkGroupSize(const cl::Device& device) {
    const unsigned int maxWorkGroup = static_cast<unsigned int>(device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>());
    unsigned int maxValidGroup = 1024;
    while (maxValidGroup > maxWorkGroup)
        maxValidGroup >>= 1;
    return maxValidGroup;
}

void launchRowMajorToColMajorFloat(const cl::Buffer& src, const cl::Buffer& dst, const int width, const int height, const int channels, cl::CommandQueue& queue) {
    constexpr int blockSize = 16;
    queue.enqueueNDRangeKernel(KernelBuilder(UtilityKernelCache::getProgram(), "row_major_to_col_major_float").args(src, dst, width, height).build(), cl::NullRange,
                               cl::NDRange(((width + (blockSize - 1)) / blockSize) * blockSize, ((height + (blockSize - 1)) / blockSize) * blockSize, channels), cl::NDRange(blockSize, blockSize, 1));
}
void launchColMajorToRowMajorU8(const cl::Buffer& src, const cl::Buffer& dst, const int width, const int height, const int channels, cl::CommandQueue& queue) {
    constexpr int blockSize = 16;
    queue.enqueueNDRangeKernel(KernelBuilder(UtilityKernelCache::getProgram(), "col_major_to_row_major_u8").args(src, dst, width, height).build(), cl::NullRange,
                               cl::NDRange(((width + (blockSize - 1)) / blockSize) * blockSize, ((height + (blockSize - 1)) / blockSize) * blockSize, channels), cl::NDRange(blockSize, blockSize, 1));
}
void launchPitchedToFloat(const cl::Buffer& src, const cl::Buffer& dst, const int width, const int height, const int pitch, cl::CommandQueue& queue) {
    constexpr int blockSize = 16;
    queue.enqueueNDRangeKernel(KernelBuilder(UtilityKernelCache::getProgram(), "pitched_to_float").args(src, dst, width, height, pitch).build(), cl::NullRange,
                               cl::NDRange(((width + (blockSize - 1)) / blockSize) * blockSize, ((height + (blockSize - 1)) / blockSize) * blockSize), cl::NDRange(blockSize, blockSize));
}
} // namespace cl_utils
