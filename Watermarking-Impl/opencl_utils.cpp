#include "kernels/kernels.hpp"
#include "opencl_utils.hpp"
#include <af/opencl.h>
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
    cl::Context context(afcl::getContext(true));
    cl::Device device(afcl::getDeviceId(), true);
    cl::Program program;
    // compile opencl kernels
    try {
        program = cl::Program(context, kernels);
        program.build(device, ("-cl-mad-enable -DWINDOW_SIZE=" + std::to_string(p)).c_str());
        return program;
    } catch (const cl::Error& e) {
        cout << "Could not build a kernel, Reason: " << e.what() << "\n\n";
        if (program.get() != NULL && program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device) != CL_BUILD_SUCCESS)
            cout << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << "\n";
    } catch (const std::exception& ex) { cout << ex.what() << "\n"; }
    throw std::runtime_error("Failed to build OpenCL kernels. Check the error messages above for details.");
}

unsigned int maxPow2WorkGroupSize(const cl::Device& device) {
    const unsigned int maxWorkGroup = static_cast<unsigned int>(device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>());
    unsigned int maxValidGroup = 1024;
    while (maxValidGroup > maxWorkGroup)
        maxValidGroup >>= 1;
    return maxValidGroup;
}

unsigned int gridSizeMeCalculate(const cl::Device& device, const cl::Kernel& kernel) {
    const size_t numComputeUnits = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
    const size_t localMemPerCU = device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();
    const size_t kernelLocalMem = kernel.getWorkGroupInfo<CL_KERNEL_LOCAL_MEM_SIZE>(device);
    int maxBlocksPerCU = 4;
    if (kernelLocalMem > 0)
        maxBlocksPerCU = std::min(maxBlocksPerCU, static_cast<int>(localMemPerCU / kernelLocalMem));
    maxBlocksPerCU = std::max(1, maxBlocksPerCU);
    // Compute Units * Max blocks per CU
    return static_cast<unsigned int>(numComputeUnits * maxBlocksPerCU);
}
} // namespace cl_utils