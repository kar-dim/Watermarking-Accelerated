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
} // namespace cl_utils