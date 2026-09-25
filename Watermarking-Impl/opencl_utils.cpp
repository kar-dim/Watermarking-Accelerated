#include "kernels/kernels.hpp"
#include "kernels/utility_kernels.hpp"
#include "luma_coefficients.hpp"
#include "OclQueueManager.hpp"
#include "opencl_utils.hpp"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <format>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using std::cout;
using std::string;

namespace {
bool hasExtension(const string& extensions, const std::string_view wanted) {
    std::istringstream list(extensions);
    string extension;
    while (list >> extension) {
        if (extension == wanted)
            return true;
    }
    return false;
}

bool hasOpenCLCFeature(const cl::Device& device, const std::string_view wanted) {
    try {
        for (const auto& feature : device.getInfo<CL_DEVICE_OPENCL_C_FEATURES>()) {
            if (std::string_view(feature.name, std::strlen(feature.name)) == wanted)
                return true;
        }
    } catch (const cl::Error&) {}
    return false;
}

string openClStdOption(const cl::Device& device) {
    const string reported = device.getInfo<CL_DEVICE_OPENCL_C_VERSION>();
    if (reported.find("OpenCL C 3.") != string::npos)
        return "-cl-std=CL3.0";
    if (reported.find("OpenCL C 2.") != string::npos)
        return "-cl-std=CL2.0";
    if (reported.find("OpenCL C 1.1") != string::npos)
        return "-cl-std=CL1.1";
    return "-cl-std=CL1.2";
}

enum class SubgroupDialect { None, Core, Khronos, Intel };

SubgroupDialect subgroupDialect(const cl::Device& device) {
    const string extensions = device.getInfo<CL_DEVICE_EXTENSIONS>();
    if (hasExtension(extensions, "cl_intel_subgroups"))
        return SubgroupDialect::Intel;
    if (hasExtension(extensions, "cl_khr_subgroups"))
        return SubgroupDialect::Khronos;
    if (hasOpenCLCFeature(device, "__opencl_c_subgroups"))
        return SubgroupDialect::Core;
    return SubgroupDialect::None;
}

bool supportsWorkGroupCollectives(const cl::Device& device) {
    const string reported = device.getInfo<CL_DEVICE_OPENCL_C_VERSION>();
    if (reported.find("OpenCL C 2.") != string::npos)
        return true;
    return reported.find("OpenCL C 3.") != string::npos && hasOpenCLCFeature(device, "__opencl_c_work_group_collective_functions");
}

string subgroupBuildDefine(const SubgroupDialect dialect) {
    if (dialect == SubgroupDialect::Intel)
        return " -DWM_SUBGROUP_REDUCTIONS=1 -DWM_INTEL_SUBGROUPS=1";
    if (dialect == SubgroupDialect::Khronos)
        return " -DWM_SUBGROUP_REDUCTIONS=1 -DWM_KHR_SUBGROUPS=1";
    if (dialect == SubgroupDialect::Core)
        return " -DWM_SUBGROUP_REDUCTIONS=1";
    return {};
}

const char* subgroupDescription(const SubgroupDialect dialect) {
    if (dialect == SubgroupDialect::Intel)
        return "cl_intel_subgroups";
    if (dialect == SubgroupDialect::Khronos)
        return "cl_khr_subgroups";
    if (dialect == SubgroupDialect::Core)
        return "core OpenCL C subgroups";
    return "portable local-memory reductions";
}

bool tryBuildProgram(cl::Program& program, const cl::Device& device, const string& options, string& buildLog) {
    try {
        program.build(device, options.c_str());
        return true;
    } catch (const cl::Error&) {
        try {
            if (program.get() != nullptr)
                buildLog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
        } catch (const cl::Error&) {}
        return false;
    }
}
} // namespace

namespace cl_utils {
KernelBuilder::KernelBuilder(const cl::Program& program, const char* name) : kernel(program, name), argsCounter(0) {}

cl::Kernel KernelBuilder::build() const { return kernel; }

bool forcePortableReductionsRequested() {
    const char* value = std::getenv("WATERMARK_OPENCL_FORCE_PORTABLE_REDUCTIONS");
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

ReductionMode reductionMode(const cl::Program& program) {
    try {
        const cl::Device device = OclQueueManager::getInstance().getDevice();
        const string options = program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(device);
        if (options.find("WM_WORK_GROUP_REDUCTIONS") != string::npos)
            return ReductionMode::WorkGroupCollective;
        if (options.find("WM_SUBGROUP_REDUCTIONS") != string::npos)
            return ReductionMode::Subgroup;
    } catch (const cl::Error&) {}
    return ReductionMode::Portable;
}

// every kernel of the program can run workgroups of groupSize workitems on the device
bool kernelsFit(cl::Program program, const cl::Device& device, const int groupSize) {
    std::vector<cl::Kernel> programKernels;
    program.createKernels(&programKernels);
    for (const auto& kernel : programKernels)
        if (kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device) < static_cast<size_t>(groupSize))
            return false;
    return true;
}

cl::Program buildKernels(const int p) {
    auto& mgr = OclQueueManager::getInstance();
    cl::Context context = mgr.getContext();
    cl::Device device = mgr.getDevice();
    try {
        const bool forcePortable = forcePortableReductionsRequested();
        const bool useWorkGroupCollectives = !forcePortable && supportsWorkGroupCollectives(device);
        const SubgroupDialect dialect = (forcePortable || useWorkGroupCollectives) ? SubgroupDialect::None : subgroupDialect(device);
        const string reductionOptions = forcePortable ? " -DWM_DISABLE_SUBGROUPS=1" : useWorkGroupCollectives ? " -DWM_WORK_GROUP_REDUCTIONS=1" : subgroupBuildDefine(dialect);
        const char* description = forcePortable ? "forced portable local-memory reductions" : useWorkGroupCollectives ? "work-group collective reductions" : subgroupDescription(dialect);
        for (int groupSize = static_cast<int>(std::min(256u, maxPow2WorkGroupSize(device))); groupSize >= 64; groupSize /= 2) {
            const string baseOptions = openClStdOption(device) + " -DWINDOW_SIZE=" + std::to_string(p) + " -DWG_SIZE=" + std::to_string(groupSize);
            string buildLog;
            cl::Program program(context, kernels);
            if (tryBuildProgram(program, device, baseOptions + reductionOptions, buildLog)) {
                if (kernelsFit(program, device, groupSize)) {
                    cout << "OpenCL kernel p=" << p << ": " << description << ", work-group size " << groupSize << " (" << openClStdOption(device) << ")\n";
                    return program;
                }
                continue;
            }
            if (useWorkGroupCollectives || dialect != SubgroupDialect::None) {
                cout << "NOTE: OpenCL optimized reduction kernel build failed for p=" << p << ", using portable local-memory reductions.\n";
                if (!buildLog.empty())
                    cout << buildLog << "\n";
                program = cl::Program(context, kernels);
                buildLog.clear();
                if (tryBuildProgram(program, device, baseOptions + " -DWM_DISABLE_SUBGROUPS=1", buildLog) && kernelsFit(program, device, groupSize))
                    return program;
            }
            if (!buildLog.empty())
                cout << buildLog << "\n";
        }
    } catch (const std::exception& ex) { cout << ex.what() << "\n"; }
    throw std::runtime_error("Failed to build OpenCL kernels. Check the error messages above for details.");
}

int workGroupSize(const cl::Program& program) {
    const cl::Kernel kernel(program, "me_shift_sums");
    return static_cast<int>(kernel.getWorkGroupInfo<CL_KERNEL_COMPILE_WORK_GROUP_SIZE>(OclQueueManager::getInstance().getDevice())[0]);
}

cl::Program buildUtilityKernels() {
    auto& mgr = OclQueueManager::getInstance();
    cl::Context context = mgr.getContext();
    cl::Device device = mgr.getDevice();
    cl::Program program;
    try {
        program = cl::Program(context, utilityKernels);
        const string options = openClStdOption(device) + " -cl-unsafe-math-optimizations" +
                               std::format(" -DK_LUMA_R={:.9g}f -DK_LUMA_G={:.9g}f -DK_LUMA_B={:.9g}f", CommonUtils::kLumaR, CommonUtils::kLumaG, CommonUtils::kLumaB);
        program.build(device, options.c_str());
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

std::size_t reductionScratchBytes(const cl::Program& program, const char* kernelName, const cl::NDRange& localRange, const std::size_t valuesPerReduction) {
    const auto* localSizes = localRange.get();
    std::size_t workItems = 1;
    for (std::size_t dimension = 0; dimension < localRange.dimensions(); ++dimension)
        workItems *= localSizes[dimension];

    std::size_t scratchSlots = workItems;
    try {
        const cl::Device device = OclQueueManager::getInstance().getDevice();
        const ReductionMode mode = reductionMode(program);
        if (mode == ReductionMode::WorkGroupCollective) {
            scratchSlots = 1;
        } else if (mode == ReductionMode::Subgroup) {
            const cl::Kernel kernel(program, kernelName);
            const std::size_t subgroupCount = kernel.getSubGroupInfo<CL_KERNEL_SUB_GROUP_COUNT_FOR_NDRANGE>(device, localRange);
            if (subgroupCount > 0 && subgroupCount <= workItems)
                scratchSlots = subgroupCount;
        }
    } catch (const cl::Error&) {
        // A conservative work-item-sized allocation is valid for either kernel path.
    }
    return scratchSlots * valuesPerReduction * sizeof(float);
}

void launchRowMajorRgbToColMajor(const cl::Buffer& src, const cl::Buffer& rgbDst, const cl::Buffer& grayDst, const int width, const int height, cl::CommandQueue& queue) {
    constexpr int blockSize = 16;
    queue.enqueueNDRangeKernel(KernelBuilder(UtilityKernelCache::getProgram(), "row_major_rgb_to_col_major").args(src, rgbDst, grayDst, width, height).build(), cl::NullRange,
        cl::NDRange(roundUp(width, blockSize), roundUp(height, blockSize)), cl::NDRange(blockSize, blockSize));
}

void launchU8ToFloatGray(const cl::Buffer& input, const cl::Buffer& output, const int planeSize, const int numChannels, cl::CommandQueue& queue) {
    constexpr int localSize = 256;
    const int globalSize = calculateLocalGroupsNumber(planeSize, localSize) * localSize;
    queue.enqueueNDRangeKernel(
        KernelBuilder(UtilityKernelCache::getProgram(), "u8_to_float_gray").args(input, output, planeSize, numChannels).build(), cl::NullRange, cl::NDRange(globalSize), cl::NDRange(localSize));
}

void launchColMajorToRowMajorU8(const cl::Buffer& src, const cl::Buffer& dst, const int width, const int height, const int channels, cl::CommandQueue& queue) {
    constexpr int blockSize = 16;
    queue.enqueueNDRangeKernel(KernelBuilder(UtilityKernelCache::getProgram(), "col_major_to_row_major_u8").args(src, dst, width, height).build(), cl::NullRange,
        cl::NDRange(roundUp(width, blockSize), roundUp(height, blockSize), channels), cl::NDRange(blockSize, blockSize));
}

void launchPitchedToFloat(const cl::Buffer& src, const cl::Buffer& dst, const int width, const int height, const int pitch, cl::CommandQueue& queue) {
    constexpr int blockSize = 16;
    queue.enqueueNDRangeKernel(KernelBuilder(UtilityKernelCache::getProgram(), "pitched_to_float").args(src, dst, width, height, pitch).build(), cl::NullRange,
        cl::NDRange(roundUp(width, blockSize), roundUp(height, blockSize)), cl::NDRange(blockSize, blockSize));
}
} // namespace cl_utils
