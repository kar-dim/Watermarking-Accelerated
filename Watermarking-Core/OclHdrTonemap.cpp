#include "CheckedSize.hpp"
#include "HdrTonemapCurves.hpp"
#include "kernels/hdr_kernels.hpp"
#include "OclHdrTonemap.hpp"
#include "OclQueueManager.hpp"
#include "opencl_utils.hpp"
#include <algorithm>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
cl::Program buildHdrKernels() {
    auto& manager = OclQueueManager::getInstance();
    cl::Program program(manager.getContext(), hdrKernels);
    try {
        program.build(manager.getDevice(), "-cl-std=CL1.2");
    } catch (const cl::Error&) { throw std::runtime_error("Failed to build OpenCL HDR kernels: " + program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(manager.getDevice())); }
    return program;
}
using HdrKernelCache = cl_utils::DeviceProgramCache<buildHdrKernels>;
} // namespace

OclHdrTonemap::OclHdrTonemap(const int width, const int height, const cl_command_queue commandQueue) : queue(commandQueue, true), width(width), height(height) {
    InternalUtils::checkImageDimensions(height, width);
    if ((width & 1) || (height & 1))
        throw std::invalid_argument("OpenCL HDR YUV420 conversion requires even dimensions");
    auto& manager = OclQueueManager::getInstance();
    const auto context = manager.getContext();
    if (queue.getInfo<CL_QUEUE_CONTEXT>()() != context())
        throw std::invalid_argument("HDR conversion queue belongs to another OpenCL context");
    const size_t pixels = InternalUtils::checkedProduct(static_cast<size_t>(width), static_cast<size_t>(height));
    if (pixels > std::numeric_limits<int>::max() / 3)
        throw std::invalid_argument("OpenCL HDR frame is too large");
    output = cl::Buffer(context, CL_MEM_READ_WRITE, pixels * 3 / 2);
    // kernel interpolates from one gather
    const auto entryPairs = [&](const float* table, const size_t size) {
        std::vector<cl_float2> entries(size);
        for (size_t index = 0; index < size; ++index)
            entries[index] = {
                {table[index], table[std::min(index + 1, size - 1)]}
            };
        return cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, entries.size() * sizeof(cl_float2), entries.data());
    };
    pq = entryPairs(pqEotfLUT, std::size(pqEotfLUT));
    gamma = entryPairs(bt1886LUT, std::size(bt1886LUT));
    kernel = cl::Kernel(HdrKernelCache::getProgram(), "p010_hdr_to_sdr");
}

const cl::Buffer& OclHdrTonemap::convert(const cl::Buffer& source, const video_utils::MobiusParams& mobius, const video_utils::ChromaSiting& siting, const cl::Buffer& floatLuma) {
    return run(source, 6, mobius, siting, floatLuma);
}

const cl::Buffer& OclHdrTonemap::convertPlanar10(
    const uint16_t* const planes[3], const int pitches[3], const video_utils::MobiusParams& mobius, const video_utils::ChromaSiting& siting, const cl::Buffer& floatLuma) {
    const size_t pixels = static_cast<size_t>(width) * height;
    if (!uploaded())
        uploaded = cl::Buffer(OclQueueManager::getInstance().getContext(), CL_MEM_READ_ONLY, pixels * 3);
    for (int plane = 0; plane < 3; ++plane) {
        const size_t rowBytes = static_cast<size_t>(plane == 0 ? width : width / 2) * sizeof(uint16_t), rows = plane == 0 ? height : height / 2;
        const size_t offset = plane == 0 ? 0 : (pixels + (plane - 1) * pixels / 4) * sizeof(uint16_t);
        if (pitches[plane] < static_cast<int>(rowBytes))
            throw std::invalid_argument("OpenCL HDR upload plane pitch is too small");
        queue.enqueueWriteBufferRect(uploaded, CL_FALSE, {offset, 0, 0}, {0, 0, 0}, {rowBytes, rows, 1}, rowBytes, 0, static_cast<size_t>(pitches[plane]), 0, planes[plane]);
    }
    const auto& converted = run(uploaded, 0, mobius, siting, floatLuma);
    // host planes belong to decoder, writes must finish before reuse
    queue.finish();
    return converted;
}

const cl::Buffer& OclHdrTonemap::run(const cl::Buffer& source, const int sampleShift, const video_utils::MobiusParams& mobius, const video_utils::ChromaSiting& siting, const cl::Buffer& floatLuma) {
    const size_t pixels = static_cast<size_t>(width) * height;
    if (source.getInfo<CL_MEM_SIZE>() < pixels * 3 || (floatLuma() && floatLuma.getInfo<CL_MEM_SIZE>() < pixels * sizeof(float)))
        throw std::invalid_argument("OpenCL HDR conversion buffer is too small");
    cl_utils::setArgs(
        kernel, source, output, floatLuma, pq, gamma, width, height, mobius.a, mobius.b, mobius.k, static_cast<cl_int>(floatLuma() != nullptr), siting.x, siting.y, static_cast<cl_int>(sampleShift));
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(((width / 2 + 15) / 16) * 16, ((height / 2 + 7) / 8) * 8), cl::NDRange(16, 8));
    return output;
}
