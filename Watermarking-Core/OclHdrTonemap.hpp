#pragma once
#include "HdrTonemap.hpp"
#include "opencl_init.h"

// limited-range BT.2020 PQ planar P010 to limited-range BT.709 YUV420P
class OclHdrTonemap {
    cl::CommandQueue queue;
    cl::Buffer output, pq, gamma, uploaded;
    cl::Kernel kernel;
    int width, height;

    const cl::Buffer& run(const cl::Buffer& source, int sampleShift, const video_utils::MobiusParams& mobius, const video_utils::ChromaSiting& siting, const cl::Buffer& floatLuma);

  public:
    OclHdrTonemap(int width, int height, cl_command_queue queue);
    // planar P010 on GPU: Y, U then V, 10-bit samples in high bits
    const cl::Buffer& convert(const cl::Buffer& source, const video_utils::MobiusParams& mobius, const video_utils::ChromaSiting& siting, const cl::Buffer& floatLuma = {});
    // CPU-decoded planar 10-bit YUV420, uploaded first
    const cl::Buffer& convertPlanar10(
        const uint16_t* const planes[3], const int pitches[3], const video_utils::MobiusParams& mobius, const video_utils::ChromaSiting& siting, const cl::Buffer& floatLuma = {});
};
