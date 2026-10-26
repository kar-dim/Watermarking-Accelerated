#pragma once

#include "opencl_init.h"
extern "C" {
#include "libavutil/pixfmt.h"
}
#include <memory>

struct AVBufferRef;
struct AVFrame;
class OclVideoFrame;
class OclVideoOutput;

class OclVideoInterop {
    struct Impl;
    std::shared_ptr<Impl> impl;
    explicit OclVideoInterop(std::shared_ptr<Impl> state);
    friend class OclVideoFrame;
    friend class OclVideoOutput;

  public:
    static std::shared_ptr<OclVideoInterop> create(cl_platform_id platform, cl_device_id device);
    cl_context context() const;
    AVBufferRef* createDevice() const;
    // preserveP010 packs 16-bit samples for HDR conversion instead of reducing to 8-bit
    std::unique_ptr<OclVideoFrame> createFrame(const AVFrame* frame, cl_command_queue queue, bool preserveP010 = false, AVPixelFormat target = AV_PIX_FMT_NONE) const;
    // negotiated NV12 (420), YUY2 (422) or AYUV (444) encoder surfaces
    std::unique_ptr<OclVideoOutput> createOutput(AVBufferRef* encoderFrames, cl_command_queue queue) const;
};

class OclVideoOutput {
    struct Impl;
    std::unique_ptr<Impl> impl;
    explicit OclVideoOutput(std::unique_ptr<Impl> state);
    friend class OclVideoInterop;

  public:
    ~OclVideoOutput();
    // starts copying decoded D3D11 frame into new encoder surface, copyFrom returns surface
    void stage(const AVFrame* source);
    // returned frame owns encoder surface until released
    AVFrame* copyFrom(const cl::Buffer& planarYuv, const cl::Buffer& lumaOverride = {});
    AVFrame* copyFrom(const AVFrame* source, const cl::Buffer& lumaOverride = {});
    void waitUntilCopied(const AVFrame* frame) const;
};

class OclVideoFrame {
    struct Impl;
    std::unique_ptr<Impl> impl;
    explicit OclVideoFrame(std::unique_ptr<Impl> state);
    friend class OclVideoInterop;

  public:
    ~OclVideoFrame();
    AVPixelFormat planarFormat() const;
    // starts packing decoded frame on GPU, copyFrom waits for it
    void stage(const AVFrame* frame, bool lumaOnly = false);
    // staged frames returned in staging order, valid until next copyFrom or destruction
    const cl::Buffer& copyFrom(const AVFrame* frame);
};
