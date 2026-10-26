#include "OclVideoInterop.hpp"
#include "CheckedSize.hpp"
#include "AvUtil.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <CL/cl_d3d11.h>
#include <cstdint>
#include <d3d11.h>
#include <d3d10.h>
#include <d3d11_4.h>
#include <d3dcompiler.h>
#include <deque>
#include <dxgi1_2.h>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <windows.h>
#include <wrl/client.h>

extern "C" {
#include "libavutil/buffer.h"
#include "libavutil/frame.h"
#include "libavutil/hwcontext.h"
#include "libavutil/hwcontext_d3d11va.h"
#include "libavutil/pixfmt.h"
}

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3dcompiler.lib")

using Microsoft::WRL::ComPtr;

namespace {
void checkHr(const HRESULT result, const char* operation) {
    if (FAILED(result))
        throw std::runtime_error(std::string(operation) + " failed (" + std::to_string(static_cast<unsigned long>(result)) + ")");
}

void checkCl(const cl_int result, const char* operation) {
    if (result != CL_SUCCESS)
        throw std::runtime_error(std::string(operation) + " failed (OpenCL " + std::to_string(result) + ")");
}

constexpr char copyShader[] = R"HLSL(
#if defined(WM_INPUT_YUY2) || defined(WM_INPUT_Y210) || defined(WM_INPUT_AYUV) || defined(WM_INPUT_Y410)
#define WM_PACKED_INPUT
Texture2D<uint4> packedInput : register(t0);
#else
Texture2D<uint> luma : register(t0);
Texture2D<uint2> chroma : register(t1);
#endif
RWByteAddressBuffer output : register(u0);
cbuffer Size : register(b0) { uint width; uint isP010; uint lumaSamples; uint totalSamples; uint chromaWidth; uint chromaHeight; uint unused0; uint unused1; };

uint toByte(uint sample) {
#ifdef WM_KEEP_P010
    return sample;
#else
    // P010 stores 10 significant bits in high bits of each 16-bit word
#ifdef WM_INPUT_Y410
    return min(255u, (sample + 2u) >> 2);
#else
    return isP010 ? min(255u, (sample + 128u) >> 8) : sample;
#endif
#endif
}

#ifdef WM_PACKED_INPUT
uint3 sourcePixel(uint x, uint y) {
#if defined(WM_INPUT_YUY2) || defined(WM_INPUT_Y210)
    uint4 sample = packedInput.Load(int3(x / 2, y, 0));
    return uint3((x & 1) ? sample.z : sample.x, sample.y, sample.w);
#elif defined(WM_INPUT_AYUV)
    return packedInput.Load(int3(x, y, 0)).zyx;
#else
    return packedInput.Load(int3(x, y, 0)).yxz;
#endif
}
#endif

uint sampleByte(uint index) {
#ifdef WM_PACKED_INPUT
    if (index < lumaSamples)
        return toByte(sourcePixel(index % width, index / width).x);
    uint planeSize = chromaWidth * chromaHeight;
    uint offset = index - lumaSamples, component = offset < planeSize ? 1 : 2;
    uint position = offset % planeSize;
    uint stepX = width / chromaWidth, stepY = (lumaSamples / width) / chromaHeight;
    uint x = (position % chromaWidth) * stepX, y = (position / chromaWidth) * stepY;
#if defined(WM_INPUT_YUY2) || defined(WM_INPUT_Y210)
    uint samplesX = 1; // two pixels of YUY2/Y210 pair share chroma
#else
    uint samplesX = stepX;
#endif
    uint sum = 0;
    for (uint dy = 0; dy < stepY; ++dy)
        for (uint dx = 0; dx < samplesX; ++dx)
            sum += sourcePixel(x + dx, y + dy)[component];
    uint count = samplesX * stepY;
    return toByte((sum + count / 2) / count);
#else
    if (index < lumaSamples)
        return toByte(luma.Load(int3(index % width, index / width, 0)));
    uint planeSize = lumaSamples / 4;
    uint offset = index - lumaSamples;
    uint planeIndex = offset % planeSize;
    uint2 uv = chroma.Load(int3(planeIndex % (width / 2), planeIndex / (width / 2), 0));
    return toByte(offset < planeSize ? uv.x : uv.y);
#endif
}

[numthreads(256, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
#ifdef WM_KEEP_P010
    const uint samplesPerWord = 2, bitsPerSample = 16;
#else
    const uint samplesPerWord = 4, bitsPerSample = 8;
#endif
    uint first = id.x * samplesPerWord;
    if (first >= totalSamples) return;
    uint packed = 0;
    [unroll] for (uint sample = 0; sample < samplesPerWord; ++sample)
        if (first + sample < totalSamples) packed |= sampleByte(first + sample) << (sample * bitsPerSample);
    output.Store(id.x * 4, packed);
}
)HLSL";

constexpr char outputShader[] = R"HLSL(
ByteAddressBuffer input : register(t0);
#if defined(WM_OUTPUT_YUY2) || defined(WM_OUTPUT_AYUV)
RWTexture2D<uint4> packed : register(u0);
#else
RWTexture2D<uint> luma : register(u0);
RWTexture2D<uint2> chroma : register(u1);
#endif
cbuffer Size : register(b0) { uint width; uint height; uint lumaSamples; uint chromaSamples; };
uint sampleByte(uint index) {
    return (input.Load(index & ~3u) >> ((index & 3u) * 8u)) & 255u;
}
[numthreads(16, 16, 1)]
void main(uint3 id : SV_DispatchThreadID) {
#if defined(WM_OUTPUT_YUY2)
    if (id.x < width / 2 && id.y < height) {
        uint y = id.y * width + id.x * 2;
        uint uv = id.y * (width / 2) + id.x;
        packed[id.xy] = uint4(sampleByte(y), sampleByte(lumaSamples + uv), sampleByte(y + 1), sampleByte(lumaSamples + chromaSamples + uv));
    }
#elif defined(WM_OUTPUT_AYUV)
    if (id.x < width && id.y < height) {
        uint pixel = id.y * width + id.x;
        packed[id.xy] = uint4(sampleByte(lumaSamples + chromaSamples + pixel), sampleByte(lumaSamples + pixel), sampleByte(pixel), 255);
    }
#else
    if (id.x < width && id.y < height)
        luma[id.xy] = sampleByte(id.y * width + id.x);
    if (id.x < width / 2 && id.y < height / 2) {
        uint offset = id.y * (width / 2) + id.x;
        chroma[id.xy] = uint2(sampleByte(lumaSamples + offset), sampleByte(lumaSamples + lumaSamples / 4 + offset));
    }
#endif
}
)HLSL";

// completion of D3D11 work submitted on immediate context
class VideoCompletion {
    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> immediate;
    ComPtr<ID3D11DeviceContext4> signalContext;
    ComPtr<ID3D11Fence> fence;
    HANDLE event = nullptr;
    // fences are free threaded, encode thread waits with its own event
    HANDLE encoderEvent = nullptr;
    UINT64 signaled = 0;
    // without fences, query event per signaled value
    std::deque<std::pair<UINT64, ComPtr<ID3D11Query>>> pending;
    std::vector<ComPtr<ID3D11Query>> idleQueries;

  public:
    ~VideoCompletion() {
        for (const HANDLE handle : {event, encoderEvent})
            if (handle)
                CloseHandle(handle);
    }
    void initialize(ID3D11Device* selectedDevice, ID3D11DeviceContext* context) {
        device = selectedDevice;
        immediate = context;
        ComPtr<ID3D11Device5> fenceDevice;
        if (SUCCEEDED(device.As(&fenceDevice)) && SUCCEEDED(immediate.As(&signalContext)) && SUCCEEDED(fenceDevice->CreateFence(0, D3D11_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)))) {
            event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
            encoderEvent = CreateEventW(nullptr, FALSE, FALSE, nullptr);
            if (!event || !encoderEvent)
                fence.Reset();
        }
    }
    // marks submitted work and starts it on GPU
    UINT64 signal() {
        const UINT64 value = ++signaled;
        if (fence) {
            checkHr(signalContext->Signal(fence.Get(), value), "Signal D3D11 video copy");
        } else {
            ComPtr<ID3D11Query> query;
            if (idleQueries.empty()) {
                D3D11_QUERY_DESC description{D3D11_QUERY_EVENT, 0};
                checkHr(device->CreateQuery(&description, &query), "Create video completion query");
            } else {
                query = std::move(idleQueries.back());
                idleQueries.pop_back();
            }
            immediate->End(query.Get());
            pending.emplace_back(value, std::move(query));
        }
        immediate->Flush();
        return value;
    }
    bool hasFence() const { return fence != nullptr; }
    void wait(const UINT64 value) {
        if (fence)
            return waitForFence(value, event);
        for (; !pending.empty() && pending.front().first <= value; pending.pop_front()) {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            HRESULT status = S_FALSE;
            while (status == S_FALSE && std::chrono::steady_clock::now() < deadline) {
                status = immediate->GetData(pending.front().second.Get(), nullptr, 0, D3D11_ASYNC_GETDATA_DONOTFLUSH);
                if (status == S_FALSE)
                    std::this_thread::yield();
            }
            if (status != S_OK)
                throw std::runtime_error("D3D11 video copy did not complete");
            idleQueries.push_back(std::move(pending.front().second));
        }
        checkHr(device->GetDeviceRemovedReason(), "D3D11 video device");
    }
    void finish() { wait(signal()); }
    // wait for signaled fence value on encode thread
    void waitOnEncoder(const UINT64 value) const { waitForFence(value, encoderEvent); }

  private:
    void waitForFence(const UINT64 value, const HANDLE completed) const {
        // removed device reports UINT64_MAX
        if (fence->GetCompletedValue() < value) {
            checkHr(fence->SetEventOnCompletion(value, completed), "Wait for D3D11 video copy");
            if (WaitForSingleObject(completed, 5000) != WAIT_OBJECT_0)
                throw std::runtime_error("D3D11 video copy timed out");
        }
        if (fence->GetCompletedValue() == UINT64_MAX)
            throw std::runtime_error("The D3D11 video device was removed");
        checkHr(device->GetDeviceRemovedReason(), "D3D11 video device");
    }
};

} // namespace

struct OclVideoInterop::Impl {
    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> immediate;
    cl::Context context;
    std::recursive_mutex mutex;
    clCreateFromD3D11BufferKHR_fn importBuffer = nullptr;
    clEnqueueAcquireD3D11ObjectsKHR_fn acquire = nullptr;
    clEnqueueReleaseD3D11ObjectsKHR_fn release = nullptr;

    cl::Buffer import(ID3D11Buffer* buffer, const cl_mem_flags flags) {
        cl_int status = CL_SUCCESS;
        const cl_mem memory = importBuffer(context(), flags, buffer, &status);
        checkCl(status, "Import D3D11 video buffer");
        return cl::Buffer(memory, false);
    }
};

OclVideoInterop::OclVideoInterop(std::shared_ptr<Impl> state) : impl(std::move(state)) {}

std::shared_ptr<OclVideoInterop> OclVideoInterop::create(const cl_platform_id platform, const cl_device_id selectedDevice) {
    size_t extensionBytes = 0;
    if (clGetDeviceInfo(selectedDevice, CL_DEVICE_EXTENSIONS, 0, nullptr, &extensionBytes) != CL_SUCCESS)
        return {};
    std::string extensions(extensionBytes, '\0');
    if (clGetDeviceInfo(selectedDevice, CL_DEVICE_EXTENSIONS, extensions.size(), extensions.data(), nullptr) != CL_SUCCESS)
        return {};
    extensions = " " + extensions.substr(0, extensionBytes - 1) + " ";
    // cl_nv_d3d11_sharing has same tokens and signatures as KHR extension
    const char* suffix = extensions.find(" cl_khr_d3d11_sharing ") != std::string::npos ? "KHR" : extensions.find(" cl_nv_d3d11_sharing ") != std::string::npos ? "NV" : nullptr;
    if (!suffix)
        return {};
    const auto entryPoint = [&](const char* name) { return clGetExtensionFunctionAddressForPlatform(platform, (std::string(name) + suffix).c_str()); };
    const auto getDevices = reinterpret_cast<clGetDeviceIDsFromD3D11KHR_fn>(entryPoint("clGetDeviceIDsFromD3D11"));
    auto state = std::make_shared<Impl>();
    state->importBuffer = reinterpret_cast<clCreateFromD3D11BufferKHR_fn>(entryPoint("clCreateFromD3D11Buffer"));
    state->acquire = reinterpret_cast<clEnqueueAcquireD3D11ObjectsKHR_fn>(entryPoint("clEnqueueAcquireD3D11Objects"));
    state->release = reinterpret_cast<clEnqueueReleaseD3D11ObjectsKHR_fn>(entryPoint("clEnqueueReleaseD3D11Objects"));
    if (!getDevices || !state->importBuffer || !state->acquire || !state->release)
        return {};
    ComPtr<IDXGIFactory1> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))))
        return {};
    for (UINT index = 0;; ++index) {
        ComPtr<IDXGIAdapter1> adapter;
        if (factory->EnumAdapters1(index, &adapter) != S_OK)
            return {};
        cl_uint count = 0;
        if (getDevices(platform, CL_D3D11_DXGI_ADAPTER_KHR, adapter.Get(), CL_PREFERRED_DEVICES_FOR_D3D11_KHR, 0, nullptr, &count) != CL_SUCCESS || count == 0)
            continue;
        std::vector<cl_device_id> matches(count);
        if (getDevices(platform, CL_D3D11_DXGI_ADAPTER_KHR, adapter.Get(), CL_PREFERRED_DEVICES_FOR_D3D11_KHR, count, matches.data(), nullptr) != CL_SUCCESS ||
            std::find(matches.begin(), matches.end(), selectedDevice) == matches.end())
            continue;
        constexpr D3D_FEATURE_LEVEL levels[]{D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0};
        if (FAILED(D3D11CreateDevice(adapter.Get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr, D3D11_CREATE_DEVICE_VIDEO_SUPPORT, levels, 2, D3D11_SDK_VERSION, &state->device, nullptr, &state->immediate)))
            return {};
        ComPtr<ID3D10Multithread> threading;
        if (SUCCEEDED(state->immediate.As(&threading)))
            threading->SetMultithreadProtected(TRUE);
        // transfers wait for D3D11 fence before acquire, user sync avoids waiting on unrelated D3D11 work
        cl_context_properties properties[]{CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(platform), CL_CONTEXT_D3D11_DEVICE_KHR,
            reinterpret_cast<cl_context_properties>(state->device.Get()), CL_CONTEXT_INTEROP_USER_SYNC, CL_TRUE, 0};
        cl_int status = CL_SUCCESS;
        cl_context context = clCreateContext(properties, 1, &selectedDevice, nullptr, nullptr, &status);
        if (status != CL_SUCCESS || !context) {
            properties[4] = 0; // driver may reject user sync property
            context = clCreateContext(properties, 1, &selectedDevice, nullptr, nullptr, &status);
        }
        if (status != CL_SUCCESS || !context)
            return {};
        state->context = cl::Context(context, false);
        return std::shared_ptr<OclVideoInterop>(new OclVideoInterop(std::move(state)));
    }
}

cl_context OclVideoInterop::context() const { return impl->context(); }

AVBufferRef* OclVideoInterop::createDevice() const {
    auto owner = std::make_unique<std::shared_ptr<Impl>>(impl);
    AVBufferRef* reference = av_hwdevice_ctx_alloc(AV_HWDEVICE_TYPE_D3D11VA);
    if (!reference)
        return nullptr;
    auto* hw = reinterpret_cast<AVHWDeviceContext*>(reference->data);
    hw->user_opaque = owner.release();
    hw->free = [](AVHWDeviceContext* context) { delete static_cast<std::shared_ptr<Impl>*>(context->user_opaque); };
    auto* d3d = static_cast<AVD3D11VADeviceContext*>(hw->hwctx);
    d3d->device = impl->device.Get();
    d3d->device->AddRef();
    d3d->lock_ctx = &impl->mutex;
    d3d->lock = [](void* mutex) { static_cast<std::recursive_mutex*>(mutex)->lock(); };
    d3d->unlock = [](void* mutex) { static_cast<std::recursive_mutex*>(mutex)->unlock(); };
    if (av_hwdevice_ctx_init(reference) < 0)
        av_buffer_unref(&reference);
    return reference;
}

struct OclVideoFrame::Impl {
    // packed copy of decoded frame imported into OpenCL
    struct Slot {
        ComPtr<ID3D11Buffer> buffer;
        ComPtr<ID3D11UnorderedAccessView> output;
        cl::Buffer shared;
        UINT64 packed = 0;
        bool acquired = false;
        // decoder surface packed from, frames must be consumed in staging order
        ID3D11Texture2D* source = nullptr;
        UINT sourceSlice = 0;
    };
    std::shared_ptr<OclVideoInterop::Impl> device;
    ComPtr<ID3D11Texture2D> texture;
    std::array<ComPtr<ID3D11ShaderResourceView>, 2> planes;
    ComPtr<ID3D11Buffer> constants;
    ComPtr<ID3D11ComputeShader> shader;
    DXGI_FORMAT sourceFormat = DXGI_FORMAT_UNKNOWN;
    VideoCompletion completion;
    cl::CommandQueue queue;
    std::vector<Slot> slots;
    // packed slots not yet returned by copyFrom, oldest first
    std::deque<size_t> staged;
    // slot returned by last copyFrom stays valid until next copyFrom
    size_t current = std::numeric_limits<size_t>::max();
    bool preserveP010 = false;
    AVPixelFormat planarFormat = AV_PIX_FMT_YUV420P;
    size_t lumaBytes = 0;
    int width = 0;
    int height = 0;
    size_t bytes = 0;

    ~Impl() {
        for (auto& slot : slots)
            if (slot.acquired) {
                const cl_mem memory = slot.shared();
                device->release(queue(), 1, &memory, 0, nullptr, nullptr);
            }
        if (queue())
            clFinish(queue());
    }

    // slot neither staged nor current, new one when lookahead holds all
    size_t freeSlot() {
        for (size_t index = 0; index < slots.size(); ++index)
            if (index != current && std::ranges::find(staged, index) == staged.end())
                return index;
        Slot slot;
        D3D11_BUFFER_DESC buffer{};
        buffer.ByteWidth = static_cast<UINT>((bytes + 3) & ~size_t(3));
        buffer.Usage = D3D11_USAGE_DEFAULT;
        buffer.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
        buffer.MiscFlags = D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
        checkHr(device->device->CreateBuffer(&buffer, nullptr, &slot.buffer), "Create shared video buffer");
        D3D11_UNORDERED_ACCESS_VIEW_DESC output{};
        output.Format = DXGI_FORMAT_R32_TYPELESS;
        output.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
        output.Buffer.NumElements = buffer.ByteWidth / 4;
        output.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_RAW;
        checkHr(device->device->CreateUnorderedAccessView(slot.buffer.Get(), &output, &slot.output), "Create video buffer view");
        slot.shared = device->import(slot.buffer.Get(), CL_MEM_READ_ONLY);
        slots.push_back(std::move(slot));
        return slots.size() - 1;
    }
};

OclVideoFrame::OclVideoFrame(std::unique_ptr<Impl> state) : impl(std::move(state)) {}
OclVideoFrame::~OclVideoFrame() = default;
AVPixelFormat OclVideoFrame::planarFormat() const { return impl->planarFormat; }

std::unique_ptr<OclVideoFrame> OclVideoInterop::createFrame(const AVFrame* source, const cl_command_queue queue, const bool preserveP010, const AVPixelFormat target) const {
    if (!source || source->format != AV_PIX_FMT_D3D11 || !source->data[0])
        throw std::invalid_argument("A D3D11 decoded frame is required for OpenCL interop");
    const int width = source->width, height = source->height;
    D3D11_TEXTURE2D_DESC sourceDescription{};
    reinterpret_cast<ID3D11Texture2D*>(source->data[0])->GetDesc(&sourceDescription);
    const bool isP010 = sourceDescription.Format == DXGI_FORMAT_P010;
    const bool packed422 = sourceDescription.Format == DXGI_FORMAT_YUY2 || sourceDescription.Format == DXGI_FORMAT_Y210;
    const bool packed444 = sourceDescription.Format == DXGI_FORMAT_AYUV || sourceDescription.Format == DXGI_FORMAT_Y410;
    if (!isP010 && sourceDescription.Format != DXGI_FORMAT_NV12 && !packed422 && !packed444)
        throw std::invalid_argument("D3D11 OpenCL interop does not support this texture format");
    const AVPixelFormat native = packed444 ? AV_PIX_FMT_YUV444P : packed422 ? AV_PIX_FMT_YUV422P : AV_PIX_FMT_YUV420P;
    const AVPixelFormat outputFormat = target == AV_PIX_FMT_NONE ? native : target;
    if (outputFormat != native && outputFormat != AV_PIX_FMT_YUV420P)
        throw std::invalid_argument("D3D11 copy can preserve chroma or convert it to 4:2:0");
    if (preserveP010 && !isP010)
        throw std::invalid_argument("OpenCL HDR interop requires P010 input");
    InternalUtils::checkImageDimensions(height, width);
    if ((outputFormat != AV_PIX_FMT_YUV444P && (width & 1)) || (outputFormat == AV_PIX_FMT_YUV420P && (height & 1)))
        throw std::invalid_argument("D3D11 copy dimensions must match its chroma layout");
    cl_context queueContext = nullptr;
    checkCl(clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, sizeof(queueContext), &queueContext, nullptr), "Query video copy queue");
    if (queueContext != impl->context())
        throw std::invalid_argument("Video copy queue belongs to another OpenCL context");
    auto frame = std::make_unique<OclVideoFrame::Impl>();
    frame->device = impl;
    frame->width = width;
    frame->height = height;
    frame->sourceFormat = sourceDescription.Format;
    frame->preserveP010 = preserveP010;
    const size_t pixels = InternalUtils::checkedProduct(static_cast<size_t>(width), static_cast<size_t>(height));
    frame->planarFormat = outputFormat;
    const int chromaWidth = outputFormat == AV_PIX_FMT_YUV444P ? width : width / 2;
    const int chromaHeight = outputFormat == AV_PIX_FMT_YUV420P ? height / 2 : height;
    const size_t samples = pixels + 2 * static_cast<size_t>(chromaWidth) * chromaHeight;
    frame->lumaBytes = pixels * (preserveP010 ? 2 : 1);
    frame->bytes = samples * (preserveP010 ? 2 : 1);
    if (frame->bytes > std::numeric_limits<UINT>::max() - 3 || (frame->bytes + 3) / 4 > 65535 * 256)
        throw std::invalid_argument("D3D11 video frame is too large");
    frame->queue = cl::CommandQueue(queue, true);
    const auto device = impl->device.Get();
    std::lock_guard lock(impl->mutex);
    D3D11_TEXTURE2D_DESC texture{};
    texture.Width = sourceDescription.Width;
    texture.Height = sourceDescription.Height;
    texture.MipLevels = texture.ArraySize = texture.SampleDesc.Count = 1;
    texture.Format = sourceDescription.Format;
    texture.Usage = D3D11_USAGE_DEFAULT;
    texture.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    checkHr(device->CreateTexture2D(&texture, nullptr, &frame->texture), "Create video copy texture");
    for (size_t plane = 0; plane < (packed422 || packed444 ? 1u : 2u); ++plane) {
        D3D11_SHADER_RESOURCE_VIEW_DESC view{};
        view.Format = sourceDescription.Format == DXGI_FORMAT_Y210   ? DXGI_FORMAT_R16G16B16A16_UINT
                      : sourceDescription.Format == DXGI_FORMAT_Y410 ? DXGI_FORMAT_R10G10B10A2_UINT
                      : packed422 || packed444                       ? DXGI_FORMAT_R8G8B8A8_UINT
                      : isP010                                       ? (plane == 0 ? DXGI_FORMAT_R16_UINT : DXGI_FORMAT_R16G16_UINT)
                                                                     : (plane == 0 ? DXGI_FORMAT_R8_UINT : DXGI_FORMAT_R8G8_UINT);
        view.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
        view.Texture2D.MipLevels = 1;
        checkHr(device->CreateShaderResourceView(frame->texture.Get(), &view, &frame->planes[plane]), "Create video plane view");
    }
    // first slot created up front to validate buffer sharing during decoder probe
    frame->freeSlot();
    const std::array<UINT, 8> sizes{static_cast<UINT>(width), static_cast<UINT>(isP010 || sourceDescription.Format == DXGI_FORMAT_Y210), static_cast<UINT>(width * height), static_cast<UINT>(samples),
        static_cast<UINT>(chromaWidth), static_cast<UINT>(chromaHeight), 0, 0};
    D3D11_SUBRESOURCE_DATA initial{sizes.data(), 0, 0};
    D3D11_BUFFER_DESC buffer{};
    buffer.ByteWidth = sizeof(sizes);
    buffer.Usage = D3D11_USAGE_IMMUTABLE;
    buffer.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    checkHr(device->CreateBuffer(&buffer, &initial, &frame->constants), "Create video copy constants");
    ComPtr<ID3DBlob> code;
    ComPtr<ID3DBlob> errors;
    std::vector<D3D_SHADER_MACRO> defines;
    if (preserveP010)
        defines.push_back({"WM_KEEP_P010", "1"});
    if (sourceDescription.Format == DXGI_FORMAT_YUY2)
        defines.push_back({"WM_INPUT_YUY2", "1"});
    if (sourceDescription.Format == DXGI_FORMAT_Y210)
        defines.push_back({"WM_INPUT_Y210", "1"});
    if (sourceDescription.Format == DXGI_FORMAT_AYUV)
        defines.push_back({"WM_INPUT_AYUV", "1"});
    if (sourceDescription.Format == DXGI_FORMAT_Y410)
        defines.push_back({"WM_INPUT_Y410", "1"});
    defines.push_back({nullptr, nullptr});
    const HRESULT compiled = D3DCompile(copyShader, sizeof(copyShader) - 1, nullptr, defines.data(), nullptr, "main", "cs_5_0", D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &code, &errors);
    if (FAILED(compiled))
        throw std::runtime_error("Compile video copy shader: " + (errors ? std::string(static_cast<const char*>(errors->GetBufferPointer()), errors->GetBufferSize()) : "unknown error"));
    checkHr(device->CreateComputeShader(code->GetBufferPointer(), code->GetBufferSize(), nullptr, &frame->shader), "Create video copy shader");
    frame->completion.initialize(device, impl->immediate.Get());
    return std::unique_ptr<OclVideoFrame>(new OclVideoFrame(std::move(frame)));
}

void OclVideoFrame::stage(const AVFrame* frame, const bool lumaOnly) {
    if (frame->format != AV_PIX_FMT_D3D11 || frame->width != impl->width || frame->height != impl->height || !frame->data[0] || frame->crop_left != 0 || frame->crop_top != 0)
        throw std::runtime_error("Unexpected D3D11 decoded frame layout");
    if (!impl->preserveP010 && (frame->color_trc == AVCOL_TRC_SMPTE2084 || frame->color_trc == AVCOL_TRC_ARIB_STD_B67))
        throw std::runtime_error("D3D11 OpenCL interop requires SDR input");
    auto* source = reinterpret_cast<ID3D11Texture2D*>(frame->data[0]);
    D3D11_TEXTURE2D_DESC description{};
    source->GetDesc(&description);
    const auto slice = reinterpret_cast<uintptr_t>(frame->data[1]);
    ComPtr<ID3D11Device> sourceDevice;
    source->GetDevice(&sourceDevice);
    if (description.Format != impl->sourceFormat || description.Width < static_cast<UINT>(impl->width) || description.Height < static_cast<UINT>(impl->height) || slice >= description.ArraySize ||
        sourceDevice.Get() != impl->device->device.Get())
        throw std::runtime_error("D3D11 decoder surface does not match the selected OpenCL device or dimensions");
    std::lock_guard lock(impl->device->mutex);
    const size_t index = impl->freeSlot();
    auto& slot = impl->slots[index];
    if (slot.acquired) {
        const cl_mem shared = slot.shared();
        checkCl(impl->device->release(impl->queue(), 1, &shared, 0, nullptr, nullptr), "Release video buffer");
        slot.acquired = false;
        checkCl(clFinish(impl->queue()), "Finish reading video buffer before D3D11 reuse");
    }
    const auto immediate = impl->device->immediate.Get();
    const D3D11_BOX region{0, 0, 0, static_cast<UINT>(impl->width), static_cast<UINT>(impl->height), 1};
    // decoder reference surface stays read-only, processing uses our texture and buffer
    immediate->CopySubresourceRegion(impl->texture.Get(), 0, 0, 0, 0, source, D3D11CalcSubresource(0, static_cast<UINT>(slice), description.MipLevels), &region);
    std::array<ID3D11ShaderResourceView*, 2> planes{impl->planes[0].Get(), impl->planes[1].Get()};
    ID3D11UnorderedAccessView* output = slot.output.Get();
    ID3D11Buffer* constants = impl->constants.Get();
    immediate->CSSetShader(impl->shader.Get(), nullptr, 0);
    immediate->CSSetShaderResources(0, 2, planes.data());
    immediate->CSSetUnorderedAccessViews(0, 1, &output, nullptr);
    immediate->CSSetConstantBuffers(0, 1, &constants);
    // luma packed first, each thread group packs 1024 bytes
    const size_t packedBytes = lumaOnly ? impl->lumaBytes : impl->bytes;
    immediate->Dispatch(static_cast<UINT>((packedBytes + 1023) / 1024), 1, 1);
    planes.fill(nullptr);
    output = nullptr;
    constants = nullptr;
    immediate->CSSetShaderResources(0, 2, planes.data());
    immediate->CSSetUnorderedAccessViews(0, 1, &output, nullptr);
    immediate->CSSetConstantBuffers(0, 1, &constants);
    immediate->CSSetShader(nullptr, nullptr, 0);
    slot.packed = impl->completion.signal();
    slot.source = source;
    slot.sourceSlice = static_cast<UINT>(slice);
    impl->staged.push_back(index);
}

const cl::Buffer& OclVideoFrame::copyFrom(const AVFrame* frame) {
    if (impl->staged.empty())
        stage(frame);
    auto& slot = impl->slots[impl->staged.front()];
    if (frame->format != AV_PIX_FMT_D3D11 || slot.source != reinterpret_cast<ID3D11Texture2D*>(frame->data[0]) || slot.sourceSlice != reinterpret_cast<uintptr_t>(frame->data[1]))
        throw std::logic_error("D3D11 video frames must be copied in the order they were staged");
    impl->current = impl->staged.front();
    impl->staged.pop_front();
    impl->completion.wait(slot.packed);
    std::lock_guard lock(impl->device->mutex);
    const cl_mem shared = slot.shared();
    checkCl(impl->device->acquire(impl->queue(), 1, &shared, 0, nullptr, nullptr), "Acquire video buffer");
    slot.acquired = true;
    return slot.shared;
}

struct OclVideoOutput::Impl {
    std::shared_ptr<OclVideoInterop::Impl> device;
    video_utils::AVBufferRefPtr frames;
    cl::CommandQueue queue;
    ComPtr<ID3D11Buffer> sharedBuffer;
    cl::Buffer shared;
    ComPtr<ID3D11ShaderResourceView> input;
    ComPtr<ID3D11Texture2D> texture;
    std::array<ComPtr<ID3D11UnorderedAccessView>, 2> planes;
    ComPtr<ID3D11Buffer> constants;
    ComPtr<ID3D11ComputeShader> shader;
    VideoCompletion completion;
    // encoder surfaces copied directly from decoder surfaces in staging order
    struct Staged {
        video_utils::AVFramePtr frame;
        ID3D11Texture2D* source = nullptr;
        UINT sourceSlice = 0;
        UINT64 copied = 0;
    };
    std::deque<Staged> staged;
    int width = 0, height = 0;
    int chromaWidth = 0, chromaHeight = 0;
    AVPixelFormat planarFormat = AV_PIX_FMT_YUV420P;
    DXGI_FORMAT textureFormat = DXGI_FORMAT_NV12;
    size_t pixels = 0, bytes = 0;
    bool acquired = false;
    // completes after D3D11 reads shared buffer, OpenCL writes again only then
    UINT64 sharedRead = 0;

    ~Impl() {
        if (acquired) {
            const cl_mem memory = shared();
            device->release(queue(), 1, &memory, 0, nullptr, nullptr);
        }
        if (queue())
            clFinish(queue());
    }

    void acquireBuffer() {
        const cl_mem memory = shared();
        if (acquired) {
            checkCl(device->release(queue(), 1, &memory, 0, nullptr, nullptr), "Release previous encoder buffer");
            acquired = false;
            checkCl(clFinish(queue()), "Finish previous encoder writes");
        }
        completion.wait(sharedRead);
        checkCl(device->acquire(queue(), 1, &memory, 0, nullptr, nullptr), "Acquire encoder buffer");
        acquired = true;
    }

    video_utils::AVFramePtr allocate(video_utils::AVFramePtr& mapped) const {
        video_utils::AVFramePtr frame(av_frame_alloc());
        if (!frame)
            throw std::bad_alloc();
        video_utils::checkAv(av_hwframe_get_buffer(frames.get(), frame.get(), 0), "Allocate hardware encoder frame");
        if (frame->format == AV_PIX_FMT_QSV) {
            mapped.reset(av_frame_alloc());
            if (!mapped)
                throw std::bad_alloc();
            mapped->format = AV_PIX_FMT_D3D11;
            video_utils::checkAv(av_hwframe_map(mapped.get(), frame.get(), AV_HWFRAME_MAP_WRITE | AV_HWFRAME_MAP_OVERWRITE | AV_HWFRAME_MAP_DIRECT), "Map encoder frame to D3D11");
        }
        return frame;
    }

    void copyTexture(const AVFrame* destination, ID3D11Texture2D* source, UINT sourceSlice) {
        auto* target = reinterpret_cast<ID3D11Texture2D*>(destination->data[0]);
        D3D11_TEXTURE2D_DESC targetDescription{}, sourceDescription{};
        target->GetDesc(&targetDescription);
        source->GetDesc(&sourceDescription);
        ComPtr<ID3D11Device> sourceDevice, targetDevice;
        source->GetDevice(&sourceDevice);
        target->GetDevice(&targetDevice);
        if (sourceDescription.Format != textureFormat || targetDescription.Format != textureFormat || sourceDescription.Width < static_cast<UINT>(width) ||
            sourceDescription.Height < static_cast<UINT>(height) || targetDescription.Width < static_cast<UINT>(width) || targetDescription.Height < static_cast<UINT>(height) ||
            sourceSlice >= sourceDescription.ArraySize || sourceDevice.Get() != device->device.Get() || targetDevice.Get() != device->device.Get())
            throw std::invalid_argument("Encoder copy requires matching textures on the selected GPU");
        const UINT targetSlice = static_cast<UINT>(reinterpret_cast<uintptr_t>(destination->data[1]));
        if (targetSlice >= targetDescription.ArraySize)
            throw std::invalid_argument("Invalid encoder texture slice");
        const D3D11_BOX region{0, 0, 0, static_cast<UINT>(width), static_cast<UINT>(height), 1};
        device->immediate->CopySubresourceRegion(
            target, D3D11CalcSubresource(0, targetSlice, targetDescription.MipLevels), 0, 0, 0, source, D3D11CalcSubresource(0, sourceSlice, sourceDescription.MipLevels), &region);
    }

    AVFrame* finish(const cl::Buffer& lumaOverride) {
        if (lumaOverride()) {
            if (lumaOverride.getInfo<CL_MEM_SIZE>() < pixels)
                throw std::invalid_argument("Encoder luma buffer is too small");
            queue.enqueueCopyBuffer(lumaOverride, shared, 0, 0, pixels);
        }
        const cl_mem memory = shared();
        checkCl(device->release(queue(), 1, &memory, 0, nullptr, nullptr), "Release OpenCL encoder buffer");
        acquired = false;
        checkCl(clFinish(queue()), "Finish OpenCL encoder writes");
        auto* immediate = device->immediate.Get();
        ID3D11ShaderResourceView* source = input.Get();
        std::array<ID3D11UnorderedAccessView*, 2> targets{planes[0].Get(), planes[1].Get()};
        ID3D11Buffer* sizes = constants.Get();
        immediate->CSSetShader(shader.Get(), nullptr, 0);
        immediate->CSSetShaderResources(0, 1, &source);
        immediate->CSSetUnorderedAccessViews(0, 2, targets.data(), nullptr);
        immediate->CSSetConstantBuffers(0, 1, &sizes);
        immediate->Dispatch((width + 15) / 16, (height + 15) / 16, 1);
        source = nullptr;
        targets.fill(nullptr);
        sizes = nullptr;
        immediate->CSSetShaderResources(0, 1, &source);
        immediate->CSSetUnorderedAccessViews(0, 2, targets.data(), nullptr);
        immediate->CSSetConstantBuffers(0, 1, &sizes);
        immediate->CSSetShader(nullptr, nullptr, 0);
        video_utils::AVFramePtr mapped;
        auto frame = allocate(mapped);
        copyTexture(mapped ? mapped.get() : frame.get(), texture.Get(), 0);
        sharedRead = completion.signal();
        return handOver(std::move(frame), sharedRead);
    }

    // with fence encode thread waits for copy, otherwise completes here
    AVFrame* handOver(video_utils::AVFramePtr frame, const UINT64 copied) {
        if (completion.hasFence())
            frame->opaque = reinterpret_cast<void*>(static_cast<uintptr_t>(copied));
        else
            completion.wait(copied);
        return frame.release();
    }
};

OclVideoOutput::OclVideoOutput(std::unique_ptr<Impl> state) : impl(std::move(state)) {}
OclVideoOutput::~OclVideoOutput() = default;

std::unique_ptr<OclVideoOutput> OclVideoInterop::createOutput(AVBufferRef* encoderFrames, const cl_command_queue queue) const {
    if (!encoderFrames)
        throw std::invalid_argument("Hardware encoder frame pool is missing");
    const auto* frames = reinterpret_cast<const AVHWFramesContext*>(encoderFrames->data);
    if ((frames->format != AV_PIX_FMT_D3D11 && frames->format != AV_PIX_FMT_QSV) ||
        (frames->sw_format != AV_PIX_FMT_NV12 && frames->sw_format != AV_PIX_FMT_YUYV422 && frames->sw_format != AV_PIX_FMT_VUYX))
        throw std::invalid_argument("OpenCL hardware encoding requires NV12, YUY2 or AYUV D3D11/QSV surfaces");
    InternalUtils::checkImageDimensions(frames->height, frames->width);
    if ((frames->sw_format != AV_PIX_FMT_VUYX && (frames->width & 1)) || (frames->sw_format == AV_PIX_FMT_NV12 && (frames->height & 1)))
        throw std::invalid_argument("Encoder dimensions must match the surface chroma subsampling");
    cl_context context = nullptr;
    checkCl(clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, sizeof(context), &context, nullptr), "Query encoder queue");
    if (context != impl->context())
        throw std::invalid_argument("Encoder queue belongs to another OpenCL context");
    auto output = std::make_unique<OclVideoOutput::Impl>();
    output->device = impl;
    output->frames.reset(av_buffer_ref(encoderFrames));
    if (!output->frames)
        throw std::bad_alloc();
    output->queue = cl::CommandQueue(queue, true);
    output->width = frames->width;
    output->height = frames->height;
    output->planarFormat = frames->sw_format == AV_PIX_FMT_VUYX ? AV_PIX_FMT_YUV444P : frames->sw_format == AV_PIX_FMT_YUYV422 ? AV_PIX_FMT_YUV422P : AV_PIX_FMT_YUV420P;
    output->textureFormat = frames->sw_format == AV_PIX_FMT_VUYX ? DXGI_FORMAT_AYUV : frames->sw_format == AV_PIX_FMT_YUYV422 ? DXGI_FORMAT_YUY2 : DXGI_FORMAT_NV12;
    output->chromaWidth = frames->sw_format == AV_PIX_FMT_VUYX ? frames->width : frames->width / 2;
    output->chromaHeight = frames->sw_format == AV_PIX_FMT_NV12 ? frames->height / 2 : frames->height;
    output->pixels = InternalUtils::checkedProduct(static_cast<size_t>(frames->width), static_cast<size_t>(frames->height));
    const size_t chromaSamples = static_cast<size_t>(output->chromaWidth) * output->chromaHeight;
    output->bytes = output->pixels + 2 * chromaSamples;
    if (output->bytes > std::numeric_limits<UINT>::max() - 3)
        throw std::invalid_argument("Encoder frame is too large");
    std::lock_guard lock(impl->mutex);
    const auto device = impl->device.Get();
    D3D11_BUFFER_DESC buffer{};
    buffer.ByteWidth = static_cast<UINT>((output->bytes + 3) & ~size_t(3));
    buffer.Usage = D3D11_USAGE_DEFAULT;
    buffer.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    buffer.MiscFlags = D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
    checkHr(device->CreateBuffer(&buffer, nullptr, &output->sharedBuffer), "Create shared encoder buffer");
    D3D11_SHADER_RESOURCE_VIEW_DESC input{};
    input.Format = DXGI_FORMAT_R32_TYPELESS;
    input.ViewDimension = D3D11_SRV_DIMENSION_BUFFEREX;
    input.BufferEx.NumElements = buffer.ByteWidth / 4;
    input.BufferEx.Flags = D3D11_BUFFEREX_SRV_FLAG_RAW;
    checkHr(device->CreateShaderResourceView(output->sharedBuffer.Get(), &input, &output->input), "Create encoder buffer view");
    output->shared = impl->import(output->sharedBuffer.Get(), CL_MEM_WRITE_ONLY);
    D3D11_TEXTURE2D_DESC texture{};
    texture.Width = frames->width;
    texture.Height = frames->height;
    texture.ArraySize = texture.MipLevels = texture.SampleDesc.Count = 1;
    texture.Format = output->textureFormat;
    texture.Usage = D3D11_USAGE_DEFAULT;
    texture.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
    checkHr(device->CreateTexture2D(&texture, nullptr, &output->texture), "Create encoder output texture");
    for (size_t plane = 0; plane < (frames->sw_format == AV_PIX_FMT_NV12 ? 2u : 1u); ++plane) {
        D3D11_UNORDERED_ACCESS_VIEW_DESC view{};
        view.Format = frames->sw_format == AV_PIX_FMT_NV12 ? (plane == 0 ? DXGI_FORMAT_R8_UINT : DXGI_FORMAT_R8G8_UINT) : DXGI_FORMAT_R8G8B8A8_UINT;
        view.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE2D;
        checkHr(device->CreateUnorderedAccessView(output->texture.Get(), &view, &output->planes[plane]), "Create encoder plane view");
    }
    const std::array<UINT, 4> sizes{static_cast<UINT>(frames->width), static_cast<UINT>(frames->height), static_cast<UINT>(output->pixels), static_cast<UINT>(chromaSamples)};
    D3D11_SUBRESOURCE_DATA data{sizes.data(), 0, 0};
    buffer = {};
    buffer.ByteWidth = sizeof(sizes);
    buffer.Usage = D3D11_USAGE_IMMUTABLE;
    buffer.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    checkHr(device->CreateBuffer(&buffer, &data, &output->constants), "Create encoder constants");
    ComPtr<ID3DBlob> code, errors;
    const D3D_SHADER_MACRO defines[]{
        {frames->sw_format == AV_PIX_FMT_VUYX ? "WM_OUTPUT_AYUV" : "WM_OUTPUT_YUY2", "1"    },
        {nullptr,                                                                    nullptr}
    };
    const auto compiled = D3DCompile(
        outputShader, sizeof(outputShader) - 1, nullptr, frames->sw_format == AV_PIX_FMT_NV12 ? nullptr : defines, nullptr, "main", "cs_5_0", D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &code, &errors);
    if (FAILED(compiled))
        throw std::runtime_error("Compile encoder shader: " + (errors ? std::string(static_cast<const char*>(errors->GetBufferPointer()), errors->GetBufferSize()) : "unknown error"));
    checkHr(device->CreateComputeShader(code->GetBufferPointer(), code->GetBufferSize(), nullptr, &output->shader), "Create encoder shader");
    output->completion.initialize(device, impl->immediate.Get());
    // validate allocation and mapping before accepting format
    video_utils::AVFramePtr mapped;
    auto probe = output->allocate(mapped);
    // OpenCL writes shared buffer first, queued resource creation must not follow writes
    output->completion.finish();
    return std::unique_ptr<OclVideoOutput>(new OclVideoOutput(std::move(output)));
}

AVFrame* OclVideoOutput::copyFrom(const cl::Buffer& planarYuv, const cl::Buffer& lumaOverride) {
    if (planarYuv.getInfo<CL_MEM_SIZE>() < impl->bytes)
        throw std::invalid_argument("Encoder YUV buffer is too small");
    std::lock_guard lock(impl->device->mutex);
    impl->acquireBuffer();
    const size_t offset = lumaOverride() ? impl->pixels : 0;
    impl->queue.enqueueCopyBuffer(planarYuv, impl->shared, offset, offset, impl->bytes - offset);
    return impl->finish(lumaOverride);
}

void OclVideoOutput::waitUntilCopied(const AVFrame* frame) const {
    if (frame->opaque)
        impl->completion.waitOnEncoder(reinterpret_cast<uintptr_t>(frame->opaque));
}

void OclVideoOutput::stage(const AVFrame* source) {
    if (!source || source->format != AV_PIX_FMT_D3D11 || source->width != impl->width || source->height != impl->height || source->crop_left || source->crop_top || !source->data[0])
        throw std::invalid_argument("Unexpected direct D3D11 encoder input");
    auto* texture = reinterpret_cast<ID3D11Texture2D*>(source->data[0]);
    const auto slice = static_cast<UINT>(reinterpret_cast<uintptr_t>(source->data[1]));
    std::lock_guard lock(impl->device->mutex);
    video_utils::AVFramePtr mapped;
    auto frame = impl->allocate(mapped);
    impl->copyTexture(mapped ? mapped.get() : frame.get(), texture, slice);
    impl->staged.push_back({std::move(frame), texture, slice, impl->completion.signal()});
}

AVFrame* OclVideoOutput::copyFrom(const AVFrame* source, const cl::Buffer& lumaOverride) {
    if (!source || source->width != impl->width || source->height != impl->height)
        throw std::invalid_argument("Unexpected encoder input dimensions");
    if (source->format == AV_PIX_FMT_D3D11) {
        if (lumaOverride())
            throw std::invalid_argument("Unexpected direct D3D11 encoder input");
        if (impl->staged.empty())
            stage(source);
        auto staged = std::move(impl->staged.front());
        impl->staged.pop_front();
        if (staged.source != reinterpret_cast<ID3D11Texture2D*>(source->data[0]) || staged.sourceSlice != reinterpret_cast<uintptr_t>(source->data[1]))
            throw std::logic_error("Direct encoder frames must be returned in the order they were staged");
        return impl->handOver(std::move(staged.frame), staged.copied);
    }
    std::lock_guard lock(impl->device->mutex);
    const auto planar = source->format == AV_PIX_FMT_YUVJ420P   ? AV_PIX_FMT_YUV420P
                        : source->format == AV_PIX_FMT_YUVJ422P ? AV_PIX_FMT_YUV422P
                        : source->format == AV_PIX_FMT_YUVJ444P ? AV_PIX_FMT_YUV444P
                                                                : static_cast<AVPixelFormat>(source->format);
    if (planar != impl->planarFormat)
        throw std::invalid_argument("CPU encoder upload requires the negotiated 8-bit planar YUV format");
    const cl_mem memory = impl->shared();
    impl->acquireBuffer();
    for (int plane = lumaOverride() ? 1 : 0; plane < 3; ++plane) {
        const size_t width = plane == 0 ? impl->width : impl->chromaWidth;
        const size_t height = plane == 0 ? impl->height : impl->chromaHeight;
        const size_t offset = plane == 0 ? 0 : impl->pixels + (plane - 1) * static_cast<size_t>(impl->chromaWidth) * impl->chromaHeight;
        if (!source->data[plane] || source->linesize[plane] < static_cast<int>(width))
            throw std::invalid_argument("Unexpected CPU encoder plane layout");
        // plane offset is a multiple of row pitch, including small even frame sizes
        const size_t destination[3]{0, offset / width, 0}, origin[3]{0, 0, 0}, region[3]{width, height, 1};
        checkCl(
            clEnqueueWriteBufferRect(impl->queue(), memory, CL_TRUE, destination, origin, region, width, 0, static_cast<size_t>(source->linesize[plane]), 0, source->data[plane], 0, nullptr, nullptr),
            "Upload encoder chroma/luma plane");
    }
    return impl->finish(lumaOverride);
}
