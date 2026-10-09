#include "CudaArray.hpp"
#include "CudaCheck.hpp"
#include "nvjpeg_utils.hpp"
#include "utils.hpp"
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <memory>
#include <mutex>
#include <nvjpeg.h>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

/*!
 *  \brief  Implementation of GPU-accelerated JPEG encoding and decoding using nvJPEG
 *  \author Dimitris Karatzas
 */

namespace {
template <auto Destroy>
struct Destroyer {
    template <typename T>
    void operator()(T* object) const {
        Destroy(object);
    }
};
// nvJPEG objects, released with their nvjpeg Destroy function
template <typename T, auto Destroy>
using Owned = std::unique_ptr<T, Destroyer<Destroy>>;
using Handle = Owned<nvjpegHandle, nvjpegDestroy>;
using JpegState = Owned<nvjpegJpegState, nvjpegJpegStateDestroy>;
using JpegStream = Owned<nvjpegJpegStream, nvjpegJpegStreamDestroy>;
using JpegDecoder = Owned<nvjpegJpegDecoder, nvjpegDecoderDestroy>;
using DecodeParams = Owned<nvjpegDecodeParams, nvjpegDecodeParamsDestroy>;
using PinnedBuffer = Owned<nvjpegBufferPinned, nvjpegBufferPinnedDestroy>;
using DeviceBuffer = Owned<nvjpegBufferDevice, nvjpegBufferDeviceDestroy>;
using EncoderState = Owned<nvjpegEncoderState, nvjpegEncoderStateDestroy>;
using EncoderParams = Owned<nvjpegEncoderParams, nvjpegEncoderParamsDestroy>;

// interpolated chroma upsampling, like libjpeg's (fancy upsampling)
constexpr unsigned int kHandleFlags = NVJPEG_FLAGS_UPSAMPLING_WITH_INTERPOLATION;
// codec object sets per device, decoders are on the critical path (batch prefetch, UI): one per image the batch loads at once.
// ONE encoder is enough, more encoders would require more VRAM for no gains
constexpr int kMaxEncoders = 1;

bool succeeded(const nvjpegStatus_t status) { return status == NVJPEG_STATUS_SUCCESS; }

template <typename Owner, typename Create>
bool create(Owner& owner, Create&& createObject) {
    typename Owner::pointer object = nullptr;
    const bool created = succeeded(createObject(&object));
    owner.reset(created ? object : nullptr);
    return created;
}

// matches CImg layout
nvjpegImage_t planarImage(uint8_t* planes, const int rows, const int cols, const int channels) {
    nvjpegImage_t image{};
    for (int channel = 0; channel < channels; channel++) {
        image.channel[channel] = planes + static_cast<size_t>(channel) * rows * cols;
        image.pitch[channel] = cols;
    }
    return image;
}

// the decode objects of one caller: nvJPEG states, streams and buffers can't be shared by concurrent decodes
struct DecodeWorker {
    JpegStream jpegStream;
    DecodeParams decodeParams;
    // GPU assisted Huffman decoding (decoupled API)
    JpegDecoder decoder;
    PinnedBuffer pinnedBuffer;
    DeviceBuffer deviceBuffer;
    JpegState decoderState;
    // hardware decoding (batched API, batch of 1), initialized for one output format at a time
    JpegState batchedState;
    std::optional<nvjpegOutputFormat_t> batchedFormat;

    static std::unique_ptr<DecodeWorker> create(nvjpegHandle_t handle) {
        auto worker = std::make_unique<DecodeWorker>();
        const bool created = ::create(worker->jpegStream, [&](auto* stream) { return nvjpegJpegStreamCreate(handle, stream); }) &&
                             ::create(worker->decodeParams, [&](auto* params) { return nvjpegDecodeParamsCreate(handle, params); }) &&
                             ::create(worker->decoder, [&](auto* decoder) { return nvjpegDecoderCreate(handle, NVJPEG_BACKEND_GPU_HYBRID, decoder); }) &&
                             ::create(worker->pinnedBuffer, [&](auto* buffer) { return nvjpegBufferPinnedCreate(handle, nullptr, buffer); }) &&
                             ::create(worker->deviceBuffer, [&](auto* buffer) { return nvjpegBufferDeviceCreate(handle, nullptr, buffer); }) &&
                             ::create(worker->decoderState, [&](auto* state) { return nvjpegDecoderStateCreate(handle, worker->decoder.get(), state); }) &&
                             succeeded(nvjpegStateAttachPinnedBuffer(worker->decoderState.get(), worker->pinnedBuffer.get())) &&
                             succeeded(nvjpegStateAttachDeviceBuffer(worker->decoderState.get(), worker->deviceBuffer.get()));
        return created ? std::move(worker) : nullptr;
    }

    // hardware decoders take baseline single scan streams only
    bool decodeHardware(nvjpegHandle_t handle, const uint8_t* jpeg, const size_t length, const nvjpegOutputFormat_t format, nvjpegImage_t& image, cudaStream_t stream) {
        int unsupported = 1;
        if (!succeeded(nvjpegDecodeBatchedSupported(handle, jpegStream.get(), &unsupported)) || unsupported != 0)
            return false;
        if (!batchedState && !::create(batchedState, [&](auto* state) { return nvjpegJpegStateCreate(handle, state); }))
            return false;
        if (batchedFormat != format) {
            if (!succeeded(nvjpegDecodeBatchedInitialize(handle, batchedState.get(), 1, 1, format)))
                return false;
            batchedFormat = format;
        }
        const unsigned char* bitstreams[] = {jpeg};
        const size_t lengths[] = {length};
        return succeeded(nvjpegDecodeBatched(handle, batchedState.get(), bitstreams, lengths, &image, stream));
    }

    // the host stage (parsing, Huffman preparation) runs on the calling thread, the Huffman decoding, IDCT and color conversion on the GPU
    bool decodeGpu(nvjpegHandle_t handle, const nvjpegOutputFormat_t format, nvjpegImage_t& image, cudaStream_t stream) const {
        int unsupported = 1;
        return succeeded(nvjpegDecodeParamsSetOutputFormat(decodeParams.get(), format)) && succeeded(nvjpegDecoderJpegSupported(decoder.get(), jpegStream.get(), decodeParams.get(), &unsupported)) &&
               unsupported == 0 && succeeded(nvjpegDecodeJpegHost(handle, decoder.get(), decoderState.get(), decodeParams.get(), jpegStream.get())) &&
               succeeded(nvjpegDecodeJpegTransferToDevice(handle, decoder.get(), decoderState.get(), jpegStream.get(), stream)) &&
               succeeded(nvjpegDecodeJpegDevice(handle, decoder.get(), decoderState.get(), &image, stream));
    }
};

// the encode objects of one caller
struct EncodeWorker {
    EncoderState state;
    EncoderParams params;

    static std::unique_ptr<EncodeWorker> create(nvjpegHandle_t handle, cudaStream_t stream) {
        auto worker = std::make_unique<EncodeWorker>();
        const auto createState = [&](const nvjpegEncBackend_t backend) {
            return ::create(worker->state, [&](auto* state) { return nvjpegEncoderStateCreateWithBackend(handle, state, backend, stream); });
        };
        // the hardware encoder when the device has one, CImg's save_jpeg settings: quality 100, no optimized Huffman tables
        const bool created = (createState(NVJPEG_ENC_BACKEND_HARDWARE) || createState(NVJPEG_ENC_BACKEND_GPU)) &&
                             ::create(worker->params, [&](auto* params) { return nvjpegEncoderParamsCreate(handle, params, stream); }) &&
                             succeeded(nvjpegEncoderParamsSetQuality(worker->params.get(), 100, stream)) && succeeded(nvjpegEncoderParamsSetOptimizedHuffman(worker->params.get(), 0, stream));
        return created ? std::move(worker) : nullptr;
    }
};

// up to "maxWorkers" workers, created on demand and reused
template <typename Worker>
class WorkerPool {
  public:
    explicit WorkerPool(const int maxWorkers) : maxWorkers(maxWorkers) {}

    // gives the worker back on destruction, empty when the worker could not be created
    class Lease {
      public:
        Lease(WorkerPool& pool, std::unique_ptr<Worker> worker) : pool(pool), worker(std::move(worker)) {}
        ~Lease() { pool.release(std::move(worker)); }
        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;
        Worker* operator->() const { return worker.get(); }
        explicit operator bool() const { return worker != nullptr; }

      private:
        WorkerPool& pool;
        std::unique_ptr<Worker> worker;
    };

    template <typename... Args>
    Lease acquire(Args&&... args) {
        std::unique_lock lock(mutex);
        available.wait(lock, [&] { return !idle.empty() || created < maxWorkers; });
        if (!idle.empty()) {
            auto worker = std::move(idle.back());
            idle.pop_back();
            return Lease(*this, std::move(worker));
        }
        created++;
        lock.unlock();
        return Lease(*this, Worker::create(std::forward<Args>(args)...));
    }

  private:
    const int maxWorkers;
    std::mutex mutex;
    std::condition_variable available;
    std::vector<std::unique_ptr<Worker>> idle;
    int created = 0;

    void release(std::unique_ptr<Worker> worker) {
        {
            std::lock_guard lock(mutex);
            if (worker)
                idle.push_back(std::move(worker));
            else
                created--;
        }
        available.notify_one();
    }
};

struct DeviceCodec {
    Handle handle;
    bool hardwareDecoder = false;
    WorkerPool<DecodeWorker> decoders{InternalUtils::batchLoadThreads()};
    WorkerPool<EncodeWorker> encoders{kMaxEncoders};
};

// the nvJPEG objects of each device, destroyed with their device selected
class Codecs {
  public:
    // null when nvJPEG is not usable on the current device
    static DeviceCodec* current() {
        static Codecs instance;
        return instance.currentDevice();
    }

    Codecs(const Codecs&) = delete;
    Codecs& operator=(const Codecs&) = delete;

  private:
    std::mutex mutex;
    std::unordered_map<int, std::unique_ptr<DeviceCodec>> devices;

    Codecs() = default;
    ~Codecs() {
        for (auto& [device, codec] : devices) {
            cudaSetDevice(device);
            codec.reset();
        }
    }

    static std::unique_ptr<DeviceCodec> probe() {
        auto codec = std::make_unique<DeviceCodec>();
        const auto createHandle = [&](const nvjpegBackend_t backend) { return create(codec->handle, [&](auto* handle) { return nvjpegCreateEx(backend, nullptr, nullptr, kHandleFlags, handle); }); };
        // hardware handle fails at creation (ARCH_MISMATCH) on devices without a JPEG engine
        codec->hardwareDecoder = createHandle(NVJPEG_BACKEND_HARDWARE);
        if (!codec->hardwareDecoder && !createHandle(NVJPEG_BACKEND_DEFAULT))
            return nullptr;
        return codec;
    }

    DeviceCodec* currentDevice() {
        int device = 0;
        if (cudaGetDevice(&device) != cudaSuccess)
            return nullptr;
        std::lock_guard lock(mutex);
        auto [entry, inserted] = devices.try_emplace(device);
        if (inserted)
            entry->second = probe();
        return entry->second.get();
    }
};
} // namespace

std::optional<CudaArray<uint8_t>> nvjpeg_utils::decode(const uint8_t* jpeg, const size_t length, cudaStream_t stream) {
    DeviceCodec* codec = Codecs::current();
    if (!codec)
        return std::nullopt;
    auto worker = codec->decoders.acquire(codec->handle.get());
    if (!worker)
        return std::nullopt;
    nvjpegHandle_t handle = codec->handle.get();
    nvjpegJpegStream_t jpegStream = worker->jpegStream.get();
    unsigned int components = 0, width = 0, height = 0, precision = 0;
    if (!succeeded(nvjpegJpegStreamParse(handle, jpeg, length, 0, 0, jpegStream)) || !succeeded(nvjpegJpegStreamGetComponentsNum(jpegStream, &components)) ||
        !succeeded(nvjpegJpegStreamGetFrameDimensions(jpegStream, &width, &height)) || !succeeded(nvjpegJpegStreamGetSamplePrecision(jpegStream, &precision)))
        return std::nullopt;
    // not supported, those will fallback to CImg
    if ((components != 1 && components != 3) || precision != 8)
        return std::nullopt;
    InternalUtils::checkImageDimensions(height, width);
    const int rows = static_cast<int>(height);
    const int cols = static_cast<int>(width);
    const int channels = static_cast<int>(components);
    const nvjpegOutputFormat_t format = channels == 3 ? NVJPEG_OUTPUT_RGB : NVJPEG_OUTPUT_Y;
    CudaArray<uint8_t> planes(rows, cols, channels, stream);
    nvjpegImage_t image = planarImage(planes.data(), rows, cols, channels);
    const bool decoded = (codec->hardwareDecoder && worker->decodeHardware(handle, jpeg, length, format, image, stream)) || worker->decodeGpu(handle, format, image, stream);
    // the next decode with this worker rewrites its pinned buffer from the host: the transfer must be done before the worker is released
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (!decoded)
        return std::nullopt;
    return planes;
}

std::vector<uint8_t> nvjpeg_utils::encode(const CudaArray<uint8_t>& planes, cudaStream_t stream) {
    const int channels = planes.getChannels();
    DeviceCodec* codec = Codecs::current();
    if (!codec || (channels != 1 && channels != 3))
        return {};
    auto worker = codec->encoders.acquire(codec->handle.get(), stream);
    if (!worker)
        return {};
    nvjpegHandle_t handle = codec->handle.get();
    const int rows = planes.getRows();
    const int cols = planes.getCols();
    // nvJPEG does not write to the source planes
    const nvjpegImage_t image = planarImage(const_cast<uint8_t*>(planes.data()), rows, cols, channels);
    // libjpeg's defaults (CImg): 4:2:0 color, grayscale with a single component
    const bool encoded = channels == 3 ? succeeded(nvjpegEncoderParamsSetSamplingFactors(worker->params.get(), NVJPEG_CSS_420, stream)) &&
                                             succeeded(nvjpegEncodeImage(handle, worker->state.get(), worker->params.get(), &image, NVJPEG_INPUT_RGB, cols, rows, stream))
                                       : succeeded(nvjpegEncoderParamsSetSamplingFactors(worker->params.get(), NVJPEG_CSS_GRAY, stream)) &&
                                             succeeded(nvjpegEncodeYUV(handle, worker->state.get(), worker->params.get(), &image, NVJPEG_CSS_GRAY, cols, rows, stream));
    size_t length = 0;
    if (!encoded || !succeeded(nvjpegEncodeRetrieveBitstream(handle, worker->state.get(), nullptr, &length, stream)))
        return {};
    std::vector<uint8_t> jpeg(length);
    if (!succeeded(nvjpegEncodeRetrieveBitstream(handle, worker->state.get(), jpeg.data(), &length, stream)))
        return {};
    CUDA_CHECK(cudaStreamSynchronize(stream));
    jpeg.resize(length);
    return jpeg;
}
