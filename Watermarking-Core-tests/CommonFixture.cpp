#include "../Watermarking-Core/AuxiliaryMux.hpp"
#include "../Watermarking-Core/AvUtil.hpp"
#include "../Watermarking-Core/CheckedSize.hpp"
#include "../Watermarking-Core/EncodeOptions.hpp"
#include "../Watermarking-Core/half_float.hpp"
#include "../Watermarking-Core/OutputFile.hpp"
#include "../Watermarking-Core/WatermarkCrypto.hpp"
#if defined(_USE_OPENCL_) || defined(_USE_CUDA_)
#include "../Watermarking-Core/HdrTonemapCurves.hpp"
#include "../Watermarking-Core/video_utils.hpp"
#include "../Watermarking-Core/VideoProcessingContext.hpp"
#endif
#if defined(_USE_OPENCL_)
#include "../Watermarking-Core/kernels/half_kernels.hpp"
#include "../Watermarking-Core/OclArray.hpp"
#include "../Watermarking-Core/OclVideoDevice.hpp"
#include "../Watermarking-Core/opencl_utils.hpp"
#pragma push_macro("small")
#include <d3d11.h>
extern "C" {
#include "libavutil/hwcontext_d3d11va.h"
}
#pragma pop_macro("small")
#include <cstdlib>
#include <optional>
#elif defined(_USE_CUDA_)
#include "../Watermarking-Core/cuda_utils.hpp"
#include "../Watermarking-Core/CudaArray.hpp"
#include "../Watermarking-Core/CudaStreamManager.hpp"
#include "../Watermarking-Core/nvjpeg_utils.hpp"
#endif
#include "WatermarkCore.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
#include <filesystem>
#include <format>
#include <fstream>
#include <future>
#include <gtest/gtest.h>
#include <iomanip>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>
// needs <cstdio> first (FILE)
#include <jpeglib.h>
#include <png.h>

#if defined(_USE_EIGEN_)
#include <omp.h>
#endif

extern "C" {
#include "libavfilter/buffersink.h"
#include "libavutil/common.h"
#include "libavutil/dict.h"
#include "libavutil/hwcontext.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"
}

/*!
 *  \brief  Comprehensive unit, integration, and regression tests for Watermarking-Core
 *  \author Dimitris Karatzas
 */

using namespace WatermarkCore;
namespace fs = std::filesystem;

namespace {
constexpr int defaultP = 3;
constexpr float defaultPsnr = 40.0f;
constexpr const char* defaultPassword = "random_watermark_password";
const fs::path colorImage = "samples/images/512.png";
const fs::path grayImage = "samples/images/512_gray.jpg";
const fs::path alphaImage = "samples/images/4k_argb.png";
// 720p.png stored rotated 90 degrees counter-clockwise with EXIF orientation 6
const fs::path exifImage = "samples/images/720p_exif6.jpg";

// Render digests in the form used by the published SHA-256 test vectors
std::string hexDigest(const std::array<uint8_t, 32>& digest) {
    std::ostringstream result;
    result << std::hex << std::setfill('0');
    for (const uint8_t byte : digest)
        result << std::setw(2) << static_cast<unsigned int>(byte);
    return result.str();
}

// every 24-bit random value once as x1 and once as x2
std::vector<uint32_t> everyRandomPair() {
    constexpr uint32_t count = 1u << 24;
    std::vector<uint32_t> pairs(2 * static_cast<size_t>(count));
    for (uint32_t i = 0; i < count; ++i) {
        pairs[2 * static_cast<size_t>(i)] = i;
        pairs[(2 * static_cast<size_t>(i)) + 1] = ((i * 0x9E3779B1u) + 0x2545F4u) & 0xFFFFFFu;
    }
    return pairs;
}

// the reference transform of the CPU on every pair
std::vector<float> referenceBoxMuller(const std::vector<uint32_t>& randomPairs) {
    std::vector<float> normals(randomPairs.size());
    for (size_t i = 0; i < randomPairs.size(); i += 2)
        std::tie(normals[i], normals[i + 1]) = WatermarkCrypto::generateBoxMullerNormalPair(static_cast<uint64_t>(randomPairs[i]) << 40, static_cast<uint64_t>(randomPairs[i + 1]) << 40);
    return normals;
}

// the scalar reference watermark: ChaCha20 block by block, the reference transform, half rounding
std::vector<uint16_t> referenceWatermark(const std::array<uint32_t, 16>& baseState, const int64_t numElements) {
    std::vector<uint16_t> halfBits(static_cast<size_t>(numElements));
    for (int64_t block = 0; block * 8 < numElements; ++block) {
        const std::array<uint64_t, 8> randomBits = WatermarkCrypto::chacha20Block(baseState, static_cast<uint64_t>(block));
        for (int j = 0; j < 8 && (block * 8) + j < numElements; ++j) {
            const auto [z0, z1] = WatermarkCrypto::generateBoxMullerNormalPair(randomBits[j & ~1], randomBits[j | 1]);
            halfBits[static_cast<size_t>((block * 8) + j)] = HalfFloat::fromFloat((j & 1) ? z1 : z0);
        }
    }
    return halfBits;
}

// empty when "actual" and "expected" have the same bits
template <typename T>
std::string bitDifferences(const std::vector<T>& actual, const std::vector<T>& expected) {
    using Bits = std::conditional_t<sizeof(T) == 4, uint32_t, uint16_t>;
    if (actual.size() != expected.size())
        return std::format("size {} instead of {}", actual.size(), expected.size());
    size_t differences = 0;
    std::string first;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (std::bit_cast<Bits>(actual[i]) == std::bit_cast<Bits>(expected[i]))
            continue;
        if (differences++ == 0)
            first = std::format("first at {}: {:#x} instead of {:#x}", i, std::bit_cast<Bits>(actual[i]), std::bit_cast<Bits>(expected[i]));
    }
    return differences == 0 ? std::string() : std::format("{} of {} values differ, {}", differences, actual.size(), first);
}

#if defined(_USE_CUDA_) || defined(_USE_OPENCL_)
// the Box-Muller transform and the watermark of the device generation kernels (the selected GPU)
#if defined(_USE_CUDA_)
std::vector<float> deviceBoxMuller(const std::vector<uint32_t>& randomPairs) {
    const cudaStream_t stream = CudaStreamManager::getInstance().getComputeStream();
    const int count = static_cast<int>(randomPairs.size());
    const CudaArray<uint32_t> input(count, 1, randomPairs.data(), stream);
    CudaArray<float> normals(count, stream);
    cuda_utils::launchBoxMullerKernel(input.data(), normals.data(), count / 2, stream);
    std::vector<float> result(randomPairs.size());
    normals.toHost(result.data());
    return result;
}

std::vector<uint16_t> deviceWatermark(const std::array<uint32_t, 16>& baseState, const int64_t numElements) {
    const cudaStream_t stream = CudaStreamManager::getInstance().getComputeStream();
    CudaArray<__half> watermark(static_cast<int>(numElements), stream);
    cuda_utils::launchGenerateWatermarkKernel(baseState, watermark.data(), numElements, stream);
    std::vector<uint16_t> halfBits(static_cast<size_t>(numElements));
    watermark.toHost(reinterpret_cast<__half*>(halfBits.data()));
    return halfBits;
}
#else
std::vector<float> deviceBoxMuller(const std::vector<uint32_t>& randomPairs) {
    auto& queueManager = OclQueueManager::getInstance();
    const int count = static_cast<int>(randomPairs.size());
    const OclArray<uint32_t> input(count, 1, randomPairs.data(), queueManager.getQueueRaw());
    const OclArray<float> normals(count, queueManager.getQueueRaw());
    cl_utils::launchBoxMullerKernel(input.clBuffer(), normals.clBuffer(), count / 2, queueManager.getQueue());
    std::vector<float> result(randomPairs.size());
    normals.toHost(result.data());
    return result;
}

std::vector<uint16_t> deviceWatermark(const std::array<uint32_t, 16>& baseState, const int64_t numElements) {
    auto& queueManager = OclQueueManager::getInstance();
    const OclArray<cl_half> watermark(static_cast<int>(numElements), queueManager.getQueueRaw());
    cl_utils::launchGenerateWatermarkKernel(baseState, watermark.clBuffer(), numElements, queueManager.getQueue());
    std::vector<uint16_t> halfBits(static_cast<size_t>(numElements));
    watermark.toHost(halfBits.data());
    return halfBits;
}
#endif

// runs "check" (with the device name) on the GPUs: CUDA the first device, OpenCL every device
template <typename Check>
void forEachDevice(Check&& check) {
#if defined(_USE_CUDA_)
    ASSERT_TRUE(initializeEnvironment(0));
    check(getDeviceName());
#else
    const std::vector<std::string> devices = getAvailableDevices();
    for (size_t deviceIndex = 0; deviceIndex < devices.size(); ++deviceIndex) {
        ASSERT_TRUE(initializeEnvironment(static_cast<int>(deviceIndex))) << devices[deviceIndex];
        check(devices[deviceIndex]);
    }
    ASSERT_TRUE(initializeEnvironment(0));
#endif
}
#endif

// Wait for backend work before copying the embedded pixels to the host
SessionPixelData embedAndRead(ImageSession* session) {
    embedImage(session);
    finish();
    return getSessionPixelData(session);
}

// writes a 24-bit BMP filled with deterministic noise
void writeNoiseBmp(const fs::path& path, const int width, const int height) {
    const int rowBytes = ((width * 3) + 3) & ~3;
    const uint32_t pixelBytes = static_cast<uint32_t>(rowBytes) * height;
    std::vector<uint8_t> bmp(54 + pixelBytes);
    const auto put32 = [&bmp](const size_t offset, const uint32_t value) {
        for (int i = 0; i < 4; ++i)
            bmp[offset + i] = static_cast<uint8_t>(value >> (8 * i));
    };
    bmp[0] = 'B';
    bmp[1] = 'M';
    put32(2, static_cast<uint32_t>(bmp.size()));
    put32(10, 54);
    put32(14, 40);
    put32(18, static_cast<uint32_t>(width));
    put32(22, static_cast<uint32_t>(height));
    bmp[26] = 1;
    bmp[28] = 24;
    put32(34, pixelBytes);
    uint32_t state = 12345;
    for (size_t i = 54; i < bmp.size(); ++i) {
        state = (state * 1664525u) + 1013904223u;
        bmp[i] = static_cast<uint8_t>(state >> 24);
    }
    std::ofstream output(path, std::ios::binary);
    ASSERT_TRUE(output.write(reinterpret_cast<const char*>(bmp.data()), static_cast<std::streamsize>(bmp.size())));
}

// writes an 8-bit binary PGM (one channel) filled with deterministic noise
void writeNoisePgm(const fs::path& path, const int width, const int height) {
    std::string pgm = std::format("P5\n{} {}\n255\n", width, height);
    uint32_t state = 54321;
    for (int i = 0; i < width * height; ++i) {
        state = (state * 1664525u) + 1013904223u;
        pgm.push_back(static_cast<char>(state >> 24));
    }
    std::ofstream output(path, std::ios::binary);
    ASSERT_TRUE(output.write(pgm.data(), static_cast<std::streamsize>(pgm.size())));
}

// the stored pixel (as in the file) of the displayed pixel (x, y) for an EXIF orientation 1-8 (the same table as PIL's ImageOps.exif_transpose)
std::pair<int, int> orientedSource(const int x, const int y, const int storedWidth, const int storedHeight, const int orientation) {
    switch (orientation) {
    case 2: return {storedWidth - 1 - x, y};
    case 3: return {storedWidth - 1 - x, storedHeight - 1 - y};
    case 4: return {x, storedHeight - 1 - y};
    case 5: return {y, x};
    case 6: return {y, storedHeight - 1 - x};
    case 7: return {storedWidth - 1 - y, storedHeight - 1 - x};
    case 8: return {storedWidth - 1 - y, x};
    default: return {x, y};
    }
}

// writes displayed pixels as quality 100 4:4:4 JPEG with EXIF orientation tag
void writeOrientedJpeg(const fs::path& path, const std::vector<uint8_t>& displayed, const int width, const int height, const int channels, const int orientation) {
    const bool swapAxes = orientation >= 5;
    const int storedWidth = swapAxes ? height : width;
    const int storedHeight = swapAxes ? width : height;
    std::vector<uint8_t> stored(displayed.size());
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x) {
            const auto [storedX, storedY] = orientedSource(x, y, storedWidth, storedHeight, orientation);
            for (int channel = 0; channel < channels; ++channel)
                stored[(static_cast<size_t>(storedY) * storedWidth + storedX) * channels + channel] = displayed[(static_cast<size_t>(y) * width + x) * channels + channel];
        }
    // APP1: "Exif\0\0", little endian TIFF header, IFD0 with one entry (0x0112 orientation, SHORT)
    const uint8_t exif[] = {'E', 'x', 'i', 'f', 0, 0, 'I', 'I', 0x2A, 0, 8, 0, 0, 0, 1, 0, 0x12, 0x01, 3, 0, 1, 0, 0, 0, static_cast<uint8_t>(orientation), 0, 0, 0, 0, 0, 0, 0};
    std::vector<uint8_t> jpeg(stored.size() * 2 + 65536);
    unsigned char* buffer = jpeg.data();
    unsigned long size = static_cast<unsigned long>(jpeg.size());
    jpeg_compress_struct compressor{};
    jpeg_error_mgr errors{};
    compressor.err = jpeg_std_error(&errors);
    jpeg_create_compress(&compressor);
    jpeg_mem_dest(&compressor, &buffer, &size);
    compressor.image_width = static_cast<JDIMENSION>(storedWidth);
    compressor.image_height = static_cast<JDIMENSION>(storedHeight);
    compressor.input_components = channels;
    compressor.in_color_space = channels == 3 ? JCS_RGB : JCS_GRAYSCALE;
    jpeg_set_defaults(&compressor);
    jpeg_set_quality(&compressor, 100, TRUE);
    // no chroma subsampling: block of one color decodes to the same pixels wherever it is (only its DC coefficient, no upsampling)
    for (int component = 0; component < compressor.num_components; ++component)
        compressor.comp_info[component].h_samp_factor = compressor.comp_info[component].v_samp_factor = 1;
    compressor.write_JFIF_header = FALSE;
    jpeg_start_compress(&compressor, TRUE);
    jpeg_write_marker(&compressor, JPEG_APP0 + 1, exif, sizeof(exif));
    for (int row = 0; row < storedHeight; ++row) {
        JSAMPROW rowPointer = stored.data() + static_cast<size_t>(row) * storedWidth * channels;
        jpeg_write_scanlines(&compressor, &rowPointer, 1);
    }
    jpeg_finish_compress(&compressor);
    jpeg_destroy_compress(&compressor);
    // the buffer was large enough, libjpeg did not replace it
    ASSERT_EQ(buffer, jpeg.data());
    std::ofstream output(path, std::ios::binary);
    ASSERT_TRUE(output.write(reinterpret_cast<const char*>(jpeg.data()), static_cast<std::streamsize>(size)));
}

// Read the whole file for codec comparisons and destination-preservation checks
std::vector<uint8_t> readBytes(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

// gradients plus deterministic noise, row-major interleaved
std::vector<uint8_t> jpegTestPattern(const int width, const int height, const int components) {
    std::vector<uint8_t> pixels(static_cast<size_t>(width) * height * components);
    uint32_t state = 4242;
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            for (int component = 0; component < components; ++component) {
                state = (state * 1664525u) + 1013904223u;
                pixels[(static_cast<size_t>(y) * width + x) * components + component] = static_cast<uint8_t>((x * 3) + (y * 2) + (component * 60) + (state >> 27));
            }
    return pixels;
}

// writes pixels (row-major interleaved, 1, 3 or 4 components) as quality 90 JPEG
void writeJpeg(const fs::path& path, const std::vector<uint8_t>& pixels, const int width, const int height, const int components, const int sampling, const bool progressive) {
    std::vector<uint8_t> jpeg(pixels.size() * 2 + 65536);
    unsigned char* buffer = jpeg.data();
    unsigned long size = static_cast<unsigned long>(jpeg.size());
    jpeg_compress_struct compressor{};
    jpeg_error_mgr errors{};
    compressor.err = jpeg_std_error(&errors);
    jpeg_create_compress(&compressor);
    jpeg_mem_dest(&compressor, &buffer, &size);
    compressor.image_width = static_cast<JDIMENSION>(width);
    compressor.image_height = static_cast<JDIMENSION>(height);
    compressor.input_components = components;
    compressor.in_color_space = components == 4 ? JCS_CMYK : components == 3 ? JCS_RGB : JCS_GRAYSCALE;
    jpeg_set_defaults(&compressor);
    jpeg_set_quality(&compressor, 90, TRUE);
    compressor.comp_info[0].h_samp_factor = compressor.comp_info[0].v_samp_factor = sampling;
    for (int component = 1; component < compressor.num_components; ++component)
        compressor.comp_info[component].h_samp_factor = compressor.comp_info[component].v_samp_factor = 1;
    if (progressive)
        jpeg_simple_progression(&compressor);
    jpeg_start_compress(&compressor, TRUE);
    for (int row = 0; row < height; ++row) {
        JSAMPROW rowPointer = const_cast<uint8_t*>(pixels.data()) + static_cast<size_t>(row) * width * components;
        jpeg_write_scanlines(&compressor, &rowPointer, 1);
    }
    jpeg_finish_compress(&compressor);
    jpeg_destroy_compress(&compressor);
    // the buffer was large enough, libjpeg did not replace it
    ASSERT_EQ(buffer, jpeg.data());
    std::ofstream output(path, std::ios::binary);
    ASSERT_TRUE(output.write(reinterpret_cast<const char*>(jpeg.data()), static_cast<std::streamsize>(size)));
}

struct DecodedJpeg {
    std::vector<uint8_t> pixels; // row-major interleaved
    int width = 0;
    int height = 0;
    int components = 0;
    // sampling factor of the first component (2 for 4:2:0)
    int lumaSampling = 0;
};

// libjpeg's decode with its defaults (as CImg decodes)
DecodedJpeg decodeJpeg(const fs::path& path) {
    const std::vector<uint8_t> file = readBytes(path);
    jpeg_decompress_struct decompressor{};
    jpeg_error_mgr errors{};
    decompressor.err = jpeg_std_error(&errors);
    jpeg_create_decompress(&decompressor);
    jpeg_mem_src(&decompressor, file.data(), static_cast<unsigned long>(file.size()));
    jpeg_read_header(&decompressor, TRUE);
    DecodedJpeg decoded;
    decoded.lumaSampling = decompressor.comp_info[0].h_samp_factor;
    jpeg_start_decompress(&decompressor);
    decoded.width = static_cast<int>(decompressor.output_width);
    decoded.height = static_cast<int>(decompressor.output_height);
    decoded.components = decompressor.output_components;
    decoded.pixels.resize(static_cast<size_t>(decoded.width) * decoded.height * decoded.components);
    while (decompressor.output_scanline < decompressor.output_height) {
        JSAMPROW rowPointer = decoded.pixels.data() + static_cast<size_t>(decompressor.output_scanline) * decoded.width * decoded.components;
        jpeg_read_scanlines(&decompressor, &rowPointer, 1);
    }
    jpeg_finish_decompress(&decompressor);
    jpeg_destroy_decompress(&decompressor);
    return decoded;
}

#if defined(_USE_CUDA_)
// true when nvJPEG decodes the file (no CImg fallback)
bool decodesWithNvJpeg(const fs::path& path) {
    const std::vector<uint8_t> file = readBytes(path);
    return nvjpeg_utils::decode(file.data(), file.size(), CudaStreamManager::getInstance().getComputeStream()).has_value();
}
#endif

// look up one key in a parsed option dictionary, empty when absent
std::string dictValue(const video_utils::ParsedEncodeOptions& parsed, const char* key) {
    const AVDictionaryEntry* entry = av_dict_get(parsed.dictionary.get(), key, nullptr, 0);
    return entry != nullptr ? entry->value : "";
}

// baseline settings for the video tests
VideoSettings makeVideoSettings(const std::string& input) {
    VideoSettings settings{};
    settings.videoFile = input;
    settings.watermarkPassword = defaultPassword;
    settings.p = defaultP;
    settings.psnr = 30.0f; // lower on purpose, to preserve watermark a bit more when re-encoding
    settings.watermarkInterval = 1;
    settings.useHwDecoder = false;
    settings.useHwEncoder = false;
    settings.encodeOptions = "-c:v libx265 -preset ultrafast -crf 20";
    return settings;
}

// textured fixture sample on 8-bit scale, 10-bit fixtures store times 4 for exact 8-bit reduction
int fixtureSample(const int plane, const int row, const int col, const int index) { return 64 + ((col * 3 + row * 2 + index * 7 + plane * 29) % 128); }

// HDR fixtures use one uniform frame, SDR regression clip uses 100 textured frames
void writeVideoFixture(const fs::path& path, const bool deep, const AVColorTransferCharacteristic transfer, const bool textured = false, const char* encoderName = nullptr,
    const AVPixelFormat sourceFormat = AV_PIX_FMT_NONE, const AVColorRange colorRange = AVCOL_RANGE_MPEG, const std::initializer_list<std::pair<const char*, const char*>> encoderOverrides = {},
    const bool coloredChroma = false, const std::pair<int, int> texturedSize = {320, 180}) {
    using namespace video_utils;
    AVFormatContext* rawOutput = nullptr;
    checkAv(avformat_alloc_output_context2(&rawOutput, nullptr, "matroska", path.string().c_str()), "Allocate fixture muxer");
    AVOutputFormatContextPtr output(rawOutput);
    const AVCodec* codec = encoderName ? avcodec_find_encoder_by_name(encoderName) : avcodec_find_encoder(AV_CODEC_ID_FFV1);
    AVCodecContextPtr encoder(avcodec_alloc_context3(codec));
    if (!encoder)
        throw std::runtime_error("Allocate fixture encoder");
    encoder->width = textured ? texturedSize.first : 16;
    encoder->height = textured ? texturedSize.second : 16;
    encoder->time_base = {1, 30};
    encoder->framerate = {30, 1};
    encoder->pix_fmt = sourceFormat != AV_PIX_FMT_NONE ? sourceFormat : (deep ? AV_PIX_FMT_YUV420P10LE : AV_PIX_FMT_YUV420P);
    encoder->color_range = colorRange;
    encoder->color_primaries = transfer == AVCOL_TRC_BT709 ? AVCOL_PRI_BT709 : AVCOL_PRI_BT2020;
    encoder->color_trc = transfer;
    encoder->colorspace = transfer == AVCOL_TRC_BT709 ? AVCOL_SPC_BT709 : AVCOL_SPC_BT2020_NCL;
    if (output->oformat->flags & AVFMT_GLOBALHEADER)
        encoder->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    AVDictionary* encoderOptions = nullptr;
    if (encoderName && std::string_view(encoderName) == "libsvtav1") {
        av_dict_set(&encoderOptions, "preset", "12", 0);
        av_dict_set(&encoderOptions, "svtav1-params", "lp=2", 0);
    }
    if (encoderName && std::string_view(encoderName) == "libx265" && transfer == AVCOL_TRC_SMPTE2084)
        av_dict_set(&encoderOptions, "x265-params", "hdr10=1:max-cll=4000,400", 0);
    for (const auto& [key, value] : encoderOverrides)
        av_dict_set(&encoderOptions, key, value, 0);
    const int encoderStatus = avcodec_open2(encoder.get(), codec, &encoderOptions);
    av_dict_free(&encoderOptions);
    checkAv(encoderStatus, "Open fixture encoder");
    AVStream* stream = avformat_new_stream(output.get(), nullptr);
    if (!stream)
        throw std::runtime_error("Allocate fixture stream");
    stream->time_base = encoder->time_base;
    stream->avg_frame_rate = encoder->framerate;
    checkAv(avcodec_parameters_from_context(stream->codecpar, encoder.get()), "Copy fixture encoder parameters");
    checkAv(avio_open(&output->pb, path.string().c_str(), AVIO_FLAG_WRITE), "Open fixture output");
    checkAv(avformat_write_header(output.get(), nullptr), "Write fixture header");
    AVFramePtr frame(av_frame_alloc());
    if (!frame)
        throw std::runtime_error("Allocate fixture frame");
    frame->format = encoder->pix_fmt;
    frame->width = encoder->width;
    frame->height = encoder->height;
    frame->pts = 0;
    frame->duration = 1;
    frame->sample_aspect_ratio = {1, 1};
    frame->color_range = encoder->color_range;
    frame->color_primaries = encoder->color_primaries;
    frame->color_trc = encoder->color_trc;
    frame->colorspace = encoder->colorspace;
    checkAv(av_frame_get_buffer(frame.get(), 32), "Allocate fixture pixels");
    AVPacketPtr packet(av_packet_alloc());
    if (!packet)
        throw std::runtime_error("Allocate fixture packet");
    const auto drain = [&] {
        int status;
        while ((status = avcodec_receive_packet(encoder.get(), packet.get())) >= 0) {
            av_packet_rescale_ts(packet.get(), encoder->time_base, stream->time_base);
            packet->stream_index = stream->index;
            checkAv(av_interleaved_write_frame(output.get(), packet.get()), "Write fixture packet");
        }
        if (status != AVERROR_EOF && status != AVERROR(EAGAIN))
            checkAv(status, "Drain fixture encoder");
    };
    const auto* descriptor = av_pix_fmt_desc_get(encoder->pix_fmt);
    for (int index = 0; index < (textured ? 100 : 1); ++index) {
        checkAv(av_frame_make_writable(frame.get()), "Make fixture frame writable");
        frame->pts = index;
        for (int plane = 0; plane < 3; ++plane) {
            const int width = plane == 0 ? frame->width : AV_CEIL_RSHIFT(frame->width, descriptor->log2_chroma_w);
            const int height = plane == 0 ? frame->height : AV_CEIL_RSHIFT(frame->height, descriptor->log2_chroma_h);
            for (int row = 0; row < height; ++row)
                for (int col = 0; col < width; ++col) {
                    const int value = textured && (plane == 0 || transfer != AVCOL_TRC_BT709 || coloredChroma) ? fixtureSample(plane, row, col, index) : 128;
                    auto* line = frame->data[plane] + row * frame->linesize[plane];
                    if (deep)
                        reinterpret_cast<uint16_t*>(line)[col] = static_cast<uint16_t>(value * 4);
                    else
                        line[col] = static_cast<uint8_t>(value);
                }
        }
        checkAv(avcodec_send_frame(encoder.get(), frame.get()), "Encode fixture frame");
        drain();
    }
    checkAv(avcodec_send_frame(encoder.get(), nullptr), "Flush fixture encoder");
    drain();
    checkAv(av_write_trailer(output.get()), "Write fixture trailer");
    checkAv(avio_closep(&output->pb), "Close fixture file");
}

const fs::path& shortVideoFixture() {
    struct Fixture {
        fs::path directory = fs::temp_directory_path() / ("watermarking-video-fixture-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::path path = directory / "textured-10bit.mkv";
        Fixture() {
            fs::create_directories(directory);
            writeVideoFixture(path, true, AVCOL_TRC_BT709, true);
        }
        ~Fixture() {
            std::error_code error;
            fs::remove_all(directory, error);
        }
    };
    static const Fixture fixture;
    return fixture.path;
}

// Inspect the first decoded sample and transfer tag to verify HDR conversion
std::pair<int, AVColorTransferCharacteristic> firstVideoLuma(const fs::path& path, video_utils::AVFramePtr* decoded = nullptr) {
    using namespace video_utils;
    AVFormatContext* rawInput = nullptr;
    checkAv(avformat_open_input(&rawInput, path.string().c_str(), nullptr, nullptr), "Open test video");
    AVFormatContextPtr input(rawInput);
    checkAv(avformat_find_stream_info(input.get(), nullptr), "Read test video streams");
    const int index = av_find_best_stream(input.get(), AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    checkAv(index, "Find test video stream");
    const auto* parameters = input->streams[index]->codecpar;
    const auto* codec = avcodec_find_decoder(parameters->codec_id);
    AVCodecContextPtr decoder(avcodec_alloc_context3(codec));
    if (!decoder)
        throw std::runtime_error("Allocate test decoder");
    checkAv(avcodec_parameters_to_context(decoder.get(), parameters), "Copy test decoder parameters");
    checkAv(avcodec_open2(decoder.get(), codec, nullptr), "Open test decoder");
    AVPacketPtr packet(av_packet_alloc());
    AVFramePtr frame(av_frame_alloc());
    if (!packet || !frame)
        throw std::runtime_error("Allocate test decode buffers");
    while (true) {
        const int read = av_read_frame(input.get(), packet.get());
        if (read != AVERROR_EOF)
            checkAv(read, "Read test packet");
        if (read == AVERROR_EOF || packet->stream_index == index) {
            checkAv(avcodec_send_packet(decoder.get(), read == AVERROR_EOF ? nullptr : packet.get()), "Decode test packet");
            if (avcodec_receive_frame(decoder.get(), frame.get()) == 0) {
                if (frame->format != AV_PIX_FMT_YUV420P && frame->format != AV_PIX_FMT_YUV422P && frame->format != AV_PIX_FMT_YUV444P)
                    throw std::runtime_error("Expected normalized 8-bit planar YUV output");
                const auto result = std::pair{static_cast<int>(frame->data[0][0]), parameters->color_trc};
                if (decoded)
                    *decoded = std::move(frame);
                return result;
            }
        }
        if (read == AVERROR_EOF)
            throw std::runtime_error("Test video produced no frame");
        av_packet_unref(packet.get());
    }
}

// every frame of an 8-bit planar YUV video, planes packed without row padding
struct DecodedVideo {
    AVPixelFormat format = AV_PIX_FMT_NONE;
    AVColorRange range = AVCOL_RANGE_UNSPECIFIED;
    int width = 0;
    int height = 0;
    int chromaWidth = 0;
    int chromaHeight = 0;
    std::vector<std::array<std::vector<uint8_t>, 3>> frames;
};

DecodedVideo decodeVideoPlanes(const fs::path& path) {
    using namespace video_utils;
    AVFormatContext* rawInput = nullptr;
    checkAv(avformat_open_input(&rawInput, path.string().c_str(), nullptr, nullptr), "Open test video");
    AVFormatContextPtr input(rawInput);
    checkAv(avformat_find_stream_info(input.get(), nullptr), "Read test video streams");
    const int index = av_find_best_stream(input.get(), AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    checkAv(index, "Find test video stream");
    const auto* parameters = input->streams[index]->codecpar;
    const auto* codec = avcodec_find_decoder(parameters->codec_id);
    AVCodecContextPtr decoder(avcodec_alloc_context3(codec));
    if (!decoder)
        throw std::runtime_error("Allocate test decoder");
    checkAv(avcodec_parameters_to_context(decoder.get(), parameters), "Copy test decoder parameters");
    checkAv(avcodec_open2(decoder.get(), codec, nullptr), "Open test decoder");
    AVPacketPtr packet(av_packet_alloc());
    AVFramePtr frame(av_frame_alloc());
    if (!packet || !frame)
        throw std::runtime_error("Allocate test decode buffers");
    DecodedVideo video;
    const auto receive = [&] {
        while (avcodec_receive_frame(decoder.get(), frame.get()) == 0) {
            const auto* descriptor = av_pix_fmt_desc_get(static_cast<AVPixelFormat>(frame->format));
            if (!descriptor || descriptor->nb_components != 3 || !(descriptor->flags & AV_PIX_FMT_FLAG_PLANAR) || descriptor->comp[0].depth != 8 || (descriptor->flags & AV_PIX_FMT_FLAG_RGB))
                throw std::runtime_error("Expected 8-bit planar YUV frames");
            if (video.frames.empty()) {
                video.format = static_cast<AVPixelFormat>(frame->format);
                video.range = frame->color_range;
                video.width = frame->width;
                video.height = frame->height;
                video.chromaWidth = AV_CEIL_RSHIFT(frame->width, descriptor->log2_chroma_w);
                video.chromaHeight = AV_CEIL_RSHIFT(frame->height, descriptor->log2_chroma_h);
            }
            if (frame->format != video.format || frame->width != video.width || frame->height != video.height)
                throw std::runtime_error("Video frame layout changed");
            auto& planes = video.frames.emplace_back();
            for (int plane = 0; plane < 3; ++plane) {
                const int width = plane == 0 ? video.width : video.chromaWidth, height = plane == 0 ? video.height : video.chromaHeight;
                for (int row = 0; row < height; ++row)
                    planes[plane].insert(planes[plane].end(), frame->data[plane] + row * frame->linesize[plane], frame->data[plane] + row * frame->linesize[plane] + width);
            }
        }
    };
    while (av_read_frame(input.get(), packet.get()) >= 0) {
        if (packet->stream_index == index) {
            checkAv(avcodec_send_packet(decoder.get(), packet.get()), "Decode test packet");
            receive();
        }
        av_packet_unref(packet.get());
    }
    checkAv(avcodec_send_packet(decoder.get(), nullptr), "Flush test decoder");
    receive();
    return video;
}

// checks the watermarked output of a colored fixture: chroma as the source (8-bit scale), luma unchanged on the skipped frames,
// tolerance covers the rounding of a 10-bit reduction
void expectFixturePlanes(const DecodedVideo& video, const int interval, const int tolerance = 0) {
    ASSERT_EQ(video.frames.size(), 100);
    for (size_t index = 0; index < video.frames.size(); ++index) {
        std::array<int, 2> maximumError{}; // luma, chroma
        for (int plane = 0; plane < 3; ++plane) {
            const int width = plane == 0 ? video.width : video.chromaWidth, height = plane == 0 ? video.height : video.chromaHeight;
            for (int row = 0; row < height; ++row)
                for (int col = 0; col < width; ++col) {
                    const int error = std::abs(video.frames[index][plane][static_cast<size_t>(row) * width + col] - fixtureSample(plane, row, col, static_cast<int>(index)));
                    maximumError[plane == 0 ? 0 : 1] = std::max(maximumError[plane == 0 ? 0 : 1], error);
                }
        }
        EXPECT_LE(maximumError[1], tolerance) << "frame " << index;
        if (index % interval == 0)
            EXPECT_GT(maximumError[0], tolerance) << "frame " << index;
        else
            EXPECT_LE(maximumError[0], tolerance) << "frame " << index;
    }
}

// Collect per-frame correlations from the public detection log
std::vector<float> capturedCorrelations(VideoSession* session, int& framesProcessed) {
    testing::internal::CaptureStdout();
    framesProcessed = detectVideo(session);
    const std::string output = testing::internal::GetCapturedStdout();
    std::vector<float> correlations;
    const std::regex pattern(R"(Correlation for frame: \d+: ([-+0-9.eE]+))");
    for (auto it = std::sregex_iterator(output.begin(), output.end(), pattern); it != std::sregex_iterator(); ++it)
        correlations.push_back(std::stof((*it)[1].str()));
    return correlations;
}

#if defined(_USE_OPENCL_)
void setPortableReductionOverride(const bool enabled) {
#ifdef _WIN32
    _putenv_s("WATERMARK_OPENCL_FORCE_PORTABLE_REDUCTIONS", enabled ? "1" : "");
#else
    if (enabled)
        setenv("WATERMARK_OPENCL_FORCE_PORTABLE_REDUCTIONS", "1", 1);
    else
        unsetenv("WATERMARK_OPENCL_FORCE_PORTABLE_REDUCTIONS");
#endif
}

// Restore the caller's reduction override when a test leaves its scope
class PortableReductionOverrideGuard {
    std::optional<std::string> original;

  public:
    PortableReductionOverrideGuard() {
        if (const char* value = std::getenv("WATERMARK_OPENCL_FORCE_PORTABLE_REDUCTIONS"))
            original = value;
    }

    ~PortableReductionOverrideGuard() {
#ifdef _WIN32
        _putenv_s("WATERMARK_OPENCL_FORCE_PORTABLE_REDUCTIONS", original ? original->c_str() : "");
#else
        if (original)
            setenv("WATERMARK_OPENCL_FORCE_PORTABLE_REDUCTIONS", original->c_str(), 1);
        else
            unsetenv("WATERMARK_OPENCL_FORCE_PORTABLE_REDUCTIONS");
#endif
    }
};

struct ReductionPathResult {
    SessionPixelData pixels;
    float correlation = 0.0f;
};

// Compare both embedding and detection across OpenCL reduction paths
ReductionPathResult runReductionPath(const int order = defaultP) {
    ImageHandle reductionSession = createImageSession(defaultPassword, order, defaultPsnr);
    loadImage(reductionSession.get(), colorImage.string());
    ReductionPathResult result;
    result.pixels = embedAndRead(reductionSession.get());
    prepareDetectionImage(reductionSession.get());
    result.correlation = detectEmbeddedBuffer(reductionSession.get());
    return result;
}
#endif
} // namespace

// Give each test a loaded session and a temporary directory cleaned up afterward
class WatermarkTest : public ::testing::Test {
  protected:
    ImageHandle session{nullptr};
    fs::path tempDir;

    void SetUp() override {
        initializeEnvironment(0);
        const auto uniqueId = std::chrono::steady_clock::now().time_since_epoch().count();
        tempDir = fs::temp_directory_path() / ("watermarking-thesis-tests-" + std::to_string(uniqueId));
        fs::create_directories(tempDir);
        session = createImageSession(defaultPassword, defaultP, defaultPsnr);
        loadImage(session.get(), colorImage.string());
    }

    void TearDown() override {
        session.reset();
        std::error_code ignored;
        fs::remove_all(tempDir, ignored);
    }
};

#if defined(_USE_OPENCL_)
TEST(VideoDeviceIdentityTest, MatchesPhysicalDevicesRatherThanEnumerationIndices) {
    using namespace video_utils;
    VideoDeviceIdentity first, second, unknown;
    first.uuid = std::array<uint8_t, 16>{1};
    second.uuid = std::array<uint8_t, 16>{2};
    EXPECT_EQ(matchingVideoDevice(first, std::array{second, first}), 1U);
    EXPECT_EQ(matchingVideoDevice(second, std::array{second, first}), 0U);
    EXPECT_THROW(matchingVideoDevice(first, std::array{first, first}), std::runtime_error);
    EXPECT_THROW(matchingVideoDevice(unknown, std::array{first, second}), std::runtime_error);
    first = {};
    first.luid = std::array<uint8_t, 8>{7};
    first.nodeMask = 1;
    second = first;
    second.nodeMask = 2;
    EXPECT_FALSE(sameVideoDevice(first, second));
    EXPECT_EQ(matchingVideoDevice(first, std::array{second, first}), 1U);
    first = {};
    first.domain = 0;
    first.bus = 3;
    first.slot = 2;
    first.function = 0;
    second = first;
    second.domain = 1;
    EXPECT_EQ(matchingVideoDevice(first, std::array{second, first}), 1U);
    first.domain.reset();
    EXPECT_THROW(matchingVideoDevice(first, std::array{second, second}), std::runtime_error);
}

TEST(VideoDeviceIdentityTest, SelectedOpenCLNvidiaDeviceCanOpenHostStagedNvenc) {
    using namespace video_utils;
    int checked = 0;
    const auto devices = getAvailableDevices();
    for (size_t index = 0; index < devices.size(); ++index) {
        ASSERT_TRUE(initializeEnvironment(static_cast<int>(index)));
        const auto& device = OclQueueManager::getInstance().getDevice();
        if (device.getInfo<CL_DEVICE_VENDOR_ID>() != 0x10de)
            continue;
        OptionDict options;
        const int ordinal = bindHostNvencDevice(device(), options);
        OptionDict conflicting;
        av_dict_set(conflicting.ptr(), "gpu", std::to_string(ordinal + 1).c_str(), 0);
        EXPECT_THROW(bindHostNvencDevice(device(), conflicting), std::runtime_error);
        const auto* codec = avcodec_find_encoder_by_name("hevc_nvenc");
        ASSERT_TRUE(codec);
        AVCodecContextPtr encoder(avcodec_alloc_context3(codec));
        ASSERT_TRUE(encoder);
        encoder->width = 320;
        encoder->height = 180;
        encoder->pix_fmt = AV_PIX_FMT_YUV420P; // no hardware surfaces or interop context
        encoder->time_base = {1, 30};
        encoder->framerate = {30, 1};
        av_dict_set(options.ptr(), "preset", "p2", 0);
        ASSERT_GE(avcodec_open2(encoder.get(), codec, options.ptr()), 0);
        int64_t actual = -1;
        ASSERT_GE(av_opt_get_int(encoder->priv_data, "gpu", 0, &actual), 0);
        EXPECT_EQ(actual, ordinal);
        AVFramePtr frame(av_frame_alloc());
        frame->format = encoder->pix_fmt;
        frame->width = encoder->width;
        frame->height = encoder->height;
        ASSERT_GE(av_frame_get_buffer(frame.get(), 32), 0);
        for (int plane = 0; plane < 3; ++plane)
            for (int y = 0; y < (plane == 0 ? frame->height : frame->height / 2); ++y)
                std::memset(frame->data[plane] + y * frame->linesize[plane], 128, plane == 0 ? frame->width : frame->width / 2);
        AVPacketPtr packet(av_packet_alloc());
        int packets = 0;
        const auto receive = [&] {
            while (avcodec_receive_packet(encoder.get(), packet.get()) >= 0) {
                ++packets;
                av_packet_unref(packet.get());
            }
        };
        for (int i = 0; i < 4; ++i) {
            frame->pts = i;
            ASSERT_GE(avcodec_send_frame(encoder.get(), frame.get()), 0);
            receive();
        }
        ASSERT_GE(avcodec_send_frame(encoder.get(), nullptr), 0);
        receive();
        EXPECT_EQ(packets, 4);
        ++checked;
    }
    if (!checked)
        GTEST_SKIP() << "No NVIDIA OpenCL GPU";
}

// Concurrent first use must publish complete programs for the active context
TEST(OpenCLCacheTest, ConcurrentColdProgramBuildsShareAValidProgram) {
    initializeEnvironment(0);
    std::vector<std::future<cl_context>> builders;
    for (int i = 0; i < 8; ++i)
        builders.push_back(std::async(std::launch::async, [] {
            const auto watermark = cl_utils::OpenCLKernelCache<9>::getProgram();
            const auto utility = cl_utils::UtilityKernelCache::getProgram();
            const auto generation = cl_utils::GenerationKernelCache::getProgram();
            const auto context = watermark.getInfo<CL_PROGRAM_CONTEXT>();
            if (context != utility.getInfo<CL_PROGRAM_CONTEXT>() || context != generation.getInfo<CL_PROGRAM_CONTEXT>())
                throw std::runtime_error("Program contexts differ");
            return context.get();
        }));
    for (auto& builder : builders)
        EXPECT_EQ(builder.get(), OclQueueManager::getInstance().getContextRaw());
}
#endif

TEST(WatermarkCryptoTest, Sha256MatchesPublishedVectors) {
    EXPECT_EQ(hexDigest(WatermarkCrypto::sha256("")), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    EXPECT_EQ(hexDigest(WatermarkCrypto::sha256("abc")), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
}

// the AVX2 (and AVX-512) Box-Muller transforms are the reference transform, bit for bit, for every input
TEST(WatermarkCryptoTest, VectorBoxMullerMatchesTheScalarReferenceForEveryInput) {
    const std::vector<uint32_t> pairs = everyRandomPair();
    std::vector<std::array<uint64_t, 8>> blocks(pairs.size() / 8);
    for (size_t i = 0; i < pairs.size(); ++i)
        blocks[i / 8][i % 8] = (static_cast<uint64_t>(pairs[i]) << 40) | ((i * 0x9E3779B97F4A7C15ull) >> 24);
    const std::vector<float> expected = referenceBoxMuller(pairs);

    std::vector<float> vector2(pairs.size());
    for (size_t block = 0; block < blocks.size(); block += 2)
        WatermarkCrypto::generateBoxMullerNormalBlockPair(blocks[block], blocks[block + 1], vector2.data() + (block * 8));
    EXPECT_EQ(bitDifferences(vector2, expected), "") << "AVX2";
#if defined(__AVX512F__)
    std::vector<float> vector4(pairs.size());
    for (size_t block = 0; block < blocks.size(); block += 4)
        WatermarkCrypto::generateBoxMullerNormalBlockQuad(blocks.data() + block, vector4.data() + (block * 8));
    EXPECT_EQ(bitDifferences(vector4, expected), "") << "AVX-512";
#endif
}

// the CPU generation (vector ChaCha20 and Box-Muller, the scalar tail) is the scalar reference, partial last blocks included
TEST(WatermarkCryptoTest, HostWatermarkMatchesTheScalarReference) {
    const std::array<uint32_t, 16> baseState = WatermarkCrypto::computeBaseState(defaultPassword);
    for (const int64_t numElements : {1, 7, 8, 9, 63, 64, 65, 127, 129, 136, 324, 1000, 12293, 3840 * 2160}) {
        const auto host = WatermarkCrypto::generateHalfWatermark(baseState, numElements);
        const std::vector<uint16_t> hostBits(host.get(), host.get() + numElements);
        EXPECT_EQ(bitDifferences(hostBits, referenceWatermark(baseState, numElements)), "") << numElements << " values";
    }
}

#if defined(_USE_CUDA_) || defined(_USE_OPENCL_)
// the Box-Muller transform of the device generation kernel is the CPU reference, bit for bit, for every input
TEST(WatermarkGenerationTest, DeviceBoxMullerMatchesTheCpuForEveryInput) {
    const std::vector<uint32_t> pairs = everyRandomPair();
    const std::vector<float> expected = referenceBoxMuller(pairs);
    forEachDevice([&](const std::string& device) { EXPECT_EQ(bitDifferences(deviceBoxMuller(pairs), expected), "") << device; });
}

// the watermark generated on the device is the CPU watermark, partial last blocks included
TEST(WatermarkGenerationTest, DeviceWatermarkMatchesTheCpu) {
    const std::array<uint32_t, 16> baseState = WatermarkCrypto::computeBaseState(defaultPassword);
    forEachDevice([&](const std::string& device) {
        for (const int64_t numElements : {1, 7, 8, 9, 63, 64, 65, 129, 324, 1920 * 1080 + 5, 3840 * 2160}) {
            const auto host = WatermarkCrypto::generateHalfWatermark(baseState, numElements);
            const std::vector<uint16_t> hostBits(host.get(), host.get() + numElements);
            EXPECT_EQ(bitDifferences(deviceWatermark(baseState, numElements), hostBits), "") << device << ", " << numElements << " values";
        }
    });
}
#endif

#if defined(_USE_OPENCL_)
TEST(OpenCLHalfTest, SelectedConversionMatchesHostAtEveryHalfAndRoundingBoundary) {
    PortableReductionOverrideGuard restoreOverride;
    setPortableReductionOverride(false);
    std::vector<float> values;
    for (uint32_t bits = 0; bits < 0x10000u; ++bits)
        values.push_back(HalfFloat::toFloat(static_cast<uint16_t>(bits)));
    for (uint32_t bits = 0; bits < 0x7C00u; ++bits) {
        const float midpoint = bits == 0x7BFFu ? 65520.0f : 0.5f * (HalfFloat::toFloat(static_cast<uint16_t>(bits)) + HalfFloat::toFloat(static_cast<uint16_t>(bits + 1)));
        for (const float value : {midpoint, -midpoint}) {
            values.push_back(value);
            values.push_back(std::nextafter(value, std::numeric_limits<float>::infinity()));
            values.push_back(std::nextafter(value, -std::numeric_limits<float>::infinity()));
        }
    }
    uint32_t state = 2026;
    for (int i = 0; i < 100000; ++i) {
        state = state * 1664525u + 1013904223u;
        values.push_back(std::bit_cast<float>(state));
    }
    std::vector<uint16_t> expected(values.size());
    std::transform(values.begin(), values.end(), expected.begin(), HalfFloat::fromFloat);
    forEachDevice([&](const std::string& name) {
        auto& manager = OclQueueManager::getInstance();
        const auto selected = cl_utils::OpenCLKernelCache<3>::getProgram();
        const std::string options = cl_utils::usesNativeHalfStorage(selected) ? "-cl-std=CL1.2 -DWM_NATIVE_HALF_STORAGE=1" : "-cl-std=CL1.2";
        cl::Program program(manager.getContext(), halfKernels + R"CLC(
            __kernel void convert_test(__global const float* input, __global ushort* output, int n) {
                int i = get_global_id(0);
                if (i < n) output[i] = toHalfBits(input[i]);
            }
        )CLC");
        program.build(manager.getDevice(), options.c_str());
        const int count = static_cast<int>(values.size());
        const OclArray<float> input(count, 1, values.data(), manager.getQueueRaw());
        const OclArray<uint16_t> output(count, manager.getQueueRaw());
        const auto kernel = cl_utils::KernelBuilder(program, "convert_test").args(input.clBuffer(), output.clBuffer(), count).build();
        manager.getQueue().enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(cl_utils::roundUp(count, 64)), cl::NDRange(64));
        std::vector<uint16_t> actual(values.size());
        output.toHost(actual.data());
        EXPECT_EQ(bitDifferences(actual, expected), "") << name;
    });
}

TEST(OpenCLReductionTest, AvailableDevicesMatchForcedPortableResults) {
    PortableReductionOverrideGuard restoreOverride;
    const std::vector<std::string> devices = getAvailableDevices();
    ASSERT_FALSE(devices.empty());

    for (size_t deviceIndex = 0; deviceIndex < devices.size(); ++deviceIndex) {
        setPortableReductionOverride(false);
        ASSERT_TRUE(initializeEnvironment(static_cast<int>(deviceIndex))) << devices[deviceIndex];
        buildOpenCLKernels(); // every documented prediction order must compile on every device
        for (const int order : {3, 5, 7, 9}) {
            SCOPED_TRACE(std::format("{} p={}", devices[deviceIndex], order));
            setPortableReductionOverride(false);
            const cl::Program selectedProgram = cl_utils::OpenCLKernelCache<defaultP>::getProgram();
            const cl_utils::ReductionMode selectedMode = cl_utils::reductionMode(selectedProgram);
            const ReductionPathResult selected = runReductionPath(order);
            EXPECT_TRUE(std::isfinite(selected.correlation)) << devices[deviceIndex];
            EXPECT_GT(selected.correlation, 0.5f) << devices[deviceIndex];

            if (selectedMode == cl_utils::ReductionMode::Portable)
                continue;

            setPortableReductionOverride(true);
            const cl::Program portableProgram = cl_utils::OpenCLKernelCache<defaultP>::getProgram();
            ASSERT_EQ(cl_utils::reductionMode(portableProgram), cl_utils::ReductionMode::Portable) << devices[deviceIndex];
            const ReductionPathResult portable = runReductionPath(order);
            EXPECT_EQ(selected.pixels.width, portable.pixels.width) << devices[deviceIndex];
            EXPECT_EQ(selected.pixels.height, portable.pixels.height) << devices[deviceIndex];
            EXPECT_EQ(selected.pixels.channels, portable.pixels.channels) << devices[deviceIndex];
            ASSERT_EQ(selected.pixels.pixels.size(), portable.pixels.pixels.size());
            int maxDifference = 0;
            size_t different = 0;
            for (size_t i = 0; i < selected.pixels.pixels.size(); ++i) {
                const int delta = std::abs(static_cast<int>(selected.pixels.pixels[i]) - portable.pixels.pixels[i]);
                maxDifference = std::max(maxDifference, delta);
                different += delta != 0;
            }
            std::cout << devices[deviceIndex] << ": " << different << " changed pixels, max difference " << maxDifference << ", correlation difference "
                      << std::abs(selected.correlation - portable.correlation) << '\n';
            // Different floating-point reduction trees can cross an 8-bit rounding
            // boundary, shift-sum references verify accuracy
            EXPECT_LE(maxDifference, 1) << devices[deviceIndex];
            EXPECT_NEAR(selected.correlation, portable.correlation, 1.0e-4f) << devices[deviceIndex];
        }
    }
}

// devices with subgroup reductions verify against double-precision reference
TEST(OpenCLReductionTest, ShiftSumsMatchDoublePrecisionReferenceOnEveryDevice) {
    PortableReductionOverrideGuard restoreOverride;
    const auto devices = getAvailableDevices();
    ASSERT_FALSE(devices.empty());
    for (size_t deviceIndex = 0; deviceIndex < devices.size(); ++deviceIndex) {
        ASSERT_TRUE(initializeEnvironment(static_cast<int>(deviceIndex))) << devices[deviceIndex];
        for (const bool portable : {false, true}) {
            setPortableReductionOverride(portable);
            for (const int order : {3, 5, 7, 9}) {
                const auto program = [&] {
                    switch (order) {
                    case 3: return cl_utils::OpenCLKernelCache<3>::getProgram();
                    case 5: return cl_utils::OpenCLKernelCache<5>::getProgram();
                    case 7: return cl_utils::OpenCLKernelCache<7>::getProgram();
                    default: return cl_utils::OpenCLKernelCache<9>::getProgram();
                    }
                }();
                const int groupSize = cl_utils::workGroupSize(program);
                const int pad = order / 2;
                const int span = 2 * (order - 1) + 1;
                const int numShifts = (span * span + 1) / 2;
                const int borderSize = 4 * pad;
                const int shiftSize = numShifts * (1 + 2 * borderSize);
                const int shiftGroups = order >= 9 ? 3 : (order >= 7 ? 2 : 1);
                const int interiorCols = order >= 7 ? 2 : 1;
                auto& manager = OclQueueManager::getInstance();
                for (const auto [width, height] : {
                         std::pair{19, 17},
                         std::pair{67, 65}
                }) {
                    SCOPED_TRACE(std::format("{} p={} {}x{} portable={}", devices[deviceIndex], order, width, height, portable));
                    std::vector<float> input(width * height);
                    uint32_t state = 2026;
                    for (float& value : input) {
                        state = state * 1664525u + 1013904223u;
                        value = static_cast<float>(state >> 8) * (255.0f / 16777216.0f);
                    }
                    const OclArray<float> image(height, width, input.data(), manager.getQueueRaw());
                    auto shifts = OclArray<uint64_t>::zeros(shiftSize, manager.getQueueRaw());
                    OclArray<float> border(6 * pad * width, manager.getQueueRaw());
                    if (order >= 7) {
                        const auto copy = cl_utils::KernelBuilder(program, "me_copy_border_rows").args(image.clBuffer(), border.clBuffer(), width, height).build();
                        manager.getQueue().enqueueNDRangeKernel(copy, cl::NullRange, cl::NDRange(cl_utils::roundUp(6 * pad * width, groupSize)), cl::NDRange(groupSize));
                    }
                    const int perLine = ((std::max(width, height) + 7) / 8 + groupSize - 1) / groupSize;
                    const int tasks = ((height + 7) / 8) * ((width - 2 * pad + interiorCols - 1) / interiorCols);
                    const int interior = std::max((tasks + groupSize - 1) / groupSize, 1) * shiftGroups;
                    const int borderGroups = perLine * 2 * borderSize * shiftGroups;
                    const size_t available = manager.getDevice().getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();
                    const size_t exactScratch = cl_utils::reductionMode(program) == cl_utils::ReductionMode::Subgroup
                                                    ? cl_utils::reductionScratchBytes(program, "me_shift_sums", cl::NDRange(groupSize), numShifts)
                                                    : static_cast<size_t>(17 * groupSize) * sizeof(float);
                    ASSERT_LE(exactScratch, available);
                    const auto kernel = cl_utils::KernelBuilder(program, "me_shift_sums")
                                            .args(image.clBuffer(), border.clBuffer(), shifts.clBuffer(), width, height, interior, perLine, cl::Local(exactScratch))
                                            .build();
                    manager.getQueue().enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange((interior + borderGroups) * groupSize), cl::NDRange(groupSize));
                    std::vector<uint64_t> actual(shiftSize);
                    shifts.toHost(actual.data());
                    const auto pixel = [&](const int row, const int col) { return static_cast<double>(input[std::clamp(col, 0, width - 1) * height + std::clamp(row, 0, height - 1)]); };
                    const auto borderCoord = [&](const int line, const int size) { return line < 2 * pad ? line - pad : size - 3 * pad + line; };
                    int shift = 0;
                    for (int dc = 0; dc < order; ++dc) {
                        for (int dr = dc == 0 ? 0 : 1 - order; dr < order; ++dr, ++shift) {
                            double reference = 0;
                            for (int col = pad; col < width - pad; ++col)
                                for (int row = pad; row < height - pad; ++row)
                                    reference += pixel(row, col) * pixel(row + dr, col + dc);
                            const auto compare = [&](const int index, const double sum) {
                                const double expected = sum / (255.0 * 255.0);
                                EXPECT_NEAR(static_cast<double>(actual[index]) * 1.0e-9, expected, std::max(2.0e-6, expected * 2.0e-6)) << "shift=" << shift << " index=" << index;
                            };
                            compare(shift, reference);
                            for (int line = 0; line < borderSize; ++line) {
                                double rowSum = 0, colSum = 0;
                                const int row = borderCoord(line, height), col = borderCoord(line, width);
                                for (int c = pad; c < width - pad; ++c)
                                    rowSum += pixel(row, c) * pixel(row + dr, c + dc);
                                for (int r = pad; r < height - pad; ++r)
                                    colSum += pixel(r, col) * pixel(r + dr, col + dc);
                                compare(numShifts + shift * borderSize + line, rowSum);
                                compare(numShifts * (1 + borderSize) + shift * borderSize + line, colSum);
                            }
                        }
                    }
                }
            }
        }
    }
}
#endif

TEST_F(WatermarkTest, RejectsImagesSmallerThanPredictionWindow) {
    // Minimal valid 2x2, 24-bit BMP: 54-byte header + 2 padded rows
    std::array<uint8_t, 70> bmp{};
    bmp[0] = 'B';
    bmp[1] = 'M';
    bmp[2] = 70;
    bmp[10] = 54;
    bmp[14] = 40;
    bmp[18] = 2;
    bmp[22] = 2;
    bmp[26] = 1;
    bmp[28] = 24;
    bmp[34] = 16;
    const fs::path tinyImage = tempDir / "tiny.bmp";
    {
        std::ofstream output(tinyImage, std::ios::binary);
        ASSERT_TRUE(output.write(reinterpret_cast<const char*>(bmp.data()), bmp.size()));
    }
    EXPECT_THROW(loadImage(session.get(), tinyImage.string()), std::invalid_argument);
}

TEST_F(WatermarkTest, OriginalPreviewInterleavesRgbPixelsAndTail) {
    // 3x3 BMP -> one 8 pixel AVX2 block AND 1 scalar pixel (test when it is not multiple of 8)
    std::array<uint8_t, 90> bmp{};
    bmp[0] = 'B';
    bmp[1] = 'M';
    bmp[2] = 90;
    bmp[10] = 54;
    bmp[14] = 40;
    bmp[18] = 3;
    bmp[22] = 3;
    bmp[26] = 1;
    bmp[28] = 24;
    bmp[34] = 36;
    std::vector<uint8_t> expected;
    for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 3; ++col) {
            const uint8_t red = static_cast<uint8_t>((row * 3 + col) * 17);
            expected.insert(expected.end(), {red, static_cast<uint8_t>(red + 1), static_cast<uint8_t>(red + 2)});
            const size_t offset = 54 + static_cast<size_t>(2 - row) * 12 + col * 3;
            bmp[offset] = static_cast<uint8_t>(red + 2);
            bmp[offset + 1] = static_cast<uint8_t>(red + 1);
            bmp[offset + 2] = red;
        }
    const fs::path input = tempDir / "preview_tail.bmp";
    {
        std::ofstream output(input, std::ios::binary);
        ASSERT_TRUE(output.write(reinterpret_cast<const char*>(bmp.data()), bmp.size()));
    }
    loadImage(session.get(), input.string(), true);
    const OriginalPixelData preview = takeOriginalPixelData(session.get());
    EXPECT_EQ(preview.width, 3);
    EXPECT_EQ(preview.height, 3);
    EXPECT_EQ(preview.channels, 3);
    EXPECT_EQ(preview.pixels, expected);
}

TEST_F(WatermarkTest, EmbedsAndDetects) {
    embedImage(session.get());
    finish();
    prepareDetectionImage(session.get());
    const float correlation = detectEmbeddedBuffer(session.get());
    EXPECT_TRUE(std::isfinite(correlation));
    EXPECT_GT(correlation, 0.5f);
}

TEST_F(WatermarkTest, SavesReloadsAndDetectsFromTemporaryDirectory) {
    embedImage(session.get());
    finish();

    const fs::path requestedPath = tempDir / "result.png";
    const fs::path savedPath = tempDir / "resultW_ME.png";
    saveImage(session.get(), requestedPath.string());
    ASSERT_TRUE(fs::exists(savedPath));

    ImageHandle diskSession = createImageSession(defaultPassword, defaultP, defaultPsnr);
    loadImage(diskSession.get(), savedPath.string());
    const float diskCorrelation = detectLoadedImage(diskSession.get());
    EXPECT_TRUE(std::isfinite(diskCorrelation));
    EXPECT_GT(diskCorrelation, 0.65f);
}

TEST_F(WatermarkTest, ReusesSessionAcrossSameSizeRgbAndGrayImages) {
    const SessionPixelData rgb = embedAndRead(session.get());
    ASSERT_EQ(rgb.channels, 3);

    loadImage(session.get(), grayImage.string());
    const SessionPixelData gray = embedAndRead(session.get());
    EXPECT_EQ(gray.width, rgb.width);
    EXPECT_EQ(gray.height, rgb.height);
    EXPECT_EQ(gray.channels, 1);
    EXPECT_EQ(gray.pixels.size(), static_cast<size_t>(gray.width) * gray.height);
}

TEST_F(WatermarkTest, SameInputsAreDeterministicAcrossSessions) {
    const SessionPixelData first = embedAndRead(session.get());

    ImageHandle secondSession = createImageSession(defaultPassword, defaultP, defaultPsnr);
    loadImage(secondSession.get(), colorImage.string());
    const SessionPixelData second = embedAndRead(secondSession.get());
    EXPECT_EQ(first.pixels, second.pixels);
}

TEST_F(WatermarkTest, ExportedImageRemainsStableAcrossSessionReuse) {
    embedImage(session.get());
    finish();
    ExportHandle exported = createReusableExportBuffer();
    exportForSave(session.get(), exported.get());
    const fs::path first = tempDir / "firstW_ME.png";
    const fs::path second = tempDir / "secondW_ME.png";
    const fs::path third = tempDir / "thirdW_ME.png";
    flushToDiskAsync(exported.get(), (tempDir / "first.png").string());

    updateSessionParams(session.get(), defaultP, 30.0f);
    embedImage(session.get());
    finish();
    flushToDiskAsync(exported.get(), (tempDir / "second.png").string());

    std::ifstream firstFile(first, std::ios::binary);
    std::ifstream secondFile(second, std::ios::binary);
    ASSERT_TRUE(firstFile && secondFile);
    const std::vector<char> firstBytes(std::istreambuf_iterator<char>{firstFile}, {});
    const std::vector<char> secondBytes(std::istreambuf_iterator<char>{secondFile}, {});
    EXPECT_EQ(firstBytes, secondBytes);

    exportForSave(session.get(), exported.get());
    flushToDiskAsync(exported.get(), (tempDir / "third.png").string());
    std::ifstream thirdFile(third, std::ios::binary);
    ASSERT_TRUE(thirdFile);
    const std::vector<char> thirdBytes(std::istreambuf_iterator<char>{thirdFile}, {});
    EXPECT_NE(firstBytes, thirdBytes);
}

TEST_F(WatermarkTest, DifferentPasswordsProduceDifferentWatermarks) {
    const SessionPixelData first = embedAndRead(session.get());

    ImageHandle secondSession = createImageSession("a_different_password", defaultP, defaultPsnr);
    loadImage(secondSession.get(), colorImage.string());
    const SessionPixelData second = embedAndRead(secondSession.get());
    EXPECT_NE(first.pixels, second.pixels);
}

TEST_F(WatermarkTest, PsnrOnlyUpdatePreservesTheDeterministicWatermark) {
    const SessionPixelData original = embedAndRead(session.get());

    updateSessionParams(session.get(), defaultP, 30.0f);
    const SessionPixelData stronger = embedAndRead(session.get());
    EXPECT_NE(original.pixels, stronger.pixels);

    updateSessionParams(session.get(), defaultP, defaultPsnr);
    const SessionPixelData restored = embedAndRead(session.get());
    EXPECT_EQ(original.pixels, restored.pixels);
}

// a p change keeps the watermark (it depends only on the password and the size): the same output as a new session with that p
TEST_F(WatermarkTest, PredictionOrderUpdateKeepsTheWatermark) {
    const SessionPixelData original = embedAndRead(session.get());

    updateSessionParams(session.get(), 5, defaultPsnr);
    const SessionPixelData updated = embedAndRead(session.get());
    ImageHandle fresh = createImageSession(defaultPassword, 5, defaultPsnr);
    loadImage(fresh.get(), colorImage.string());
    EXPECT_EQ(updated.pixels, embedAndRead(fresh.get()).pixels);

    updateSessionParams(session.get(), defaultP, defaultPsnr);
    EXPECT_EQ(original.pixels, embedAndRead(session.get()).pixels);
}

TEST_F(WatermarkTest, SupportsEveryDocumentedPredictionOrder) {
    for (const int predictionOrder : {3, 5, 7, 9}) {
        ImageHandle pSession = createImageSession(defaultPassword, predictionOrder, defaultPsnr);
        loadImage(pSession.get(), colorImage.string());
        const SessionPixelData output = embedAndRead(pSession.get());
        EXPECT_EQ(output.pixels.size(), static_cast<size_t>(output.width) * output.height * output.channels) << "p=" << predictionOrder;
        prepareDetectionImage(pSession.get());
        const float correlation = detectEmbeddedBuffer(pSession.get());
        EXPECT_TRUE(std::isfinite(correlation)) << "p=" << predictionOrder;
        EXPECT_GT(correlation, 0.5f) << "p=" << predictionOrder;
    }
}

TEST_F(WatermarkTest, RejectsUndocumentedPredictionOrder) { EXPECT_THROW(createImageSession(defaultPassword, 4, defaultPsnr), std::invalid_argument); }

// A failed parameter or image update must leave the previous session usable
TEST_F(WatermarkTest, RejectedUpdatesPreserveAWorkingSession) {
    const auto before = embedAndRead(session.get());
    EXPECT_THROW(updateSessionParams(session.get(), 4, defaultPsnr), std::invalid_argument);
    EXPECT_THROW(updateSessionParams(session.get(), 3, std::numeric_limits<float>::infinity()), std::invalid_argument);
    updateSessionParams(session.get(), defaultP, defaultPsnr);
    EXPECT_EQ(embedAndRead(session.get()).pixels, before.pixels);

    const fs::path tiny = tempDir / "tiny.bmp";
    writeNoiseBmp(tiny, 5, 5);
    loadImage(session.get(), tiny.string());
    const auto small = embedAndRead(session.get());
    EXPECT_THROW(updateSessionParams(session.get(), 9, defaultPsnr), std::invalid_argument);
    EXPECT_EQ(embedAndRead(session.get()).pixels, small.pixels);
    writeNoiseBmp(tiny, 2, 2);
    EXPECT_THROW(loadImage(session.get(), tiny.string()), std::invalid_argument);
    EXPECT_EQ(embedAndRead(session.get()).pixels, small.pixels);
}

// Output and detection buffers are valid only after their preparation step
TEST_F(WatermarkTest, UnpreparedAndStaleBuffersReportErrors) {
    auto unloaded = createImageSession(defaultPassword, defaultP, defaultPsnr);
    EXPECT_THROW(embedImage(unloaded.get()), std::logic_error);
    EXPECT_THROW(detectLoadedImage(unloaded.get()), std::logic_error);
    EXPECT_THROW(prepareDetectionImage(session.get()), std::logic_error);
    EXPECT_THROW(getSessionPixelData(session.get()), std::logic_error);
    embedImage(session.get());
    EXPECT_THROW(detectEmbeddedBuffer(session.get()), std::logic_error);
    prepareDetectionImage(session.get());
    EXPECT_TRUE(std::isfinite(detectEmbeddedBuffer(session.get())));
    embedImage(session.get());
    EXPECT_THROW(detectEmbeddedBuffer(session.get()), std::logic_error);
    loadImage(session.get(), colorImage.string());
    EXPECT_THROW(getSessionPreviewFormat(session.get()), std::logic_error);
}

// Exercise size limits without attempting enormous allocations
TEST(BufferSizeTest, RejectsOverflowBeforeAllocating) {
    using namespace InternalUtils;
    EXPECT_EQ(checkedElements(3840, 2160, 4), 33177600);
    EXPECT_THROW(checkedElements(65536, 65536), std::length_error);
    EXPECT_THROW(checkedElements(32768, 32768, 3), std::length_error);
    EXPECT_THROW(checkedElements(-1, 1), std::invalid_argument);
    EXPECT_THROW(checkedProduct(std::numeric_limits<size_t>::max(), 2), std::length_error);
    EXPECT_THROW(checkedPowerOfTwo(std::numeric_limits<size_t>::max()), std::length_error);
    EXPECT_THROW(checkImageDimensions(std::numeric_limits<unsigned int>::max(), 3), std::invalid_argument);
    EXPECT_NO_THROW(checkVideoPitch(2160, 3840, 4096));
    EXPECT_THROW(checkVideoPitch(2160, 3840, 1 << 30), std::length_error);
    EXPECT_THROW(checkVideoPitch(2160, 3840, -4096), std::invalid_argument);
#if defined(_USE_OPENCL_)
    EXPECT_THROW(OclArray<uint8_t>(65536, 65536, nullptr), std::length_error);
#elif defined(_USE_CUDA_)
    EXPECT_THROW(CudaArray<uint8_t>(65536, 65536, nullptr), std::length_error);
#endif
}

#if defined(_USE_CUDA_)
// A constructor that fails during upload must return its allocation to the pool
TEST(CudaBufferTest, FailedUploadReturnsItsAllocationToThePool) {
    initializeEnvironment(0);
    auto& manager = CudaStreamManager::getInstance();
    const auto stream = manager.getComputeStream();
    for (const int channels : {1, 3}) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        manager.getPool().reset(stream);
        float* allocation = nullptr;
        {
            CudaArray<float> buffer(317, 631, channels, stream);
            allocation = buffer.data();
        }
        if (channels == 1)
            EXPECT_THROW(CudaArray<float>(317, 631, static_cast<const float*>(nullptr), stream), std::runtime_error);
        else
            EXPECT_THROW(CudaArray<float>(317, 631, channels, static_cast<const float*>(nullptr), stream), std::runtime_error);
        // consume intentional runtime error before next test
        cudaGetLastError();
        CudaArray<float> recovered(317, 631, channels, stream);
        EXPECT_EQ(recovered.data(), allocation);
        recovered.fillZero();
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
}
#endif

#if defined(_USE_OPENCL_)
// Sessions, preloads and exported buffers must release their old device first
TEST_F(WatermarkTest, OpenCLSwitchRequiresReleasingLiveHandles) {
    if (getAvailableDevices().size() < 2)
        GTEST_SKIP() << "Requires two OpenCL devices";
    auto preload = preloadImageFromDisk(colorImage.string());
    EXPECT_THROW(initializeEnvironment(1), std::runtime_error);
    EXPECT_EQ(getCurrentDeviceIndex(), 0);
    preload.reset();
    EXPECT_THROW(initializeEnvironment(1), std::runtime_error);
    EXPECT_GT(detectLoadedImage(session.get()) + 1.0f, 0.0f);
    session.reset();
    ASSERT_TRUE(initializeEnvironment(1));
    session = createImageSession(defaultPassword, defaultP, defaultPsnr);
    loadImage(session.get(), colorImage.string());
    embedImage(session.get());
    auto exported = createReusableExportBuffer();
    exportForSave(session.get(), exported.get());
    session.reset();
    EXPECT_THROW(initializeEnvironment(0), std::runtime_error);
    flushToDiskAsync(exported.get(), (tempDir / "old-device.png").string());
    exported.reset();
    ASSERT_TRUE(initializeEnvironment(0));
}
#endif

// for p >= 7 the GPU builds copy 3 * pad rows from the top and 3 * pad rows from the bottom of the image. Below 6 * pad rows the two parts
// overlap, and below 3 * pad rows the copy used to read outside the image (for CUDA: run with compute-sanitizer to check the reads)
TEST_F(WatermarkTest, EmbedsAndDetectsTheSmallestImagesForLargeWindows) {
    for (const int order : {7, 9}) {
        const int pad = order / 2;
        for (int height = order; height < 6 * pad; ++height) {
            for (const int width : {order, order + 3}) {
                const std::string size = std::to_string(width) + "x" + std::to_string(height);
                const fs::path input = tempDir / ("small_" + size + ".bmp");
                writeNoiseBmp(input, width, height);
                ImageHandle small = createImageSession(defaultPassword, order, defaultPsnr);
                loadImage(small.get(), input.string());
                const SessionPixelData output = embedAndRead(small.get());
                EXPECT_EQ(output.width, width) << "p=" << order << " " << size;
                EXPECT_EQ(output.height, height) << "p=" << order << " " << size;
                prepareDetectionImage(small.get());
                const float correlation = detectEmbeddedBuffer(small.get());
                EXPECT_TRUE(std::isfinite(correlation)) << "p=" << order << " " << size;
            }
        }
    }
}

TEST(ParameterValidationTest, RejectsNonFiniteAndNonPositivePsnr) {
    constexpr float nan = std::numeric_limits<float>::quiet_NaN();
    constexpr float infinity = std::numeric_limits<float>::infinity();
    for (const float psnr : {nan, infinity, -infinity, 0.0f, -5.0f})
        EXPECT_THROW(createImageSession(defaultPassword, defaultP, psnr), std::invalid_argument) << psnr;
    ImageHandle session = createImageSession(defaultPassword, defaultP, defaultPsnr);
    EXPECT_THROW(updateSessionParams(session.get(), defaultP, nan), std::invalid_argument);
    EXPECT_THROW(updateSessionParams(session.get(), defaultP, infinity), std::invalid_argument);
    EXPECT_NO_THROW(updateSessionParams(session.get(), defaultP, 42.0f));
}

TEST(ParameterValidationTest, RejectsInvalidVideoSettingsBeforeOpeningTheFile) {
    // the file does not exist: invalid settings must fail first (invalid_argument), valid ones then fail to open it (runtime_error)
    VideoSettings settings = makeVideoSettings("missing-video-file.mkv");
    for (const int interval : {0, -1}) {
        settings.watermarkInterval = interval;
        EXPECT_THROW(initVideo(settings), std::invalid_argument) << "interval " << interval;
    }
    settings.watermarkInterval = 1;
    settings.psnr = std::numeric_limits<float>::quiet_NaN();
    EXPECT_THROW(initVideo(settings), std::invalid_argument);
    settings.psnr = 30.0f;
    EXPECT_THROW(initVideo(settings), std::runtime_error);
}

// the Qt UI passes the core UTF-8 paths (QString::toStdString) and builds std::filesystem paths from them, this works only with the
// UTF-8 process code page (utf8.manifest, the test executable has it too). Same steps as the single image and batch workers
TEST_F(WatermarkTest, HandlesNonAsciiPathsLikeTheUi) {
    const auto utf8 = [](const fs::path& path) {
        const std::u8string text = path.u8string();
        return std::string(text.begin(), text.end());
    };
    const fs::path folder = tempDir / u8"\u0392\u03af\u03bd\u03c4\u03b5\u03bf \u03b4\u03bf\u03ba\u03b9\u03bc\u03ae";
    fs::create_directories(folder);
    fs::copy_file(colorImage, folder / u8"\u03b5\u03b9\u03ba\u03cc\u03bd\u03b1.png");
    fs::copy_file(grayImage, folder / u8"\u03b3\u03ba\u03c1\u03af\u03b6\u03b1.jpg");

    // single image: load, embed and save through UTF-8 strings
    ImageHandle single = createImageSession(defaultPassword, defaultP, defaultPsnr);
    loadImage(single.get(), utf8(folder / u8"\u03b5\u03b9\u03ba\u03cc\u03bd\u03b1.png"), true);
    embedImage(single.get());
    finish();
    const fs::path saved = folder / u8"\u03b1\u03c0\u03bf\u03c4\u03ad\u03bb\u03b5\u03c3\u03bc\u03b1.png";
    saveImageExact(single.get(), utf8(saved));
    EXPECT_TRUE(fs::exists(saved));

    // batch: the folder comes back from its UTF-8 text, then list, preload, embed and save
    const fs::path batchFolder(utf8(folder));
    ASSERT_TRUE(fs::is_directory(batchFolder));
    const std::vector<fs::path> files = WatermarkCore::getValidImageFiles(batchFolder);
    ASSERT_EQ(files.size(), 3u);
    const fs::path outputDir = batchFolder / "watermark_output";
    fs::create_directories(outputDir);
    ImageHandle batch = createImageSession(defaultPassword, defaultP, defaultPsnr);
    ExportHandle exportBuffer = createReusableExportBuffer();
    for (const fs::path& file : files) {
        bindPreloadedImage(batch.get(), preloadImageFromDisk(file.string()));
        embedImage(batch.get());
        exportForSave(batch.get(), exportBuffer.get());
        flushToDiskAsync(exportBuffer.get(), (outputDir / file.filename()).string());
    }
    EXPECT_EQ(std::distance(fs::directory_iterator(outputDir), fs::directory_iterator{}), 3);
}

TEST_F(WatermarkTest, PreservesTheAlphaChannelWhenSaving) {
    ASSERT_TRUE(fs::exists(alphaImage)) << alphaImage;
    ImageHandle alphaSession = createImageSession(defaultPassword, defaultP, defaultPsnr);
    loadImage(alphaSession.get(), alphaImage.string(), true);
    const OriginalPixelData original = takeOriginalPixelData(alphaSession.get());
    ASSERT_EQ(original.channels, 4);
    embedImage(alphaSession.get());
    finish();

    const fs::path requested = tempDir / "alpha.png";
    const fs::path saved = tempDir / "alphaW_ME.png";
    saveImage(alphaSession.get(), requested.string());
    ASSERT_TRUE(fs::exists(saved));

    ExportHandle exported = createReusableExportBuffer();
    exportForSave(alphaSession.get(), exported.get());
    const fs::path exportedSaved = tempDir / "alpha_exportW_ME.png";
    auto saveTask = std::async(std::launch::async, flushToDiskAsync, exported.get(), (tempDir / "alpha_export.png").string());
    loadImage(alphaSession.get(), colorImage.string());
    saveTask.get();
    ASSERT_TRUE(fs::exists(exportedSaved));

    for (const fs::path& output : {saved, exportedSaved}) {
        // Reloading must preserve every alpha byte and keep the watermark detectable
        ImageHandle reloaded = createImageSession(defaultPassword, defaultP, defaultPsnr);
        loadImage(reloaded.get(), output.string(), true);
        const OriginalPixelData result = takeOriginalPixelData(reloaded.get());
        ASSERT_EQ(result.width, original.width);
        ASSERT_EQ(result.height, original.height);
        ASSERT_EQ(result.channels, 4);
        ASSERT_EQ(result.pixels.size(), original.pixels.size());
        for (size_t offset = 3; offset < original.pixels.size(); offset += 4) {
            if (result.pixels[offset] != original.pixels[offset]) {
                ADD_FAILURE() << output << " differs in alpha at pixel " << offset / 4;
                break;
            }
        }
        const float correlation = detectLoadedImage(reloaded.get());
        EXPECT_TRUE(std::isfinite(correlation));
        EXPECT_GT(correlation, 0.5f);
    }
}

#if defined(_USE_EIGEN_)
TEST(EigenDetectionTest, LargePredictionWindowsRemainStableWithOneThread) {
    initializeEnvironment(0);
    const int originalThreads = omp_get_max_threads();
    struct RestoreThreadCount {
        int count;
        ~RestoreThreadCount() { omp_set_num_threads(count); }
    } restore{originalThreads};

    const auto correlationAt = [](const int threads, const int order) {
        omp_set_num_threads(threads);
        auto image = createImageSession(defaultPassword, order, defaultPsnr);
        loadImage(image.get(), "samples/images/4k.png");
        embedImage(image.get());
        prepareDetectionImage(image.get());
        return detectEmbeddedBuffer(image.get());
    };

    for (const int order : {7, 9}) {
        const float singleThread = correlationAt(1, order);
        EXPECT_TRUE(std::isfinite(singleThread)) << "p=" << order;
        EXPECT_GT(singleThread, 0.75f) << "p=" << order;
        if (originalThreads > 1) {
            const float parallel = correlationAt(std::min(originalThreads, 16), order);
            EXPECT_NEAR(singleThread, parallel, 0.01f) << "p=" << order;
        }
    }
}

#endif

#if defined(_USE_GPU_)
TEST(GpuDetectionTest, FourKLargePredictionWindowsRemainDetectable) {
    const auto devices = getAvailableDevices();
    ASSERT_FALSE(devices.empty());
    for (size_t deviceIndex = 0; deviceIndex < devices.size(); ++deviceIndex) {
        ASSERT_TRUE(initializeEnvironment(static_cast<int>(deviceIndex))) << devices[deviceIndex];
        for (const int order : {7, 9}) {
            auto image = createImageSession(defaultPassword, order, defaultPsnr);
            loadImage(image.get(), "samples/images/4k.png");
            embedImage(image.get());
            prepareDetectionImage(image.get());
            const float correlation = detectEmbeddedBuffer(image.get());
            EXPECT_TRUE(std::isfinite(correlation)) << devices[deviceIndex] << " p=" << order;
            EXPECT_GT(correlation, 0.75f) << devices[deviceIndex] << " p=" << order;
        }
    }
}
#endif

// the display buffer is the planar session output interleaved, for sizes with partial GPU tiles and SIMD blocks, the row padding stays untouched
TEST_F(WatermarkTest, PreviewMatchesTheSessionPixelsWithPaddedRows) {
    // the fixture session belongs to device 0, every device gets its own session
    session.reset();
    const auto check = [&](const std::string& device) {
        ImageHandle previewSession = createImageSession(defaultPassword, defaultP, defaultPsnr);
        for (const auto [width, height] : {
                 std::pair{9,  9 },
                 std::pair{13, 11},
                 std::pair{16, 31},
                 std::pair{33, 35},
                 std::pair{70, 47}
        }) {
            for (const int channels : {1, 3}) {
                const std::string name = std::format("{} preview_{}x{}_{}", device, width, height, channels);
                const fs::path input = tempDir / std::format("preview_{}x{}{}", width, height, channels == 3 ? ".bmp" : ".pgm");
                if (channels == 3)
                    writeNoiseBmp(input, width, height);
                else
                    writeNoisePgm(input, width, height);
                loadImage(previewSession.get(), input.string());
                const SessionPixelData source = embedAndRead(previewSession.get());
                ASSERT_EQ(source.channels, channels) << name;
                const PreviewFormat format = getSessionPreviewFormat(previewSession.get());
                EXPECT_EQ(format.width, width) << name;
                EXPECT_EQ(format.height, height) << name;
                EXPECT_EQ(format.channels, channels) << name;

                const size_t stride = static_cast<size_t>(width) * channels + 3;
                const size_t planeSize = static_cast<size_t>(width) * height;
                std::vector<uint8_t> preview(stride * height, 0xA5);
                copySessionPreview(previewSession.get(), preview.data(), stride);
                int mismatches = 0;
                for (int row = 0; row < height; ++row) {
                    for (int col = 0; col < width; ++col)
                        for (int channel = 0; channel < channels; ++channel)
                            mismatches += preview[static_cast<size_t>(row) * stride + col * channels + channel] !=
                                          source.pixels[static_cast<size_t>(channel) * planeSize + static_cast<size_t>(col) * height + row];
                    for (size_t padding = static_cast<size_t>(width) * channels; padding < stride; ++padding)
                        mismatches += preview[static_cast<size_t>(row) * stride + padding] != 0xA5;
                }
                EXPECT_EQ(mismatches, 0) << name;
            }
        }
    };
#if defined(_USE_CUDA_) || defined(_USE_OPENCL_)
    forEachDevice(check);
#else
    check(getDeviceName());
#endif
}

// RGBA previews must match saved pixels while leaving row padding untouched
TEST_F(WatermarkTest, RgbaPreviewMatchesSavedPixelsAndPreservesPadding) {
    constexpr int width = 23, height = 17;
    std::vector<uint8_t> input(width * height * 4);
    for (size_t i = 0; i < input.size(); ++i)
        input[i] = static_cast<uint8_t>((i * 73 + 19) % 256);
    for (size_t i = 3; i < input.size(); i += 4)
        input[i] = std::array<uint8_t, 3>{0, 128, 255}[(i / 4) % 3];
    const auto path = tempDir / "rgba.png";
    png_image image{};
    image.version = PNG_IMAGE_VERSION;
    image.width = width;
    image.height = height;
    image.format = PNG_FORMAT_RGBA;
    ASSERT_TRUE(png_image_write_to_file(&image, path.string().c_str(), 0, input.data(), 0, nullptr));
    loadImage(session.get(), path.string());
    const auto planar = embedAndRead(session.get());
    ASSERT_EQ(getSessionPreviewFormat(session.get()).channels, 4);
    constexpr size_t stride = width * 4 + 7;
    std::vector<uint8_t> preview(stride * height, 0xA5);
    copySessionPreview(session.get(), preview.data(), stride);
    const auto saved = tempDir / "saved-rgba.png";
    saveImageExact(session.get(), saved.string());
    png_image decoded{};
    decoded.version = PNG_IMAGE_VERSION;
    ASSERT_TRUE(png_image_begin_read_from_file(&decoded, saved.string().c_str()));
    decoded.format = PNG_FORMAT_RGBA;
    std::vector<uint8_t> savedPixels(PNG_IMAGE_SIZE(decoded));
    ASSERT_TRUE(png_image_finish_read(&decoded, nullptr, savedPixels.data(), 0, nullptr));
    png_image_free(&decoded);
    int mismatches = 0;
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            const size_t pixel = static_cast<size_t>(row) * width + col;
            for (int channel = 0; channel < 4; ++channel) {
                const uint8_t actual = preview[row * stride + col * 4 + channel];
                mismatches += actual != savedPixels[pixel * 4 + channel];
                mismatches += actual != (channel == 3 ? input[pixel * 4 + 3] : planar.pixels[channel * width * height + col * height + row]);
            }
        }
        for (size_t padding = width * 4; padding < stride; ++padding)
            mismatches += preview[row * stride + padding] != 0xA5;
    }
    EXPECT_EQ(mismatches, 0);
}

// EXIF orientations load upright, matching orientation 1 pixels and preview
// The image is 8x8 blocks of one color, exact in a quality 100 4:4:4 JPEG
TEST_F(WatermarkTest, ExifOrientationsLoadUpright) {
    // the fixture session belongs to device 0, every device gets its own sessions
    session.reset();
    constexpr int width = 40;
    constexpr int height = 72;
    const auto check = [&](const std::string& device) {
        for (const int channels : {1, 3}) {
            std::vector<uint8_t> blockColors(static_cast<size_t>(width / 8) * (height / 8) * channels);
            uint32_t state = 777;
            for (uint8_t& color : blockColors) {
                state = (state * 1664525u) + 1013904223u;
                color = static_cast<uint8_t>(state >> 24);
            }
            std::vector<uint8_t> displayed(static_cast<size_t>(width) * height * channels);
            for (int y = 0; y < height; ++y)
                for (int x = 0; x < width; ++x)
                    for (int channel = 0; channel < channels; ++channel)
                        displayed[(static_cast<size_t>(y) * width + x) * channels + channel] = blockColors[((y / 8) * (width / 8) + (x / 8)) * channels + channel];

            SessionPixelData reference;
            OriginalPixelData referenceOriginal;
            for (int orientation = 1; orientation <= 8; ++orientation) {
                const std::string name = std::format("{} channels={} orientation={}", device, channels, orientation);
                const fs::path input = tempDir / std::format("oriented_{}_{}.jpg", channels, orientation);
                writeOrientedJpeg(input, displayed, width, height, channels, orientation);
                ImageHandle oriented = createImageSession(defaultPassword, defaultP, defaultPsnr);
                loadImage(oriented.get(), input.string(), true);
                OriginalPixelData original = takeOriginalPixelData(oriented.get());
                const SessionPixelData output = embedAndRead(oriented.get());
                ASSERT_EQ(output.width, width) << name;
                ASSERT_EQ(output.height, height) << name;
                ASSERT_EQ(output.channels, channels) << name;
                ASSERT_EQ(original.width, width) << name;
                ASSERT_EQ(original.height, height) << name;
                ASSERT_EQ(original.channels, channels) << name;
                if (orientation == 1) {
                    // gray blocks survive the JPEG exactly (color blocks go through YCbCr)
                    if (channels == 1)
                        EXPECT_TRUE(original.pixels == displayed) << name;
                    reference = output;
                    referenceOriginal = std::move(original);
                    continue;
                }
                EXPECT_TRUE(output.pixels == reference.pixels) << name;
                EXPECT_TRUE(original.pixels == referenceOriginal.pixels) << name;
            }
        }
    };
#if defined(_USE_CUDA_) || defined(_USE_OPENCL_)
    forEachDevice(check);
#else
    check(getDeviceName());
#endif
}

// the EXIF sample loads upright (1280x720), close to 720p.png (it is a quality 95 JPEG of it)
TEST_F(WatermarkTest, ExifSampleLoadsUpright) {
    ImageHandle rotated = createImageSession(defaultPassword, defaultP, defaultPsnr);
    loadImage(rotated.get(), exifImage.string(), true);
    loadImage(session.get(), "samples/images/720p.png", true);
    EXPECT_EQ(getImageDims(rotated.get()), std::make_pair(720, 1280));
    const OriginalPixelData shown = takeOriginalPixelData(rotated.get());
    const OriginalPixelData expected = takeOriginalPixelData(session.get());
    ASSERT_EQ(shown.width, 1280);
    ASSERT_EQ(shown.height, 720);
    ASSERT_EQ(shown.channels, expected.channels);
    ASSERT_EQ(shown.pixels.size(), expected.pixels.size());
    double difference = 0.0;
    for (size_t i = 0; i < shown.pixels.size(); ++i)
        difference += std::abs(static_cast<int>(shown.pixels[i]) - static_cast<int>(expected.pixels[i]));
    EXPECT_LT(difference / static_cast<double>(shown.pixels.size()), 2.0);
}

// images greater than 8 bits per sample are rejected (the 8-bit loaders would truncate them)
TEST_F(WatermarkTest, RejectsHighBitDepthImages) {
    constexpr int width = 24;
    constexpr int height = 16;
    const auto writePng = [&](const fs::path& path, const uint32_t format, const void* pixels, const void* colormap, const uint32_t colormapEntries) {
        png_image image{};
        image.version = PNG_IMAGE_VERSION;
        image.width = width;
        image.height = height;
        image.format = format;
        image.colormap_entries = colormapEntries;
        ASSERT_NE(png_image_write_to_file(&image, path.string().c_str(), 0, pixels, 0, colormap), 0) << image.message;
    };
    const std::vector<uint16_t> deep(static_cast<size_t>(width) * height * 3, 40000);
    const fs::path deepRgb = tempDir / "deep_rgb.png";
    const fs::path deepGray = tempDir / "deep_gray.png";
    writePng(deepRgb, PNG_FORMAT_LINEAR_RGB, deep.data(), nullptr, 0);
    writePng(deepGray, PNG_FORMAT_LINEAR_Y, deep.data(), nullptr, 0);
    EXPECT_THROW(loadImage(session.get(), deepRgb.string()), std::runtime_error);
    EXPECT_THROW(loadImage(session.get(), deepGray.string()), std::runtime_error);

    // 4 colors: a 2-bit palette PNG, loaded as 8-bit RGB
    constexpr std::array<uint8_t, 12> palette{10, 200, 30, 250, 5, 90, 0, 0, 0, 255, 255, 255};
    std::vector<uint8_t> indices(static_cast<size_t>(width) * height);
    for (size_t i = 0; i < indices.size(); ++i)
        indices[i] = static_cast<uint8_t>((i * 7 + i / width) % 4);
    const fs::path palettePng = tempDir / "palette.png";
    writePng(palettePng, PNG_FORMAT_RGB_COLORMAP, indices.data(), palette.data(), 4);
    loadImage(session.get(), palettePng.string(), true);
    const OriginalPixelData original = takeOriginalPixelData(session.get());
    ASSERT_EQ(original.channels, 3);
    std::vector<uint8_t> expected;
    for (const uint8_t index : indices)
        expected.insert(expected.end(), palette.begin() + index * 3, palette.begin() + index * 3 + 3);
    EXPECT_TRUE(original.pixels == expected);
}

// JPEG decoding matches libjpeg reference within +-3 rounding tolerance
// The odd size has partial chroma blocks
TEST_F(WatermarkTest, LoadsJpegVariantsLikeLibjpeg) {
#if defined(_USE_CUDA_)
    constexpr int tolerance = 3;
#else
    constexpr int tolerance = 0;
#endif
    struct Variant {
        const char* name;
        int components;
        int sampling;
        bool progressive;
    };
    constexpr Variant variants[] = {
        {"420",              3, 2, false},
        {"444",              3, 1, false},
        {"gray",             1, 1, false},
        {"progressive_420",  3, 2, true },
        {"progressive_gray", 1, 1, true }
    };
    for (const auto [width, height] : {
             std::pair{101, 77},
             std::pair{128, 64}
    }) {
        for (const Variant& variant : variants) {
            const std::string name = std::format("{}_{}x{}", variant.name, width, height);
            const fs::path path = tempDir / (name + ".jpg");
            writeJpeg(path, jpegTestPattern(width, height, variant.components), width, height, variant.components, variant.sampling, variant.progressive);
            const DecodedJpeg expected = decodeJpeg(path);
#if defined(_USE_CUDA_)
            EXPECT_TRUE(decodesWithNvJpeg(path)) << name;
#endif
            ImageHandle jpegSession = createImageSession(defaultPassword, defaultP, defaultPsnr);
            loadImage(jpegSession.get(), path.string(), true);
            const OriginalPixelData original = takeOriginalPixelData(jpegSession.get());
            ASSERT_EQ(original.width, width) << name;
            ASSERT_EQ(original.height, height) << name;
            ASSERT_EQ(original.channels, variant.components) << name;
            ASSERT_EQ(original.pixels.size(), expected.pixels.size()) << name;
            int maxDifference = 0;
            for (size_t i = 0; i < original.pixels.size(); ++i)
                maxDifference = std::max(maxDifference, std::abs(static_cast<int>(original.pixels[i]) - static_cast<int>(expected.pixels[i])));
            EXPECT_LE(maxDifference, tolerance) << name;
        }
    }
}

// JPEG has no alpha channel: 4 component (CMYK) JPEG is rejected instead of taking K as the alpha
TEST_F(WatermarkTest, RejectsCmykJpeg) {
    constexpr int width = 32;
    constexpr int height = 24;
    const fs::path path = tempDir / "cmyk.jpg";
    writeJpeg(path, jpegTestPattern(width, height, 4), width, height, 4, 1, false);
#if defined(_USE_CUDA_)
    EXPECT_FALSE(decodesWithNvJpeg(path));
#endif
    EXPECT_THROW(loadImage(session.get(), path.string()), std::runtime_error);
}

// JPEG saving (nvJPEG on CUDA, else CImg) with libjpeg's defaults: 4:2:0 color, one component gray, close to the session pixels and detectable after reloading
TEST_F(WatermarkTest, SavesJpegWithLibjpegDefaults) {
    for (const fs::path& input : {colorImage, grayImage}) {
        const std::string name = input.stem().string();
        ImageHandle jpegSession = createImageSession(defaultPassword, defaultP, defaultPsnr);
        loadImage(jpegSession.get(), input.string());
        const SessionPixelData pixels = embedAndRead(jpegSession.get());
        saveImage(jpegSession.get(), (tempDir / (name + ".jpg")).string());
        const fs::path saved = tempDir / (name + "W_ME.jpg");
        const DecodedJpeg decoded = decodeJpeg(saved);
        ASSERT_EQ(decoded.width, pixels.width) << name;
        ASSERT_EQ(decoded.height, pixels.height) << name;
        ASSERT_EQ(decoded.components, pixels.channels) << name;
        EXPECT_EQ(decoded.lumaSampling, pixels.channels == 3 ? 2 : 1) << name;
        // the session pixels are column-major planes
        const size_t planeSize = static_cast<size_t>(pixels.width) * pixels.height;
        double difference = 0.0;
        for (int y = 0; y < pixels.height; ++y)
            for (int x = 0; x < pixels.width; ++x)
                for (int channel = 0; channel < pixels.channels; ++channel) {
                    const int savedValue = decoded.pixels[(static_cast<size_t>(y) * pixels.width + x) * pixels.channels + channel];
                    const int sessionValue = pixels.pixels[channel * planeSize + static_cast<size_t>(x) * pixels.height + y];
                    difference += std::abs(savedValue - sessionValue);
                }
        const double meanDifference = difference / static_cast<double>(planeSize * pixels.channels);
        EXPECT_LT(meanDifference, pixels.channels == 3 ? 2.5 : 0.25) << name;
        ImageHandle reloaded = createImageSession(defaultPassword, defaultP, defaultPsnr);
        loadImage(reloaded.get(), saved.string());
        EXPECT_GT(detectLoadedImage(reloaded.get()), 0.5f) << name;
    }
}

// batch saves run on many threads (they share a few nvJPEG encoders on CUDA): every file is complete and the same
TEST_F(WatermarkTest, SavesJpegsFromSeveralThreadsAtOnce) {
    embedImage(session.get());
    finish();
    constexpr int threads = 6;
    std::vector<std::future<void>> saves;
    for (int i = 0; i < threads; ++i)
        saves.push_back(std::async(std::launch::async, [&, i] { saveImage(session.get(), (tempDir / std::format("parallel_{}.jpg", i)).string()); }));
    for (auto& save : saves)
        save.get();
    const std::vector<uint8_t> first = readBytes(tempDir / "parallel_0W_ME.jpg");
    ASSERT_FALSE(first.empty());
    for (int i = 1; i < threads; ++i)
        EXPECT_TRUE(readBytes(tempDir / std::format("parallel_{}W_ME.jpg", i)) == first) << i;
}

TEST_F(WatermarkTest, RejectsHighBitDepthVideoDetection) {
    VideoSettings settings = makeVideoSettings(shortVideoFixture().string());
    settings.psnr = defaultPsnr;
    settings.encodeOptions.clear();

    VideoHandle video = initVideo(settings);
    EXPECT_THROW(detectVideo(video.get()), std::runtime_error);
}

#if defined(_USE_OPENCL_) || defined(_USE_CUDA_)
namespace {
// the GPU HDR contract (FFmpeg's zscale + tonemap=mobius rendering) on the CPU: R'G'B' of a P010 luma sample and its interpolated 10-bit chroma
std::array<float, 3> hdrReference(const uint16_t y, const std::array<float, 2>& chroma, const video_utils::MobiusParams& mobius) {
    // the tables cover [0, 2], beyond 1 the curves are extended like zimg's
    const auto curve = [](const float* table, const float value) {
        const float position = std::clamp(value, 0.0f, 2.0f) * 1023.0f;
        const int lower = static_cast<int>(position);
        return std::fma(table[std::min(lower + 1, 2046)] - table[lower], position - lower, table[lower]);
    };
    const auto gamma = [&](const float value) { return curve(bt1886LUT, value); };
    const auto pq = [&](const float value) { return curve(pqEotfLUT, value); };
    const float Y = std::fma(static_cast<float>(y >> 6), 1.0f / 876.0f, -64.0f / 876.0f);
    const float Cb = std::fma(chroma[0], 1.0f / 896.0f, -512.0f / 896.0f);
    const float Cr = std::fma(chroma[1], 1.0f / 896.0f, -512.0f / 896.0f);
    std::array<float, 3> rgb{pq(std::fma(1.4746f, Cr, Y)), pq(std::fma(-0.16455f, Cb, std::fma(-0.57135f, Cr, Y))), pq(std::fma(1.8814f, Cb, Y))};
    const float luminance = std::fma(0.2627f, rgb[0], std::fma(0.6780f, rgb[1], 0.0593f * rgb[2]));
    const float desaturation = std::max(luminance - 2.0f, 1.0e-6f) / std::max(luminance, 1.0e-6f);
    for (auto& channel : rgb)
        channel = std::fma(desaturation, luminance - channel, channel);
    const float peak = std::max({rgb[0], rgb[1], rgb[2]});
    const float mapped = peak <= 0.3f ? peak : mobius.k * (peak + mobius.a) / (peak + mobius.b);
    for (auto& channel : rgb)
        channel *= peak > 1.0e-6f ? mapped / peak : 0.0f;
    return {gamma(std::fma(1.6605f, rgb[0], std::fma(-0.5876f, rgb[1], -0.0728f * rgb[2]))), gamma(std::fma(-0.1246f, rgb[0], std::fma(1.1329f, rgb[1], -0.0083f * rgb[2]))),
        gamma(std::fma(-0.0182f, rgb[0], std::fma(-0.1006f, rgb[1], 1.1187f * rgb[2])))};
}
float reference709Luma(const std::array<float, 3>& rgb) { return std::fma(0.2126f, rgb[0], std::fma(0.7152f, rgb[1], 0.0722f * rgb[2])); }
} // namespace

// The GPU HDR conversion matches the CPU transcription of the contract: bilinear chroma at the siting (edges replicated), tonemapping per pixel,
// zimg's 2:1 bilinear chroma filter and rounding. The 34x18 frame has partial GPU tiles
TEST(GpuHdrTest, MatchesTheHdrContractForColoredPixelsPeaksAndChromaSitings) {
    constexpr int width = 34, height = 18, pixels = width * height, chromaWidth = width / 2, chromaHeight = height / 2;
    std::vector<uint16_t> source(pixels * 3 / 2), interleaved(pixels / 2);
    for (int index = 0; index < pixels; ++index)
        source[index] = static_cast<uint16_t>(((index * 37 + index / width * 71) % 1024) << 6);
    for (int index = 0; index < pixels / 4; ++index) {
        source[pixels + index] = interleaved[index * 2] = static_cast<uint16_t>(((index * 113 + 79) % 1024) << 6);
        source[pixels + pixels / 4 + index] = interleaved[index * 2 + 1] = static_cast<uint16_t>(((index * 61 + 257) % 1024) << 6);
    }
    const auto chromaAt = [&](const int x, const int y, const video_utils::ChromaSiting& siting) {
        const float u = (x - siting.x) * 0.5f, v = (y - siting.y) * 0.5f;
        const int u0 = static_cast<int>(std::floor(u)), v0 = static_cast<int>(std::floor(v));
        const auto sample = [&](const int plane, const int cx, const int cy) {
            return static_cast<float>(source[pixels + plane * pixels / 4 + std::clamp(cy, 0, chromaHeight - 1) * chromaWidth + std::clamp(cx, 0, chromaWidth - 1)] >> 6);
        };
        std::array<float, 2> chroma{};
        for (int plane = 0; plane < 2; ++plane) {
            const float top = std::lerp(sample(plane, u0, v0), sample(plane, u0 + 1, v0), u - u0);
            const float bottom = std::lerp(sample(plane, u0, v0 + 1), sample(plane, u0 + 1, v0 + 1), u - u0);
            chroma[plane] = std::lerp(top, bottom, v - v0);
        }
        return chroma;
    };
    const auto weight = [](const int offset, const float site) { return std::max(0.0f, 1.0f - std::abs(offset - site) * 0.5f) * 0.5f; };
    const auto devices = getAvailableDevices();
    for (size_t deviceIndex = 0; deviceIndex < devices.size(); ++deviceIndex) {
        ASSERT_TRUE(initializeEnvironment(static_cast<int>(deviceIndex)));
        SCOPED_TRACE(devices[deviceIndex]);
#if defined(_USE_OPENCL_)
        auto& manager = OclQueueManager::getInstance();
        cl::Buffer input(manager.getContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, source.size() * sizeof(uint16_t), source.data());
        cl::Buffer floatOutput(manager.getContext(), CL_MEM_READ_WRITE, pixels * sizeof(float));
        OclHdrTonemap converter(width, height, manager.getQueueRaw());
#else
        const auto stream = CudaStreamManager::getInstance().getComputeStream();
        CudaArray<uint16_t> ySource(height, width, source.data(), stream), uvSource(height / 2, width, interleaved.data(), stream);
        CudaArray<uint8_t> yOutput(height, width, stream), uvOutput(height / 2, width, stream);
        CudaArray<float> floatOutput(height, width, stream);
#endif
        for (const auto siting : {
                 video_utils::ChromaSiting{0.0f, 0.5f},
                 video_utils::ChromaSiting{0.0f, 0.0f},
                 video_utils::ChromaSiting{0.5f, 0.5f}
        })
            for (const float peak : {1.0f, 4.0f, 10.0f, 40.0f, 100.0f}) {
                SCOPED_TRACE(std::format("siting ({}, {}) peak {}", siting.x, siting.y, peak));
                const auto mobius = video_utils::MobiusParams::fromHdrPeak(peak);
                std::vector<uint8_t> actual(pixels * 3 / 2);
                std::vector<float> luma(pixels);
#if defined(_USE_OPENCL_)
                const auto& output = converter.convert(input, mobius, siting, floatOutput);
                manager.getQueue().enqueueReadBuffer(output, CL_TRUE, 0, actual.size(), actual.data());
                manager.getQueue().enqueueReadBuffer(floatOutput, CL_TRUE, 0, luma.size() * sizeof(float), luma.data());
#else
                cuda_utils::launchP010HdrToSdrKernel(
                    ySource.data(), width * 2, uvSource.data(), width * 2, yOutput.data(), width, floatOutput.data(), uvOutput.data(), width, width, height, siting, mobius, stream);
                yOutput.toHost(actual.data());
                floatOutput.toHost(luma.data());
                std::vector<uint8_t> uv(pixels / 2);
                uvOutput.toHost(uv.data());
                for (int index = 0; index < pixels / 4; ++index) {
                    actual[pixels + index] = uv[index * 2];
                    actual[pixels + pixels / 4 + index] = uv[index * 2 + 1];
                }
#endif
                // reference R'G'B' -> Y and full resolution Cb/Cr of every pixel
                std::vector<float> lumaReference(pixels), cbReference(pixels), crReference(pixels);
                for (int y = 0; y < height; ++y)
                    for (int x = 0; x < width; ++x) {
                        const auto rgb = hdrReference(source[y * width + x], chromaAt(x, y, siting), mobius);
                        const float Y = reference709Luma(rgb);
                        lumaReference[y * width + x] = std::clamp(std::fma(Y, 219.0f, 16.0f), 0.0f, 255.0f);
                        cbReference[y * width + x] = (rgb[2] - Y) * (1.0f / 1.8556f);
                        crReference[y * width + x] = (rgb[0] - Y) * (1.0f / 1.5748f);
                    }
                int maximumByteError = 0;
                float maximumFloatError = 0.0f;
                for (int y = 0; y < height; ++y)
                    for (int x = 0; x < width; ++x) {
                        const float expected = lumaReference[y * width + x];
                        maximumByteError = std::max(maximumByteError, std::abs(static_cast<int>(actual[y * width + x]) - static_cast<int>(std::lround(expected))));
                        ASSERT_TRUE(std::isfinite(luma[x * height + y]));
                        maximumFloatError = std::max(maximumFloatError, std::abs(luma[x * height + y] - expected));
                    }
                for (int cy = 0; cy < chromaHeight; ++cy)
                    for (int cx = 0; cx < chromaWidth; ++cx) {
                        float cb = 0.0f, cr = 0.0f;
                        for (int dy = -1; dy <= 2; ++dy)
                            for (int dx = -1; dx <= 2; ++dx) {
                                const int x = std::clamp(2 * cx + dx, 0, width - 1), y = std::clamp(2 * cy + dy, 0, height - 1);
                                const float tap = weight(dx, siting.x) * weight(dy, siting.y);
                                cb += tap * cbReference[y * width + x];
                                cr += tap * crReference[y * width + x];
                            }
                        const auto expectedU = std::lround(std::clamp(std::fma(cb, 224.0f, 128.0f), 0.0f, 255.0f));
                        const auto expectedV = std::lround(std::clamp(std::fma(cr, 224.0f, 128.0f), 0.0f, 255.0f));
                        maximumByteError = std::max({maximumByteError, std::abs(static_cast<int>(actual[pixels + cy * chromaWidth + cx]) - static_cast<int>(expectedU)),
                            std::abs(static_cast<int>(actual[pixels + pixels / 4 + cy * chromaWidth + cx]) - static_cast<int>(expectedV))});
                    }
                EXPECT_LE(maximumByteError, 1);
                EXPECT_LE(maximumFloatError, 0.02f);
            }
    }
}
#endif

#if defined(_USE_OPENCL_)
// manual timing of the HDR conversion of a 4K frame on every device: --gtest_also_run_disabled_tests --gtest_filter=*HdrConversionTiming*
TEST(OpenCLVideoTest, DISABLED_HdrConversionTiming) {
    constexpr int width = 3840, height = 2160, pixels = width * height;
    std::vector<uint16_t> source(pixels * 3 / 2);
    for (int index = 0; index < pixels * 3 / 2; ++index)
        source[index] = static_cast<uint16_t>((index < pixels ? 300 + (index * 7 + index / width * 3) % 500 : 420 + (index * 5) % 180) << 6);
    const auto devices = getAvailableDevices();
    for (size_t deviceIndex = 0; deviceIndex < devices.size(); ++deviceIndex) {
        ASSERT_TRUE(initializeEnvironment(static_cast<int>(deviceIndex)));
        auto& manager = OclQueueManager::getInstance();
        cl::Buffer input(manager.getContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, source.size() * sizeof(uint16_t), source.data());
        cl::Buffer floatOutput(manager.getContext(), CL_MEM_READ_WRITE, pixels * sizeof(float));
        OclHdrTonemap converter(width, height, manager.getQueueRaw());
        const auto mobius = video_utils::MobiusParams::fromHdrPeak(10.0f);
        for (const bool floatLuma : {false, true}) {
            converter.convert(input, mobius, {0.0f, 0.5f}, floatLuma ? floatOutput : cl::Buffer{});
            manager.getQueue().finish();
            constexpr int runs = 50;
            const auto start = std::chrono::steady_clock::now();
            for (int run = 0; run < runs; ++run)
                converter.convert(input, mobius, {0.0f, 0.5f}, floatLuma ? floatOutput : cl::Buffer{});
            manager.getQueue().finish();
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() / runs;
            std::cout << std::format("{}: {:.3f} ms per 4K frame{}\n", devices[deviceIndex], ms, floatLuma ? " (with float luma)" : "");
        }
    }
}

TEST(OpenCLVideoTest, PacksEncoderPlanesExactlyAndRetainsEarlierOutputSurfaces) {
    using namespace video_utils;
    const auto devices = getAvailableDevices();
    int checked = 0;
    for (size_t index = 0; index < devices.size(); ++index) {
        ASSERT_TRUE(initializeEnvironment(static_cast<int>(index)));
        auto& manager = OclQueueManager::getInstance();
        const auto interop = manager.getVideoInterop();
        if (!interop)
            continue;
        SCOPED_TRACE(devices[index]);
        constexpr int width = 66, height = 34;
        constexpr size_t pixels = width * height;
        AVBufferRefPtr device(interop->createDevice());
        AVBufferRefPtr frames(av_hwframe_ctx_alloc(device.get()));
        ASSERT_TRUE(frames);
        auto* context = reinterpret_cast<AVHWFramesContext*>(frames->data);
        context->format = AV_PIX_FMT_D3D11;
        context->sw_format = AV_PIX_FMT_NV12;
        context->width = width;
        context->height = height;
        static_cast<AVD3D11VAFramesContext*>(context->hwctx)->BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_VIDEO_ENCODER;
        ASSERT_GE(av_hwframe_ctx_init(frames.get()), 0);
        auto output = interop->createOutput(frames.get(), manager.getQueueRaw());
        AVFramePtr source(av_frame_alloc());
        source->format = AV_PIX_FMT_YUV420P;
        source->width = width;
        source->height = height;
        ASSERT_GE(av_frame_get_buffer(source.get(), 64), 0);
        std::vector<uint8_t> planar(pixels * 3 / 2), replacement(pixels);
        for (size_t position = 0; position < replacement.size(); ++position)
            replacement[position] = static_cast<uint8_t>((position * 11 + 19) % 256);
        for (int plane = 0; plane < 3; ++plane) {
            const int planeWidth = plane == 0 ? width : width / 2, planeHeight = plane == 0 ? height : height / 2;
            const size_t offset = plane == 0 ? 0 : pixels + (plane - 1) * pixels / 4;
            for (int row = 0; row < planeHeight; ++row)
                for (int col = 0; col < planeWidth; ++col)
                    source->data[plane][row * source->linesize[plane] + col] = planar[offset + row * planeWidth + col] = static_cast<uint8_t>((row * 13 + col * 7 + plane * 73) % 256);
        }
        cl::Buffer packed(manager.getContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, planar.size(), planar.data());
        cl::Buffer luma(manager.getContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, replacement.size(), replacement.data());
        AVFramePtr first(output->copyFrom(source.get()));
        AVFramePtr modified(output->copyFrom(packed, luma));
        AVFramePtr copied(output->copyFrom(first.get()));
        AVFramePtr uploaded(output->copyFrom(source.get(), luma));
        const std::array<std::pair<const char*, AVFrame*>, 4> outputs{
            {{"CPU upload", first.get()}, {"OpenCL buffer", modified.get()}, {"D3D11 copy", copied.get()}, {"CPU upload with luma", uploaded.get()}}
        };
        for (const auto& [name, frame] : outputs) {
            SCOPED_TRACE(name);
            output->waitUntilCopied(frame);
            AVFramePtr readback(av_frame_alloc());
            ASSERT_GE(av_hwframe_transfer_data(readback.get(), frame, 0), 0);
            ASSERT_EQ(readback->format, AV_PIX_FMT_NV12);
            int lumaError = 0, chromaError = 0;
            for (int row = 0; row < height; ++row)
                for (int col = 0; col < width; ++col) {
                    const size_t offset = row * width + col;
                    const int expected = frame == modified.get() || frame == uploaded.get() ? replacement[offset] : planar[offset];
                    lumaError = std::max(lumaError, std::abs(static_cast<int>(readback->data[0][row * readback->linesize[0] + col]) - expected));
                }
            for (int row = 0; row < height / 2; ++row)
                for (int col = 0; col < width; ++col) {
                    const size_t offset = pixels + (col % 2) * pixels / 4 + row * (width / 2) + col / 2;
                    chromaError = std::max(chromaError, std::abs(static_cast<int>(readback->data[1][row * readback->linesize[1] + col]) - planar[offset]));
                }
            EXPECT_EQ(lumaError, 0);
            EXPECT_EQ(chromaError, 0);
        }
        ++checked;
    }
    if (!checked)
        GTEST_SKIP() << "No OpenCL D3D11-sharing GPU";
}

TEST(OpenCLVideoTest, QsvPackedChromaSurfacesPreservePlanesAndEarlierFrames) {
    using namespace video_utils;
    const auto devices = getAvailableDevices();
    int checked = 0;
    for (size_t index = 0; index < devices.size(); ++index) {
        ASSERT_TRUE(initializeEnvironment(static_cast<int>(index)));
        auto& manager = OclQueueManager::getInstance();
        const auto interop = manager.getVideoInterop();
        if (!interop || manager.getDevice().getInfo<CL_DEVICE_VENDOR_ID>() != 0x8086)
            continue;
        AVBufferRefPtr d3d(interop->createDevice());
        AVBufferRef* rawQsv = nullptr;
        if (av_hwdevice_ctx_create_derived(&rawQsv, AV_HWDEVICE_TYPE_QSV, d3d.get(), 0) < 0)
            continue;
        AVBufferRefPtr qsv(rawQsv);
        for (const auto format : {AV_PIX_FMT_YUV422P, AV_PIX_FMT_YUV444P}) {
            SCOPED_TRACE(std::string(av_get_pix_fmt_name(format)) + " on " + devices[index]);
            constexpr int width = 34, height = 18, pixels = width * height;
            const int chromaWidth = format == AV_PIX_FMT_YUV444P ? width : width / 2;
            const int chromaSamples = chromaWidth * height;
            AVBufferRefPtr frames(av_hwframe_ctx_alloc(qsv.get()));
            auto* fc = reinterpret_cast<AVHWFramesContext*>(frames->data);
            fc->format = AV_PIX_FMT_QSV;
            fc->sw_format = format == AV_PIX_FMT_YUV444P ? AV_PIX_FMT_VUYX : AV_PIX_FMT_YUYV422;
            fc->width = width;
            fc->height = height;
            fc->initial_pool_size = 0;
            if (av_hwframe_ctx_init(frames.get()) < 0)
                continue;
            auto output = interop->createOutput(frames.get(), manager.getQueueRaw());
            AVFramePtr source(av_frame_alloc());
            source->format = format;
            source->width = width;
            source->height = height;
            ASSERT_GE(av_frame_get_buffer(source.get(), 64), 0);
            std::vector<uint8_t> planar(pixels + 2 * chromaSamples), replacement(pixels);
            for (int plane = 0; plane < 3; ++plane)
                for (int row = 0; row < height; ++row)
                    for (int col = 0; col < (plane == 0 ? width : chromaWidth); ++col) {
                        const auto value = static_cast<uint8_t>((plane * 53 + row * 3 + col) % 256);
                        source->data[plane][row * source->linesize[plane] + col] = value;
                        planar[(plane == 0 ? 0 : pixels + (plane - 1) * chromaSamples) + row * (plane == 0 ? width : chromaWidth) + col] = value;
                    }
            for (int pixel = 0; pixel < pixels; ++pixel)
                replacement[pixel] = static_cast<uint8_t>(255 - planar[pixel]);
            cl::Buffer packed(manager.getContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, planar.size(), planar.data());
            cl::Buffer luma(manager.getContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, replacement.size(), replacement.data());
            AVFramePtr first(output->copyFrom(source.get())), modified(output->copyFrom(packed, luma));
            source->format = format == AV_PIX_FMT_YUV444P ? AV_PIX_FMT_YUVJ444P : AV_PIX_FMT_YUVJ422P;
            AVFramePtr uploaded(output->copyFrom(source.get(), luma));
            output->waitUntilCopied(first.get());
            AVFramePtr mapped(av_frame_alloc());
            mapped->format = AV_PIX_FMT_D3D11;
            ASSERT_GE(av_hwframe_map(mapped.get(), first.get(), AV_HWFRAME_MAP_READ | AV_HWFRAME_MAP_DIRECT), 0);
            AVFramePtr copied(output->copyFrom(mapped.get()));
            for (const auto* frame : {first.get(), modified.get(), uploaded.get(), copied.get()}) {
                output->waitUntilCopied(frame);
                AVFramePtr host(av_frame_alloc());
                ASSERT_GE(av_hwframe_transfer_data(host.get(), frame, 0), 0);
                ASSERT_EQ(host->format, fc->sw_format);
                int maximumError = 0;
                for (int row = 0; row < height; ++row)
                    for (int col = 0; col < width; ++col) {
                        const auto* pixel = host->data[0] + row * host->linesize[0] + col * (format == AV_PIX_FMT_YUV444P ? 4 : 2);
                        const int y = format == AV_PIX_FMT_YUV444P ? pixel[2] : pixel[0];
                        const int expectedY = frame == modified.get() || frame == uploaded.get() ? replacement[row * width + col] : planar[row * width + col];
                        maximumError = std::max(maximumError, std::abs(y - expectedY));
                        if (format == AV_PIX_FMT_YUV444P || col % 2 == 0) {
                            const int cx = format == AV_PIX_FMT_YUV444P ? col : col / 2;
                            const int u = pixel[1], v = format == AV_PIX_FMT_YUV444P ? pixel[0] : pixel[3];
                            maximumError = std::max({maximumError, std::abs(u - planar[pixels + row * chromaWidth + cx]), std::abs(v - planar[pixels + chromaSamples + row * chromaWidth + cx])});
                        }
                    }
                EXPECT_EQ(maximumError, 0);
            }
            ++checked;
        }
    }
    if (!checked)
        GTEST_SKIP() << "No QSV packed-chroma frame pool";
}

TEST(OpenCLVideoTest, PackedDecoderTexturesPreserveChromaAndConvertTo420) {
    using namespace video_utils;
    int checked = 0;
    const auto devices = getAvailableDevices();
    for (size_t index = 0; index < devices.size(); ++index) {
        ASSERT_TRUE(initializeEnvironment(static_cast<int>(index)));
        auto& manager = OclQueueManager::getInstance();
        const auto interop = manager.getVideoInterop();
        if (!interop)
            continue;
        AVBufferRefPtr device(interop->createDevice());
        auto* dx = static_cast<AVD3D11VADeviceContext*>(reinterpret_cast<AVHWDeviceContext*>(device->data)->hwctx);
        for (const auto textureFormat : {DXGI_FORMAT_YUY2, DXGI_FORMAT_Y210, DXGI_FORMAT_AYUV, DXGI_FORMAT_Y410}) {
            SCOPED_TRACE(devices[index] + " DXGI " + std::to_string(textureFormat));
            constexpr int width = 34, height = 18, pixels = width * height;
            const bool is422 = textureFormat == DXGI_FORMAT_YUY2 || textureFormat == DXGI_FORMAT_Y210;
            const int cw = is422 ? width / 2 : width;
            const int pitch = width * (textureFormat == DXGI_FORMAT_YUY2 ? 2 : 4);
            std::vector<uint8_t> data(static_cast<size_t>(pitch) * height);
            for (int row = 0; row < height; ++row)
                for (int col = 0; col < width; ++col) {
                    const unsigned y = fixtureSample(0, row, col, 0), u = fixtureSample(1, row, is422 ? col / 2 : col, 0), v = fixtureSample(2, row, is422 ? col / 2 : col, 0);
                    auto* pixel = data.data() + row * pitch;
                    if (textureFormat == DXGI_FORMAT_YUY2) {
                        pixel[col * 2] = static_cast<uint8_t>(y);
                        pixel[col * 2 + 1] = static_cast<uint8_t>(col % 2 ? v : u);
                    } else if (textureFormat == DXGI_FORMAT_Y210) {
                        auto* words = reinterpret_cast<uint16_t*>(pixel);
                        words[col * 2] = static_cast<uint16_t>(y << 8);
                        words[col * 2 + 1] = static_cast<uint16_t>((col % 2 ? v : u) << 8);
                    } else if (textureFormat == DXGI_FORMAT_AYUV) {
                        pixel[col * 4] = static_cast<uint8_t>(v);
                        pixel[col * 4 + 1] = static_cast<uint8_t>(u);
                        pixel[col * 4 + 2] = static_cast<uint8_t>(y);
                        pixel[col * 4 + 3] = 255;
                    } else {
                        reinterpret_cast<uint32_t*>(pixel)[col] = (u << 2) | (y << 12) | (v << 22) | (3u << 30);
                    }
                }
            D3D11_TEXTURE2D_DESC desc{};
            desc.Width = width;
            desc.Height = height;
            desc.MipLevels = desc.ArraySize = desc.SampleDesc.Count = 1;
            desc.Format = textureFormat;
            desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
            D3D11_SUBRESOURCE_DATA initial{data.data(), static_cast<UINT>(pitch), 0};
            ID3D11Texture2D* raw = nullptr;
            if (FAILED(dx->device->CreateTexture2D(&desc, &initial, &raw)))
                continue;
            const auto release = [](ID3D11Texture2D* texture) { texture->Release(); };
            std::unique_ptr<ID3D11Texture2D, decltype(release)> texture(raw, release);
            D3D11_SHADER_RESOURCE_VIEW_DESC view{};
            view.Format = textureFormat == DXGI_FORMAT_Y210 ? DXGI_FORMAT_R16G16B16A16_UINT : textureFormat == DXGI_FORMAT_Y410 ? DXGI_FORMAT_R10G10B10A2_UINT : DXGI_FORMAT_R8G8B8A8_UINT;
            view.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
            view.Texture2D.MipLevels = 1;
            ID3D11ShaderResourceView* supportedView = nullptr;
            if (FAILED(dx->device->CreateShaderResourceView(texture.get(), &view, &supportedView)))
                continue;
            supportedView->Release();
            AVFramePtr frame(av_frame_alloc());
            frame->format = AV_PIX_FMT_D3D11;
            frame->width = width;
            frame->height = height;
            frame->color_trc = AVCOL_TRC_BT709;
            frame->data[0] = reinterpret_cast<uint8_t*>(texture.get());
            for (const bool subsample : {false, true}) {
                const auto format = subsample ? AV_PIX_FMT_YUV420P : is422 ? AV_PIX_FMT_YUV422P : AV_PIX_FMT_YUV444P;
                const int outputWidth = subsample ? width / 2 : cw, outputHeight = subsample ? height / 2 : height;
                auto copy = interop->createFrame(frame.get(), manager.getQueueRaw(), false, format);
                const auto& buffer = copy->copyFrom(frame.get());
                std::vector<uint8_t> actual(pixels + 2 * outputWidth * outputHeight);
                manager.getQueue().enqueueReadBuffer(buffer, CL_TRUE, 0, actual.size(), actual.data());
                int maximumError = 0;
                for (int row = 0; row < height; ++row)
                    for (int col = 0; col < width; ++col)
                        maximumError = std::max(maximumError, std::abs(actual[row * width + col] - fixtureSample(0, row, col, 0)));
                for (int plane = 1; plane < 3; ++plane)
                    for (int row = 0; row < outputHeight; ++row)
                        for (int col = 0; col < outputWidth; ++col) {
                            int sum = 0;
                            const int sx = cw / outputWidth, sy = height / outputHeight;
                            for (int dy = 0; dy < sy; ++dy)
                                for (int dx = 0; dx < sx; ++dx)
                                    sum += fixtureSample(plane, row * sy + dy, col * sx + dx, 0);
                            const int expected = (sum + sx * sy / 2) / (sx * sy);
                            maximumError = std::max(maximumError, std::abs(actual[pixels + (plane - 1) * outputWidth * outputHeight + row * outputWidth + col] - expected));
                        }
                EXPECT_EQ(maximumError, 0);
            }
            ++checked;
        }
    }
    if (!checked)
        GTEST_SKIP() << "No compatible packed-video texture views";
}

TEST_F(WatermarkTest, HeldHardwareDecoderFrameIsNotOverwrittenByLaterFrames) {
    using namespace video_utils;
    session.reset();
    const auto input = tempDir / "decoder-pool.mkv";
    writeVideoFixture(input, false, AVCOL_TRC_BT709, true, "libx264");
    const auto devices = getAvailableDevices();
    int checked = 0;
    for (size_t index = 0; index < devices.size(); ++index) {
        ASSERT_TRUE(initializeEnvironment(static_cast<int>(index)));
        auto settings = makeVideoSettings(input.string());
        settings.useHwDecoder = true;
        auto video = initVideo(settings);
        if (!video->useHwDecoder)
            continue;
        SCOPED_TRACE(devices[index]);
        AVFramePtr frame(av_frame_alloc()), held, before(av_frame_alloc()), after(av_frame_alloc());
        AVPacketPtr packet(av_packet_alloc());
        int frames = 0;
        const auto drain = [&] {
            int result;
            while ((result = avcodec_receive_frame(video->inputDecoderCtx.get(), frame.get())) >= 0) {
                ++frames;
                if (!held) {
                    held.reset(av_frame_clone(frame.get()));
                    ASSERT_TRUE(held);
                    ASSERT_GE(av_hwframe_transfer_data(before.get(), held.get(), 0), 0);
                } else {
                    EXPECT_FALSE(frame->data[0] == held->data[0] && frame->data[1] == held->data[1]);
                }
                av_frame_unref(frame.get());
            }
            EXPECT_TRUE(result == AVERROR(EAGAIN) || result == AVERROR_EOF);
        };
        while (av_read_frame(video->inputFormatCtx.get(), packet.get()) >= 0) {
            if (packet->stream_index == video->videoStreamIndex) {
                ASSERT_GE(avcodec_send_packet(video->inputDecoderCtx.get(), packet.get()), 0);
                drain();
            }
            av_packet_unref(packet.get());
        }
        ASSERT_GE(avcodec_send_packet(video->inputDecoderCtx.get(), nullptr), 0);
        drain();
        ASSERT_EQ(frames, 100);
        ASSERT_TRUE(held);
        ASSERT_GE(av_hwframe_transfer_data(after.get(), held.get(), 0), 0);
        for (int plane = 0; plane < 2; ++plane)
            for (int row = 0; row < (plane == 0 ? held->height : held->height / 2); ++row)
                ASSERT_EQ(std::memcmp(before->data[plane] + row * before->linesize[plane], after->data[plane] + row * after->linesize[plane], held->width), 0);
        ++checked;
    }
    if (!checked)
        GTEST_SKIP() << "No D3D11 hardware decoder";
}

TEST_F(WatermarkTest, HardwareVideoEncodersHandleGpuAndCpuDecodeAndWatermarkIntervals) {
    session.reset();
    const auto input = tempDir / "encode-source.mkv";
    writeVideoFixture(input, false, AVCOL_TRC_BT709, true, "libx264");
    const auto devices = getAvailableDevices();
    for (size_t index = 0; index < devices.size(); ++index) {
        ASSERT_TRUE(initializeEnvironment(static_cast<int>(index)));
        auto& manager = OclQueueManager::getInstance();
        if (!manager.getVideoInterop())
            continue;
        const auto vendor = manager.getDevice().getInfo<CL_DEVICE_VENDOR_ID>();
        if (vendor != 0x8086 && vendor != 0x1002 && vendor != 0x10de)
            continue;
        const auto encoders = vendor == 0x8086   ? std::vector<std::string>{"h264_qsv", "hevc_qsv", "av1_qsv"}
                              : vendor == 0x1002 ? std::vector<std::string>{"h264_amf", "hevc_amf"}
                                                 : std::vector<std::string>{"h264_nvenc", "hevc_nvenc"};
        for (const auto& name : encoders)
            for (const bool hardwareDecode : {false, true}) {
                SCOPED_TRACE(devices[index] + " " + name + (hardwareDecode ? " GPU decode" : " CPU decode"));
                const auto output = tempDir / std::format("{}-{}-{}.mkv", index, name, hardwareDecode);
                auto settings = makeVideoSettings(input.string());
                settings.useHwDecoder = hardwareDecode;
                settings.useHwEncoder = true;
                settings.watermarkInterval = 7;
                settings.encodeOptions = "-c:v " + name + " -g 30 -b:v 8M";
                settings.encodeOutputPath = output.string();
                auto video = initVideo(settings);
                ASSERT_EQ(embedVideo(video.get()), 100);
                ASSERT_TRUE(video->openclVideoOutput);
                ASSERT_EQ(video->outputEncoderCtx->pix_fmt, vendor == 0x8086 ? AV_PIX_FMT_QSV : AV_PIX_FMT_D3D11);
                video.reset();
                EXPECT_EQ(firstVideoLuma(output).second, AVCOL_TRC_BT709);
                settings.videoFile = output.string();
                settings.useHwEncoder = false;
                settings.useHwDecoder = false;
                auto detection = initVideo(settings);
                int frames = 0;
                auto correlations = capturedCorrelations(detection.get(), frames);
                EXPECT_EQ(frames, 100);
                ASSERT_EQ(correlations.size(), 15);
                std::sort(correlations.begin(), correlations.end());
                EXPECT_GT(correlations[correlations.size() / 2], 0.5f);
            }
    }
}

TEST_F(WatermarkTest, HardwareEncoderDefaultsConvertTenBitAndHdrEvenOnSkippedFrames) {
    session.reset();
    const auto devices = getAvailableDevices();
    for (const auto transfer : {AVCOL_TRC_BT709, AVCOL_TRC_SMPTE2084}) {
        const auto input = tempDir / std::format("encode-tenbit-{}.mkv", static_cast<int>(transfer));
        writeVideoFixture(input, true, transfer, true, "libx265");
        for (size_t index = 0; index < devices.size(); ++index) {
            ASSERT_TRUE(initializeEnvironment(static_cast<int>(index)));
            auto& manager = OclQueueManager::getInstance();
            if (!manager.getVideoInterop())
                continue;
            const auto vendor = manager.getDevice().getInfo<CL_DEVICE_VENDOR_ID>();
            if (vendor != 0x8086 && vendor != 0x1002 && vendor != 0x10de)
                continue;
            SCOPED_TRACE(devices[index] + " transfer " + std::to_string(transfer));
            auto settings = makeVideoSettings(input.string());
            settings.useHwDecoder = settings.useHwEncoder = true;
            settings.watermarkInterval = 7;
            settings.encodeOptions.clear();
            settings.encodeOutputPath = (tempDir / std::format("encoded-{}-{}.mkv", index, static_cast<int>(transfer))).string();
            auto video = initVideo(settings);
            ASSERT_EQ(embedVideo(video.get()), 100);
            ASSERT_TRUE(video->openclVideoOutput);
            const auto* encoder = video->outputEncoderCtx.get();
            if (vendor == 0x10de) {
                EXPECT_STREQ(encoder->codec->name, "hevc_nvenc");
                EXPECT_EQ(encoder->pix_fmt, AV_PIX_FMT_D3D11);
            } else if (vendor == 0x8086) {
                EXPECT_STREQ(encoder->codec->name, "hevc_qsv");
                EXPECT_EQ(encoder->flags & AV_CODEC_FLAG_QSCALE, 0);
                EXPECT_EQ(encoder->global_quality, 20);
                EXPECT_EQ(encoder->max_b_frames, -1);
            } else {
                EXPECT_STREQ(encoder->codec->name, "hevc_amf");
                EXPECT_EQ(encoder->gop_size, 120);
                int64_t qpI = 0, qpP = 0, quality = 0;
                ASSERT_GE(av_opt_get_int(encoder->priv_data, "qp_i", 0, &qpI), 0);
                ASSERT_GE(av_opt_get_int(encoder->priv_data, "qp_p", 0, &qpP), 0);
                ASSERT_GE(av_opt_get_int(encoder->priv_data, "quality", 0, &quality), 0);
                EXPECT_EQ(qpI, 22);
                EXPECT_EQ(qpP, 22);
                EXPECT_EQ(quality, 5); // AMF balanced preset
            }
            video.reset();
            EXPECT_EQ(firstVideoLuma(settings.encodeOutputPath).second, AVCOL_TRC_BT709);
            settings.videoFile = settings.encodeOutputPath;
            settings.useHwEncoder = settings.useHwDecoder = false;
            auto detection = initVideo(settings);
            int frames = 0;
            const auto correlations = capturedCorrelations(detection.get(), frames);
            EXPECT_EQ(frames, 100);
            EXPECT_EQ(correlations.size(), 15);
        }
    }
}

TEST_F(WatermarkTest, HardwareEncoderCannotSilentlyUseAnotherGpuOrOverwriteOutputOnFailure) {
    session.reset();
    const auto output = tempDir / "existing.mkv";
    const std::string sentinel = "preserve existing output";
    {
        std::ofstream stream(output, std::ios::binary);
        stream << sentinel;
    }
    const auto devices = getAvailableDevices();
    for (size_t index = 0; index < devices.size(); ++index) {
        ASSERT_TRUE(initializeEnvironment(static_cast<int>(index)));
        const auto vendor = OclQueueManager::getInstance().getDevice().getInfo<CL_DEVICE_VENDOR_ID>();
        if (vendor != 0x8086 && vendor != 0x1002 && vendor != 0x10de)
            continue;
        auto settings = makeVideoSettings(shortVideoFixture().string());
        settings.useHwEncoder = true;
        settings.encodeOutputPath = output.string();
        settings.encodeOptions = vendor == 0x8086 ? "-c:v hevc_amf" : "-c:v hevc_qsv";
        auto video = initVideo(settings);
        EXPECT_THROW(embedVideo(video.get()), std::runtime_error);
        video.reset();
        const auto bytes = readBytes(output);
        EXPECT_EQ(std::string(bytes.begin(), bytes.end()), sentinel);
        settings.encodeOptions = "-c:v h264_mf -hw_encoding 0";
        video = initVideo(settings);
        EXPECT_THROW(embedVideo(video.get()), std::runtime_error);
    }
}

TEST(OpenCLVideoTest, CopiesNv12PlanesWithoutChangingTheSourceSurface) {
    using namespace video_utils;
    const auto devices = getAvailableDevices();
    size_t checked = 0;
    for (size_t deviceIndex = 0; deviceIndex < devices.size(); ++deviceIndex) {
        ASSERT_TRUE(initializeEnvironment(static_cast<int>(deviceIndex)));
        auto& manager = OclQueueManager::getInstance();
        const auto& interop = manager.getVideoInterop();
        if (!interop)
            continue;
        SCOPED_TRACE(devices[deviceIndex]);
        constexpr int width = 322, height = 182;
        constexpr size_t lumaBytes = width * height;
        AVBufferRefPtr device(interop->createDevice());
        ASSERT_NE(device, nullptr);
        AVBufferRefPtr frames(av_hwframe_ctx_alloc(device.get()));
        ASSERT_NE(frames, nullptr);
        auto* context = reinterpret_cast<AVHWFramesContext*>(frames->data);
        context->format = AV_PIX_FMT_D3D11;
        context->sw_format = AV_PIX_FMT_NV12;
        context->width = width;
        context->height = height;
        context->initial_pool_size = 2;
        ASSERT_GE(av_hwframe_ctx_init(frames.get()), 0);
        AVFramePtr hardware(av_frame_alloc()), source(av_frame_alloc()), readback(av_frame_alloc());
        ASSERT_GE(av_hwframe_get_buffer(frames.get(), hardware.get(), 0), 0);
        auto copy = interop->createFrame(hardware.get(), manager.getQueueRaw());
        source->format = AV_PIX_FMT_NV12;
        source->width = width;
        source->height = height;
        ASSERT_GE(av_frame_get_buffer(source.get(), 32), 0);
        for (int generation = 0; generation < 3; ++generation) {
            std::vector<uint8_t> expected(lumaBytes * 3 / 2);
            for (int row = 0; row < height; ++row)
                for (int col = 0; col < width; ++col) {
                    const uint8_t value = static_cast<uint8_t>(row * 7 + col * 3 + generation * 17);
                    source->data[0][row * source->linesize[0] + col] = value;
                    expected[row * width + col] = value;
                }
            for (int row = 0; row < height / 2; ++row)
                for (int col = 0; col < width / 2; ++col) {
                    const uint8_t u = static_cast<uint8_t>(row * 3 + col * 11 + generation * 19);
                    const uint8_t v = static_cast<uint8_t>(row * 13 + col * 5 + generation * 23);
                    source->data[1][row * source->linesize[1] + col * 2] = u;
                    source->data[1][row * source->linesize[1] + col * 2 + 1] = v;
                    expected[lumaBytes + row * (width / 2) + col] = u;
                    expected[lumaBytes + lumaBytes / 4 + row * (width / 2) + col] = v;
                }
            ASSERT_GE(av_hwframe_transfer_data(hardware.get(), source.get(), 0), 0);
            std::vector<uint8_t> actual(expected.size());
            manager.getQueue().enqueueReadBuffer(copy->copyFrom(hardware.get()), CL_TRUE, 0, actual.size(), actual.data());
            EXPECT_EQ(actual, expected);
            av_frame_unref(readback.get());
            ASSERT_GE(av_hwframe_transfer_data(readback.get(), hardware.get(), 0), 0);
            for (int plane = 0; plane < 2; ++plane)
                for (int row = 0; row < (plane == 0 ? height : height / 2); ++row)
                    ASSERT_EQ(std::memcmp(source->data[plane] + row * source->linesize[plane], readback->data[plane] + row * readback->linesize[plane], width), 0);
        }
        ++checked;
    }
    if (checked == 0)
        GTEST_SKIP() << "No OpenCL D3D11-sharing GPU is available";
}

TEST(OpenCLVideoTest, StagedFramesKeepTheirContentAndOrder) {
    using namespace video_utils;
    const auto devices = getAvailableDevices();
    size_t checked = 0;
    for (size_t deviceIndex = 0; deviceIndex < devices.size(); ++deviceIndex) {
        ASSERT_TRUE(initializeEnvironment(static_cast<int>(deviceIndex)));
        auto& manager = OclQueueManager::getInstance();
        const auto& interop = manager.getVideoInterop();
        if (!interop)
            continue;
        SCOPED_TRACE(devices[deviceIndex]);
        constexpr int width = 322, height = 182, framesHeld = 3;
        constexpr size_t lumaBytes = width * height;
        AVBufferRefPtr device(interop->createDevice());
        AVBufferRefPtr frames(av_hwframe_ctx_alloc(device.get()));
        ASSERT_NE(frames, nullptr);
        auto* context = reinterpret_cast<AVHWFramesContext*>(frames->data);
        context->format = AV_PIX_FMT_D3D11;
        context->sw_format = AV_PIX_FMT_NV12;
        context->width = width;
        context->height = height;
        context->initial_pool_size = framesHeld;
        // decoder surfaces, like the frames a D3D11 decoder stages
        static_cast<AVD3D11VAFramesContext*>(context->hwctx)->BindFlags = D3D11_BIND_DECODER;
        ASSERT_GE(av_hwframe_ctx_init(frames.get()), 0);
        AVFramePtr source(av_frame_alloc());
        source->format = AV_PIX_FMT_NV12;
        source->width = width;
        source->height = height;
        ASSERT_GE(av_frame_get_buffer(source.get(), 32), 0);
        const auto upload = [&](AVFrame* hardware, const int generation) {
            for (int plane = 0; plane < 2; ++plane)
                for (int row = 0; row < (plane == 0 ? height : height / 2); ++row)
                    for (int col = 0; col < width; ++col)
                        source->data[plane][row * source->linesize[plane] + col] = static_cast<uint8_t>(row * 5 + col * 3 + plane * 41 + generation * 29);
            ASSERT_GE(av_hwframe_transfer_data(hardware, source.get(), 0), 0);
        };
        const auto expected = [&](const int generation) {
            std::vector<uint8_t> packed(lumaBytes * 3 / 2);
            for (int row = 0; row < height; ++row)
                for (int col = 0; col < width; ++col)
                    packed[row * width + col] = static_cast<uint8_t>(row * 5 + col * 3 + generation * 29);
            for (int row = 0; row < height / 2; ++row)
                for (int col = 0; col < width; ++col)
                    packed[lumaBytes + (col % 2) * lumaBytes / 4 + row * (width / 2) + col / 2] = static_cast<uint8_t>(row * 5 + col * 3 + 41 + generation * 29);
            return packed;
        };
        std::array<AVFramePtr, framesHeld> hardware;
        for (int index = 0; index < framesHeld; ++index) {
            hardware[index].reset(av_frame_alloc());
            ASSERT_GE(av_hwframe_get_buffer(frames.get(), hardware[index].get(), 0), 0);
            upload(hardware[index].get(), index);
        }
        auto copy = interop->createFrame(hardware[0].get(), manager.getQueueRaw());
        for (const auto& frame : hardware)
            copy->stage(frame.get());
        // staging copied the surfaces: overwriting them afterwards does not change the queued frames
        for (int index = 0; index < framesHeld; ++index)
            upload(hardware[index].get(), index + framesHeld);
        EXPECT_THROW(copy->copyFrom(hardware[1].get()), std::logic_error);
        for (int index = 0; index < framesHeld; ++index) {
            std::vector<uint8_t> actual(lumaBytes * 3 / 2);
            manager.getQueue().enqueueReadBuffer(copy->copyFrom(hardware[index].get()), CL_TRUE, 0, actual.size(), actual.data());
            EXPECT_EQ(actual, expected(index)) << "frame " << index;
        }
        // without staging, copyFrom copies the current surface
        std::vector<uint8_t> actual(lumaBytes * 3 / 2);
        manager.getQueue().enqueueReadBuffer(copy->copyFrom(hardware[1].get()), CL_TRUE, 0, actual.size(), actual.data());
        EXPECT_EQ(actual, expected(1 + framesHeld));
        ++checked;
    }
    if (checked == 0)
        GTEST_SKIP() << "No OpenCL D3D11-sharing GPU is available";
}

TEST(OpenCLVideoTest, ConvertsP010PlanesWithoutChangingTheSourceSurface) {
    using namespace video_utils;
    size_t checked = 0;
    const auto devices = getAvailableDevices();
    for (size_t deviceIndex = 0; deviceIndex < devices.size(); ++deviceIndex) {
        ASSERT_TRUE(initializeEnvironment(static_cast<int>(deviceIndex)));
        auto& manager = OclQueueManager::getInstance();
        const auto& interop = manager.getVideoInterop();
        if (!interop)
            continue;
        SCOPED_TRACE(devices[deviceIndex]);
        constexpr int width = 322, height = 182, allocationWidth = 352, allocationHeight = 192;
        constexpr size_t lumaBytes = width * height;
        AVBufferRefPtr device(interop->createDevice());
        ASSERT_NE(device, nullptr);
        AVBufferRefPtr frames(av_hwframe_ctx_alloc(device.get()));
        auto* context = reinterpret_cast<AVHWFramesContext*>(frames->data);
        context->format = AV_PIX_FMT_D3D11;
        context->sw_format = AV_PIX_FMT_P010;
        context->width = allocationWidth;
        context->height = allocationHeight;
        context->initial_pool_size = 2;
        static_cast<AVD3D11VAFramesContext*>(context->hwctx)->BindFlags = D3D11_BIND_DECODER;
        if (av_hwframe_ctx_init(frames.get()) < 0)
            continue;
        AVFramePtr firstSlice(av_frame_alloc()), hardware(av_frame_alloc()), source(av_frame_alloc()), readback(av_frame_alloc());
        ASSERT_GE(av_hwframe_get_buffer(frames.get(), firstSlice.get(), 0), 0);
        ASSERT_GE(av_hwframe_get_buffer(frames.get(), hardware.get(), 0), 0);
        source->format = AV_PIX_FMT_P010;
        source->width = allocationWidth;
        source->height = allocationHeight;
        ASSERT_GE(av_frame_get_buffer(source.get(), 32), 0);
        hardware->width = width;
        hardware->height = height;
        auto copy = interop->createFrame(hardware.get(), manager.getQueueRaw());
        auto rawCopy = interop->createFrame(hardware.get(), manager.getQueueRaw(), true);
        for (const auto matrix : {AVCOL_SPC_BT709, AVCOL_SPC_SMPTE170M, AVCOL_SPC_BT2020_NCL}) {
            for (const bool fullRange : {false, true}) {
                SCOPED_TRACE(std::format("matrix={} full={}", static_cast<int>(matrix), fullRange));
                for (int generation = 0; generation < 3; ++generation) {
                    for (int plane = 0; plane < 2; ++plane)
                        for (int row = 0; row < (plane == 0 ? allocationHeight : allocationHeight / 2); ++row)
                            for (int col = 0; col < allocationWidth; ++col) {
                                const int value = fullRange ? (row * 17 + col * 13 + generation * 73) % 1024 : 64 + (row * 17 + col * 13 + generation * 73) % (plane == 0 ? 877 : 897);
                                reinterpret_cast<uint16_t*>(source->data[plane] + row * source->linesize[plane])[col] = static_cast<uint16_t>(value << 6);
                            }
                    hardware->width = allocationWidth;
                    hardware->height = allocationHeight;
                    ASSERT_GE(av_hwframe_transfer_data(hardware.get(), source.get(), 0), 0);
                    hardware->width = width;
                    hardware->height = height;
                    hardware->colorspace = matrix;
                    hardware->color_range = fullRange ? AVCOL_RANGE_JPEG : AVCOL_RANGE_MPEG;
                    hardware->color_trc = AVCOL_TRC_BT709;
                    std::vector<uint8_t> actual(lumaBytes * 3 / 2);
                    manager.getQueue().enqueueReadBuffer(copy->copyFrom(hardware.get()), CL_TRUE, 0, actual.size(), actual.data());
                    std::vector<uint16_t> raw(lumaBytes * 3 / 2);
                    manager.getQueue().enqueueReadBuffer(rawCopy->copyFrom(hardware.get()), CL_TRUE, 0, raw.size() * sizeof(uint16_t), raw.data());
                    std::array<int, 3> maximumError{};
                    for (int plane = 0; plane < 3; ++plane) {
                        const int planeWidth = plane == 0 ? width : width / 2;
                        const int planeHeight = plane == 0 ? height : height / 2;
                        const size_t offset = plane == 0 ? 0 : lumaBytes + (plane - 1) * lumaBytes / 4;
                        for (int row = 0; row < planeHeight; ++row)
                            for (int col = 0; col < planeWidth; ++col) {
                                const auto* line = reinterpret_cast<const uint16_t*>(source->data[plane == 0 ? 0 : 1] + row * source->linesize[plane == 0 ? 0 : 1]);
                                const int input = line[plane == 0 ? col : col * 2 + plane - 1] >> 6;
                                ASSERT_EQ(raw[offset + row * planeWidth + col], line[plane == 0 ? col : col * 2 + plane - 1]);
                                const int expected = std::min(255, (input + 2) / 4);
                                maximumError[plane] = std::max(maximumError[plane], std::abs(static_cast<int>(actual[offset + row * planeWidth + col]) - expected));
                            }
                        EXPECT_EQ(maximumError[plane], 0) << "plane " << plane;
                    }
                    hardware->width = allocationWidth;
                    hardware->height = allocationHeight;
                    av_frame_unref(readback.get());
                    ASSERT_GE(av_hwframe_transfer_data(readback.get(), hardware.get(), 0), 0);
                    for (int plane = 0; plane < 2; ++plane)
                        for (int row = 0; row < (plane == 0 ? allocationHeight : allocationHeight / 2); ++row)
                            ASSERT_EQ(std::memcmp(source->data[plane] + row * source->linesize[plane], readback->data[plane] + row * readback->linesize[plane], allocationWidth * 2), 0);
                    ++checked;
                }
            }
        }
    }
    if (checked == 0)
        GTEST_SKIP() << "No OpenCL D3D11-sharing GPU supports P010 input";
}

TEST_F(WatermarkTest, OpenCLHardwareVideoDecodeEmbedsAndDetectsWithCpuFallback) {
    session.reset();
    const auto input = tempDir / "h264.mkv";
    writeVideoFixture(input, false, AVCOL_TRC_BT709, true, "libx264");
    const auto devices = getAvailableDevices();
    for (size_t deviceIndex = 0; deviceIndex < devices.size(); ++deviceIndex) {
        ASSERT_TRUE(initializeEnvironment(static_cast<int>(deviceIndex)));
        SCOPED_TRACE(devices[deviceIndex]);
        auto settings = makeVideoSettings(input.string());
        settings.useHwDecoder = true;
        settings.watermarkInterval = 3;
        settings.encodeOptions = "-c:v ffv1";
        settings.encodeOutputPath = (tempDir / std::format("watermarked-{}.mkv", deviceIndex)).string();
        {
            auto session = initVideo(settings);
            if (OclQueueManager::getInstance().getVideoInterop())
                EXPECT_TRUE(session->useHwDecoder);
            else
                EXPECT_FALSE(session->useHwDecoder);
            ASSERT_EQ(embedVideo(session.get()), 100);
        }
        settings.videoFile = settings.encodeOutputPath;
        settings.encodeOutputPath.clear();
        auto detected = initVideo(settings);
        EXPECT_FALSE(detected->useHwDecoder) << "FFV1 must fall back to software decoding";
        int frameCount = 0;
        const auto correlations = capturedCorrelations(detected.get(), frameCount);
        ASSERT_EQ(frameCount, 100);
        ASSERT_EQ(correlations.size(), 34U);
        auto sorted = correlations;
        std::sort(sorted.begin(), sorted.end());
        EXPECT_GT(sorted[sorted.size() / 2], 0.5f);
        detected.reset();
        settings.useHwDecoder = false;
        auto software = initVideo(settings);
        const auto baseline = capturedCorrelations(software.get(), frameCount);
        EXPECT_EQ(correlations, baseline);
        software.reset();
        settings.videoFile = input.string();
        settings.useHwDecoder = true;
        auto hardware = initVideo(settings);
        const auto hardwareCorrelations = capturedCorrelations(hardware.get(), frameCount);
        hardware.reset();
        settings.useHwDecoder = false;
        auto reference = initVideo(settings);
        const auto softwareCorrelations = capturedCorrelations(reference.get(), frameCount);
        ASSERT_EQ(hardwareCorrelations.size(), softwareCorrelations.size());
        for (size_t index = 0; index < hardwareCorrelations.size(); ++index)
            EXPECT_NEAR(hardwareCorrelations[index], softwareCorrelations[index], 1.0e-5f);
    }
}
TEST_F(WatermarkTest, OpenCLHardwareDecoderHandlesHevcAndAv1) {
    session.reset();
    for (const char* encoderName : {"libx265", "libsvtav1"}) {
        const auto input = tempDir / (std::string(encoderName) + ".mkv");
        writeVideoFixture(input, false, AVCOL_TRC_BT709, true, encoderName);
        const auto devices = getAvailableDevices();
        for (size_t deviceIndex = 0; deviceIndex < devices.size(); ++deviceIndex) {
            ASSERT_TRUE(initializeEnvironment(static_cast<int>(deviceIndex)));
            SCOPED_TRACE(std::string(encoderName) + " on " + devices[deviceIndex]);
            auto settings = makeVideoSettings(input.string());
            settings.useHwDecoder = true;
            settings.watermarkInterval = 10;
            auto hardware = initVideo(settings);
            if (hardware->useHwDecoder)
                EXPECT_EQ(hardware->inputDecoderCtx->pix_fmt, AV_PIX_FMT_D3D11);
            int hardwareFrames = 0;
            const auto actual = capturedCorrelations(hardware.get(), hardwareFrames);
            hardware.reset();
            settings.useHwDecoder = false;
            auto software = initVideo(settings);
            int softwareFrames = 0;
            const auto expected = capturedCorrelations(software.get(), softwareFrames);
            EXPECT_EQ(hardwareFrames, 100);
            EXPECT_EQ(hardwareFrames, softwareFrames);
            ASSERT_EQ(actual.size(), expected.size());
            bool saved = false;
            for (size_t index = 0; index < actual.size(); ++index) {
                if (std::abs(actual[index] - expected[index]) > 1.0e-4f && !saved) {
                    const auto directory = fs::temp_directory_path() / ("watermarking-decode-correlation-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
                    fs::create_directories(directory);
                    fs::copy_file(input, directory / "input.mkv");
                    std::ofstream details(directory / "details.txt");
                    details << devices[deviceIndex] << '\n'
                            << OclQueueManager::getInstance().getDevice().getInfo<CL_DRIVER_VERSION>() << '\n'
                            << "Frame " << index * settings.watermarkInterval << "\nHardware " << std::setprecision(9) << actual[index] << "\nSoftware " << expected[index] << '\n';
                    ADD_FAILURE() << "Decode correlation mismatch; original bitstream retained at " << directory.string();
                    saved = true;
                }
                EXPECT_NEAR(actual[index], expected[index], 1.0e-4f);
            }
        }
    }
}

// check pixels before detection to isolate failures to decoder, copies or reductions
TEST_F(WatermarkTest, OpenCLAv1StagedLumaMatchesSoftwarePixelsAndDetectionIsRepeatable) {
    using namespace video_utils;
    session.reset();
    const auto input = tempDir / "av1-pixels.mkv";
    writeVideoFixture(input, false, AVCOL_TRC_BT709, true, "libsvtav1");
    const auto reference = decodeVideoPlanes(input);
    ASSERT_EQ(reference.frames.size(), 100);
    const auto devices = getAvailableDevices();
    int checked = 0;
    for (size_t deviceIndex = 0; deviceIndex < devices.size(); ++deviceIndex) {
        ASSERT_TRUE(initializeEnvironment(static_cast<int>(deviceIndex)));
        SCOPED_TRACE(devices[deviceIndex]);
        auto settings = makeVideoSettings(input.string());
        settings.useHwDecoder = true;
        auto video = initVideo(settings);
        if (!video->useHwDecoder)
            continue;
        auto& manager = OclQueueManager::getInstance();
        ASSERT_GT(video->decodeLookahead, 0U);
        ImageBuffer luma(reference.height, reference.width, manager.getQueueRaw());
        std::deque<AVFramePtr> held;
        AVFramePtr frame(av_frame_alloc());
        AVPacketPtr packet(av_packet_alloc());
        size_t count = 0;
        bool saved = false;
        const auto consume = [&] {
            const auto& packed = video->openclVideoFrame->copyFrom(held.front().get());
            std::vector<uint8_t> pixels(static_cast<size_t>(reference.width) * reference.height);
            manager.getQueue().enqueueReadBuffer(packed, CL_TRUE, 0, pixels.size(), pixels.data());
            ASSERT_LT(count, reference.frames.size());
            const auto& expected = reference.frames[count][0];
            if (pixels != expected && !saved) {
                const auto directory = fs::temp_directory_path() / ("watermarking-av1-mismatch-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
                fs::create_directories(directory);
                fs::copy_file(input, directory / "input.mkv");
                const auto write = [&](const char* name, const std::vector<uint8_t>& data) {
                    std::ofstream stream(directory / name, std::ios::binary);
                    stream.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
                };
                write("software.y", expected);
                write("staged.y", pixels);
                AVFramePtr direct(av_frame_alloc());
                if (av_hwframe_transfer_data(direct.get(), held.front().get(), 0) >= 0) {
                    std::vector<uint8_t> decoded(pixels.size());
                    for (int y = 0; y < reference.height; ++y)
                        std::memcpy(decoded.data() + y * reference.width, direct->data[0] + y * direct->linesize[0], reference.width);
                    write("hardware.y", decoded);
                }
                std::ofstream details(directory / "details.txt");
                details << devices[deviceIndex] << '\n'
                        << manager.getDevice().getInfo<CL_DRIVER_VERSION>() << '\n'
                        << "Frame " << count << ", size " << reference.width << 'x' << reference.height << '\n';
                ADD_FAILURE() << "Decoded/staged AV1 luma differs from software. Artifacts: " << directory.string();
                saved = true;
            }
            EXPECT_TRUE(pixels == expected) << "frame " << count;
            cl_utils::launchPitchedToFloat(packed, luma.clBuffer(), reference.width, reference.height, reference.width, manager.getQueue());
            const float first = video->watermarkObj->detectWatermark(luma);
            const float second = video->watermarkObj->detectWatermark(luma);
            // AMD driver can change float partial by an ulp when D3D11 shares GPU
            EXPECT_NEAR(first, second, 1.0e-4f) << "Repeated detection on identical luma, frame " << count;
            ++count;
        };
        const auto receive = [&] {
            int status;
            while ((status = avcodec_receive_frame(video->inputDecoderCtx.get(), frame.get())) >= 0) {
                AVFramePtr owned(av_frame_clone(frame.get()));
                ASSERT_TRUE(owned);
                held.push_back(std::move(owned));
                if (held.size() > video->decodeLookahead) {
                    consume();
                    held.pop_front();
                }
                video->openclVideoFrame->stage(held.back().get(), true);
                av_frame_unref(frame.get());
            }
            EXPECT_TRUE(status == AVERROR(EAGAIN) || status == AVERROR_EOF);
        };
        while (av_read_frame(video->inputFormatCtx.get(), packet.get()) >= 0) {
            if (packet->stream_index == video->videoStreamIndex) {
                ASSERT_GE(avcodec_send_packet(video->inputDecoderCtx.get(), packet.get()), 0);
                receive();
            }
            av_packet_unref(packet.get());
        }
        ASSERT_GE(avcodec_send_packet(video->inputDecoderCtx.get(), nullptr), 0);
        receive();
        for (; !held.empty(); held.pop_front())
            consume();
        EXPECT_EQ(count, reference.frames.size());
        ++checked;
    }
    if (!checked)
        GTEST_SKIP() << "No AV1 D3D11/OpenCL decoder";
}

// unsupported devices retain frames through software fallback
TEST_F(WatermarkTest, OpenCLHardwareHdrToneMapsAllFramesAndPreservesDetectionIntervals) {
    session.reset();
    const auto input = tempDir / "hdr-main10.mkv";
    writeVideoFixture(input, true, AVCOL_TRC_SMPTE2084, true, "libx265");
    const auto devices = getAvailableDevices();
    for (size_t deviceIndex = 0; deviceIndex < devices.size(); ++deviceIndex) {
        ASSERT_TRUE(initializeEnvironment(static_cast<int>(deviceIndex)));
        SCOPED_TRACE(devices[deviceIndex]);
        auto settings = makeVideoSettings(input.string());
        settings.useHwDecoder = true;
        settings.watermarkInterval = 3;
        settings.encodeOptions = "-c:v ffv1";
        settings.encodeOutputPath = (tempDir / std::format("sdr-{}.mkv", deviceIndex)).string();
        auto video = initVideo(settings);
        if (OclQueueManager::getInstance().getVideoInterop()) {
            EXPECT_TRUE(video->useHwDecoder);
            EXPECT_TRUE(video->isHdr);
        }
        EXPECT_THROW(detectVideo(video.get()), std::runtime_error);
        ASSERT_EQ(embedVideo(video.get()), 100);
        video.reset();
        video_utils::AVFramePtr pixels;
        const auto [sample, transfer] = firstVideoLuma(settings.encodeOutputPath, &pixels);
        EXPECT_EQ(transfer, AVCOL_TRC_BT709);
        EXPECT_EQ(pixels->color_primaries, AVCOL_PRI_BT709);
        EXPECT_EQ(pixels->colorspace, AVCOL_SPC_BT709);
        EXPECT_EQ(pixels->color_range, AVCOL_RANGE_MPEG);
        EXPECT_EQ(av_frame_get_side_data(pixels.get(), AV_FRAME_DATA_MASTERING_DISPLAY_METADATA), nullptr);
        EXPECT_EQ(av_frame_get_side_data(pixels.get(), AV_FRAME_DATA_CONTENT_LIGHT_LEVEL), nullptr);
        settings.videoFile = settings.encodeOutputPath;
        settings.encodeOutputPath.clear();
        auto detection = initVideo(settings);
        int frames = 0;
        auto correlations = capturedCorrelations(detection.get(), frames);
        EXPECT_EQ(frames, 100);
        ASSERT_EQ(correlations.size(), 34U);
        std::sort(correlations.begin(), correlations.end());
        EXPECT_GT(correlations[correlations.size() / 2], 0.5f);
    }
}

TEST_F(WatermarkTest, OpenCLUnsupportedHdrFormatsKeepSoftwareToneMapping) {
    session.reset();
    for (const auto transfer : {AVCOL_TRC_ARIB_STD_B67, AVCOL_TRC_SMPTE2084}) {
        const auto input = tempDir / std::format("hdr-fallback-{}.mkv", static_cast<int>(transfer));
        const auto range = transfer == AVCOL_TRC_SMPTE2084 ? AVCOL_RANGE_JPEG : AVCOL_RANGE_MPEG;
        writeVideoFixture(input, true, transfer, false, nullptr, AV_PIX_FMT_NONE, range);
        auto settings = makeVideoSettings(input.string());
        settings.useHwDecoder = true;
        settings.encodeOptions = "-c:v ffv1";
        settings.encodeOutputPath = (tempDir / std::format("fallback-sdr-{}.mkv", static_cast<int>(transfer))).string();
        auto video = initVideo(settings);
        EXPECT_FALSE(video->useHwDecoder);
        EXPECT_FALSE(video->isHdr);
        ASSERT_EQ(embedVideo(video.get()), 1);
        video.reset();
        EXPECT_EQ(firstVideoLuma(settings.encodeOutputPath).second, AVCOL_TRC_BT709);
    }
}

TEST_F(WatermarkTest, OpenCLTenBitSdrConversionMatchesSoftwareOrFallsBack) {
    session.reset();
    for (const char* encoderName : {"libx265", "libsvtav1"}) {
        SCOPED_TRACE(encoderName);
        const auto input = tempDir / (std::string(encoderName) + "-main10.mkv");
        writeVideoFixture(input, true, AVCOL_TRC_BT709, true, encoderName);
        const auto devices = getAvailableDevices();
        for (size_t deviceIndex = 0; deviceIndex < devices.size(); ++deviceIndex) {
            ASSERT_TRUE(initializeEnvironment(static_cast<int>(deviceIndex)));
            SCOPED_TRACE(devices[deviceIndex]);
            std::array<int, 2> samples{};
            for (const bool hardware : {false, true}) {
                const auto output = tempDir / std::format("sdr10-{}-{}.mkv", deviceIndex, hardware);
                auto settings = makeVideoSettings(input.string());
                settings.useHwDecoder = hardware;
                settings.encodeOptions = "-c:v ffv1";
                settings.encodeOutputPath = output.string();
                auto video = initVideo(settings);
                if (video->useHwDecoder)
                    EXPECT_EQ(video->decodedFormat, AV_PIX_FMT_P010);
                ASSERT_EQ(embedVideo(video.get()), 100);
                video.reset();
                const auto [sample, transfer] = firstVideoLuma(output);
                EXPECT_EQ(transfer, AVCOL_TRC_BT709);
                samples[hardware ? 1 : 0] = sample;
            }
            EXPECT_NEAR(samples[0], samples[1], 2);
        }
    }
}
#endif

#if defined(_USE_GPU_)
// GPU HDR tonemapping renders like the CPU chain (FFmpeg zscale + tonemap=mobius with BT.2020 desaturation): the frames that are not watermarked
// agree within the resampling and rounding differences of the two implementations
TEST_F(WatermarkTest, GpuHdrRenderingMatchesTheCpuChain) {
    session.reset();
    const auto input = tempDir / "hdr-agreement.mkv";
    writeVideoFixture(input, true, AVCOL_TRC_SMPTE2084, true, "libx265");
    const auto devices = getAvailableDevices();
    for (size_t deviceIndex = 0; deviceIndex < devices.size(); ++deviceIndex) {
        SCOPED_TRACE(devices[deviceIndex]);
        ASSERT_TRUE(initializeEnvironment(static_cast<int>(deviceIndex)));
        std::array<DecodedVideo, 2> outputs;
        bool gpuDecoded = true;
        for (const bool hardware : {false, true}) {
            auto settings = makeVideoSettings(input.string());
            settings.useHwDecoder = hardware;
            settings.watermarkInterval = 1000;
            settings.encodeOptions = "-c:v ffv1";
            settings.encodeOutputPath = (tempDir / std::format("hdr-agreement-{}-{}.mkv", deviceIndex, hardware)).string();
            auto video = initVideo(settings);
            gpuDecoded = !hardware || video->useHwDecoder;
            if (!gpuDecoded)
                break;
            // force reference session through FFmpeg CPU chain
            if (!hardware) {
                ASSERT_FALSE(video->useHwDecoder);
                video->isHdr = false;
            } else {
                ASSERT_TRUE(video->isHdr);
            }
            ASSERT_EQ(embedVideo(video.get()), 100);
            bool cpuTonemap = false;
            if (video->filterGraph)
                for (unsigned index = 0; index < video->filterGraph->nb_filters; ++index)
                    cpuTonemap |= std::string_view(video->filterGraph->filters[index]->filter->name) == "tonemap";
            ASSERT_EQ(cpuTonemap, !hardware) << "The reference must run FFmpeg tonemap; the GPU case must not";
            video.reset();
            outputs[hardware ? 1 : 0] = decodeVideoPlanes(settings.encodeOutputPath);
        }
        if (!gpuDecoded)
            continue;
        ASSERT_EQ(outputs[0].frames.size(), 100);
        ASSERT_EQ(outputs[1].frames.size(), 100);
        for (int plane = 0; plane < 3; ++plane) {
            int maximumError = 0;
            double totalError = 0.0;
            size_t samples = 0, beyondTwo = 0;
            for (size_t index = 1; index < outputs[0].frames.size(); ++index)
                for (size_t sample = 0; sample < outputs[0].frames[index][plane].size(); ++sample) {
                    const int error = std::abs(outputs[0].frames[index][plane][sample] - outputs[1].frames[index][plane][sample]);
                    maximumError = std::max(maximumError, error);
                    totalError += error;
                    beyondTwo += error > 2;
                    ++samples;
                }
            const double meanError = totalError / static_cast<double>(samples), beyondTwoShare = static_cast<double>(beyondTwo) / static_cast<double>(samples);
            std::cout << std::format("{} plane {}: max {}, mean {:.3f}, beyond 2: {} of {}\n", devices[deviceIndex], plane, maximumError, meanError, beyondTwo, samples);
            // measured on a real 4K HDR10 clip: mean below 0.5 and 0.3% beyond 2. Steep edges of colors outside the gamut differ most (approximated powers)
            EXPECT_LE(meanError, 0.2) << "plane " << plane;
            EXPECT_LE(beyondTwoShare, 1.0e-2) << "plane " << plane;
            EXPECT_LE(maximumError, 24) << "plane " << plane;
        }
    }
}

// CPU-decoded PQ HDR (FFV1) is uploaded and tonemapped by the GPU kernels: identical output to the same samples decoded in hardware (lossless HEVC)
TEST_F(WatermarkTest, GpuTonemapsCpuDecodedHdrLikeHardwareDecodedHdr) {
    session.reset();
    const auto hevc = tempDir / "hdr-lossless.mkv", ffv1 = tempDir / "hdr-ffv1.mkv";
    writeVideoFixture(hevc, true, AVCOL_TRC_SMPTE2084, true, "libx265", AV_PIX_FMT_NONE, AVCOL_RANGE_MPEG,
        {
            {"preset",      "ultrafast"         },
            {"x265-params", "lossless=1:pools=2"}
    });
    writeVideoFixture(ffv1, true, AVCOL_TRC_SMPTE2084, true);
    const auto devices = getAvailableDevices();
    for (size_t deviceIndex = 0; deviceIndex < devices.size(); ++deviceIndex) {
        SCOPED_TRACE(devices[deviceIndex]);
        ASSERT_TRUE(initializeEnvironment(static_cast<int>(deviceIndex)));
        std::array<DecodedVideo, 2> outputs;
        bool hardwareDecoded = true;
        for (const bool cpu : {false, true}) {
            auto settings = makeVideoSettings((cpu ? ffv1 : hevc).string());
            settings.useHwDecoder = true;
            settings.watermarkInterval = 7;
            settings.encodeOptions = "-c:v ffv1";
            settings.encodeOutputPath = (tempDir / std::format("hdr-gpu-{}-{}.mkv", deviceIndex, cpu)).string();
            auto video = initVideo(settings);
            EXPECT_TRUE(video->isHdr);
            if (cpu)
                EXPECT_FALSE(video->useHwDecoder);
            else
                hardwareDecoded = video->useHwDecoder;
            ASSERT_EQ(embedVideo(video.get()), 100);
            video.reset();
            outputs[cpu ? 1 : 0] = decodeVideoPlanes(settings.encodeOutputPath);
        }
        if (!hardwareDecoded)
            continue;
        ASSERT_EQ(outputs[0].frames.size(), 100);
        ASSERT_EQ(outputs[1].frames.size(), 100);
        for (size_t index = 0; index < outputs[0].frames.size(); ++index)
            for (int plane = 0; plane < 3; ++plane)
                EXPECT_TRUE(outputs[0].frames[index][plane] == outputs[1].frames[index][plane]) << "frame " << index << " plane " << plane;
    }
}

// GPU decoding keeps 4:2:2/4:4:4 chroma like CPU decoding (or falls back to it): exact chroma and skipped-frame luma with a lossless encoder
TEST_F(WatermarkTest, HardwareDecoderKeepsSdr422And444ChromaOrFallsBack) {
    session.reset();
    constexpr int interval = 10;
    const auto devices = getAvailableDevices();
    for (const auto format : {AV_PIX_FMT_YUV422P, AV_PIX_FMT_YUV444P, AV_PIX_FMT_YUV422P10LE, AV_PIX_FMT_YUV444P10LE}) {
        const auto* descriptor = av_pix_fmt_desc_get(format);
        const bool deep = descriptor->comp[0].depth > 8;
        const auto input = tempDir / std::format("hevc-chroma-{}.mkv", static_cast<int>(format));
        writeVideoFixture(input, deep, AVCOL_TRC_BT709, true, "libx265", format, AVCOL_RANGE_MPEG,
            {
                {"preset",      "ultrafast"         },
                {"x265-params", "lossless=1:pools=2"}
        },
            true);
        for (size_t deviceIndex = 0; deviceIndex < devices.size(); ++deviceIndex) {
            SCOPED_TRACE(std::string(av_get_pix_fmt_name(format)) + " on " + devices[deviceIndex]);
            ASSERT_TRUE(initializeEnvironment(static_cast<int>(deviceIndex)));
            for (const bool hardware : {false, true}) {
                SCOPED_TRACE(hardware ? "GPU decode" : "CPU decode");
                const auto output = tempDir / std::format("hevc-chroma-out-{}-{}-{}.mkv", static_cast<int>(format), deviceIndex, hardware);
                auto settings = makeVideoSettings(input.string());
                settings.useHwDecoder = hardware;
                settings.watermarkInterval = interval;
                settings.encodeOptions = "-c:v ffv1";
                settings.encodeOutputPath = output.string();
                auto video = initVideo(settings);
                if (!hardware)
                    ASSERT_FALSE(video->useHwDecoder);
                ASSERT_EQ(embedVideo(video.get()), 100);
                video.reset();
                EXPECT_EQ(firstVideoLuma(output).second, AVCOL_TRC_BT709);
                const auto decoded = decodeVideoPlanes(output);
                EXPECT_EQ(av_pix_fmt_desc_get(decoded.format)->log2_chroma_w, descriptor->log2_chroma_w);
                EXPECT_EQ(av_pix_fmt_desc_get(decoded.format)->log2_chroma_h, descriptor->log2_chroma_h);
                expectFixturePlanes(decoded, interval, deep ? 1 : 0);
            }
        }
    }
}
#endif

#if defined(_USE_CUDA_)
// Exercise Blackwell's 4:2:2 surface layouts without requiring an NVDEC engine that can decode 4:2:2: NV12 for a 4:2:0 output, NV16 to keep the chroma
TEST(CudaVideoTest, Converts422SurfacesToNv12OrNv16OnTheGpu) {
    using namespace video_utils;
    ASSERT_TRUE(initializeEnvironment(0));
    AVBufferRef* rawDevice = nullptr;
    AVDictionary* options = nullptr;
    av_dict_set(&options, "current_ctx", "1", 0);
    const int status = av_hwdevice_ctx_create(&rawDevice, AV_HWDEVICE_TYPE_CUDA, "0", options, 0);
    av_dict_free(&options);
    AVBufferRefPtr device(rawDevice);
    ASSERT_GE(status, 0);
    for (const auto format : {AV_PIX_FMT_NV16, AV_PIX_FMT_P210LE, AV_PIX_FMT_P216LE}) {
        SCOPED_TRACE(av_get_pix_fmt_name(format));
        constexpr int width = 320, height = 180;
        const bool deep = format != AV_PIX_FMT_NV16;
        AVBufferRefPtr frames(av_hwframe_ctx_alloc(device.get()));
        ASSERT_TRUE(frames);
        auto* context = reinterpret_cast<AVHWFramesContext*>(frames->data);
        context->format = AV_PIX_FMT_CUDA;
        context->sw_format = format;
        context->width = width;
        context->height = height;
        ASSERT_GE(av_hwframe_ctx_init(frames.get()), 0);
        AVFramePtr source(av_frame_alloc()), hardware(av_frame_alloc()), converted(av_frame_alloc()), result(av_frame_alloc());
        ASSERT_TRUE(source && hardware && converted && result);
        source->format = format;
        source->width = width;
        source->height = height;
        ASSERT_GE(av_frame_get_buffer(source.get(), 32), 0);
        for (int plane = 0; plane < 2; ++plane)
            for (int row = 0; row < height; ++row)
                for (int col = 0; col < width; ++col) {
                    const int value = plane == 0 ? 64 + ((row + col) % 128) : (col % 2 == 0 ? 128 : 96);
                    auto* line = source->data[plane] + row * source->linesize[plane];
                    if (deep)
                        reinterpret_cast<uint16_t*>(line)[col] = static_cast<uint16_t>((value * 4 + col % 4) << 6);
                    else
                        line[col] = static_cast<uint8_t>(value);
                }
        ASSERT_GE(av_hwframe_get_buffer(frames.get(), hardware.get(), 0), 0);
        ASSERT_GE(av_hwframe_transfer_data(hardware.get(), source.get(), 0), 0);
        hardware->pts = 0;
        AVFormatContextPtr input(avformat_alloc_context());
        ASSERT_TRUE(input);
        auto* stream = avformat_new_stream(input.get(), nullptr);
        ASSERT_NE(stream, nullptr);
        stream->time_base = {1, 30};
        for (const auto [output, surface] : {
                 std::pair{AV_PIX_FMT_YUV420P, AV_PIX_FMT_NV12},
                 std::pair{AV_PIX_FMT_YUV422P, AV_PIX_FMT_NV16}
        }) {
            SCOPED_TRACE(av_get_pix_fmt_name(surface));
            VideoSession probe;
            probe.videoStream = stream;
            probe.useHwDecoder = true;
            probe.decodedFormat = format;
            probe.outputFormat = output;
            probe.inputDecoderCtx.reset(avcodec_alloc_context3(nullptr));
            ASSERT_TRUE(probe.inputDecoderCtx);
            auto* decoder = probe.inputDecoderCtx.get();
            decoder->width = width;
            decoder->height = height;
            decoder->pix_fmt = AV_PIX_FMT_CUDA;
            decoder->sw_pix_fmt = format;
            decoder->sample_aspect_ratio = {1, 1};
            decoder->hw_frames_ctx = av_buffer_ref(frames.get());
            // an NV16 surface already is the 4:2:2 output layout
            if (format == surface) {
                EXPECT_FALSE(initFilterGraph(&probe));
                continue;
            }
            ASSERT_TRUE(initFilterGraph(&probe));
            av_frame_unref(converted.get());
            av_frame_unref(result.get());
            ASSERT_GE(av_buffersrc_add_frame_flags(probe.buffersrcCtx, hardware.get(), AV_BUFFERSRC_FLAG_KEEP_REF), 0);
            ASSERT_GE(av_buffersink_get_frame(probe.buffersinkCtx, converted.get()), 0);
            ASSERT_GE(av_hwframe_transfer_data(result.get(), converted.get(), 0), 0);
            ASSERT_EQ(result->format, surface);
            for (int plane = 0; plane < 2; ++plane) {
                int maximumError = 0;
                for (int row = 0; row < (plane == 0 || surface == AV_PIX_FMT_NV16 ? height : height / 2); ++row)
                    for (int col = 0; col < width; ++col) {
                        const int expected = plane == 0 ? 64 + ((row + col) % 128) : (col % 2 == 0 ? 128 : 96);
                        maximumError = std::max(maximumError, std::abs(static_cast<int>(result->data[plane][row * result->linesize[plane] + col]) - expected));
                    }
                EXPECT_LE(maximumError, 1) << "plane " << plane;
            }
        }
    }
}

#endif

#if defined(_USE_GPU_)
TEST_F(WatermarkTest, HardwareDecoderRejectsLosslessH264EvenWith420PixelsOrAnIntraProfile) {
    session.reset();
    const auto devices = getAvailableDevices();
    for (size_t deviceIndex = 0; deviceIndex < devices.size(); ++deviceIndex) {
        ASSERT_TRUE(initializeEnvironment(static_cast<int>(deviceIndex)));
        SCOPED_TRACE(devices[deviceIndex]);
        for (const bool intra : {false, true}) {
            const auto input = tempDir / std::format("h264-lossless-{}.mkv", intra);
            writeVideoFixture(input, false, AVCOL_TRC_BT709, true, "libx264", intra ? AV_PIX_FMT_YUV444P : AV_PIX_FMT_YUV420P, AVCOL_RANGE_MPEG,
                {
                    {"qp", "0"               },
                    {"g",  intra ? "1" : "30"}
            });
            auto settings = makeVideoSettings(input.string());
            settings.useHwDecoder = true;
            settings.watermarkInterval = 10;
            auto video = initVideo(settings);
            ASSERT_FALSE(video->useHwDecoder);
            EXPECT_EQ(video->videoStream->codecpar->profile & ~AV_PROFILE_H264_INTRA, AV_PROFILE_H264_HIGH_444_PREDICTIVE);
            int frames = 0;
            const auto correlations = capturedCorrelations(video.get(), frames);
            EXPECT_EQ(frames, 100);
            EXPECT_EQ(correlations.size(), 10);
        }
    }
}
#endif

#if defined(_USE_CUDA_)
TEST_F(WatermarkTest, CpuDecodedFramesReachNvencThroughCudaSurfaces) {
    // NVDEC misdecodes the High 4:4:4 Predictive profile of lossless H.264, these frames are decoded on the CPU
    const auto input = tempDir / "nvenc-cpu-decode.mkv";
    writeVideoFixture(input, false, AVCOL_TRC_BT709, true, "libx264", AV_PIX_FMT_YUV420P, AVCOL_RANGE_MPEG,
        {
            {"qp", "0"}
    },
        true);
    auto settings = makeVideoSettings(input.string());
    settings.useHwDecoder = settings.useHwEncoder = true;
    settings.watermarkInterval = 7;
    settings.encodeOptions = "-c:v hevc_nvenc -tune lossless";
    settings.encodeOutputPath = (tempDir / "nvenc-cpu-decode-out.mkv").string();
    auto video = initVideo(settings);
    ASSERT_FALSE(video->useHwDecoder);
    ASSERT_EQ(embedVideo(video.get()), 100);
    EXPECT_EQ(video->outputEncoderCtx->pix_fmt, AV_PIX_FMT_CUDA);
    video.reset();
    // lossless NVENC keeps every chroma sample and the luma of the frames that are not watermarked
    const auto source = decodeVideoPlanes(input).frames, encoded = decodeVideoPlanes(settings.encodeOutputPath).frames;
    ASSERT_EQ(source.size(), 100);
    ASSERT_EQ(encoded.size(), 100);
    for (size_t index = 0; index < source.size(); ++index) {
        SCOPED_TRACE(std::format("frame {}", index));
        EXPECT_EQ(encoded[index][1], source[index][1]);
        EXPECT_EQ(encoded[index][2], source[index][2]);
        if (index % settings.watermarkInterval == 0)
            EXPECT_NE(encoded[index][0], source[index][0]);
        else
            EXPECT_EQ(encoded[index][0], source[index][0]);
    }
    settings.videoFile = settings.encodeOutputPath;
    settings.useHwEncoder = false;
    auto detection = initVideo(settings);
    int frames = 0;
    auto correlations = capturedCorrelations(detection.get(), frames);
    EXPECT_EQ(frames, 100);
    ASSERT_EQ(correlations.size(), 15);
    std::sort(correlations.begin(), correlations.end());
    EXPECT_GT(correlations[correlations.size() / 2], 0.5f);
}
#endif

// Software encoders keep the SDR chroma resolution: 8-bit and 10-bit sources give exact 8-bit chroma (the 10-bit fixture samples are multiples of 4),
// skipped frames keep their luma, the range survives and detection reads the 4:2:2/4:4:4 luma without converting it
TEST_F(WatermarkTest, SoftwareEncodersKeepSdrChromaResolution) {
    constexpr int interval = 7;
    for (const auto format : {AV_PIX_FMT_YUV420P, AV_PIX_FMT_YUV422P, AV_PIX_FMT_YUV444P, AV_PIX_FMT_YUV420P10LE, AV_PIX_FMT_YUV422P10LE, AV_PIX_FMT_YUV444P10LE})
        for (const auto range : {AVCOL_RANGE_MPEG, AVCOL_RANGE_JPEG}) {
            const auto* descriptor = av_pix_fmt_desc_get(format);
            const bool deep = descriptor->comp[0].depth > 8;
            SCOPED_TRACE(std::format("{} {}", av_get_pix_fmt_name(format), av_color_range_name(range)));
            const auto input = tempDir / std::format("chroma-{}-{}.mkv", static_cast<int>(format), static_cast<int>(range));
            const auto output = tempDir / std::format("chroma-out-{}-{}.mkv", static_cast<int>(format), static_cast<int>(range));
            writeVideoFixture(input, deep, AVCOL_TRC_BT709, true, nullptr, format, range, {}, true);
            auto settings = makeVideoSettings(input.string());
            settings.watermarkInterval = interval;
            settings.encodeOptions = "-c:v ffv1";
            settings.encodeOutputPath = output.string();
            auto video = initVideo(settings);
            ASSERT_EQ(embedVideo(video.get()), 100);
            video.reset();
            const auto decoded = decodeVideoPlanes(output);
            EXPECT_EQ(av_pix_fmt_desc_get(decoded.format)->log2_chroma_w, descriptor->log2_chroma_w);
            EXPECT_EQ(av_pix_fmt_desc_get(decoded.format)->log2_chroma_h, descriptor->log2_chroma_h);
            EXPECT_EQ(decoded.range, range);
            expectFixturePlanes(decoded, interval, deep ? 1 : 0);
            settings.videoFile = output.string();
            auto detection = initVideo(settings);
            int frames = 0;
            auto correlations = capturedCorrelations(detection.get(), frames);
            EXPECT_EQ(frames, 100);
            ASSERT_EQ(correlations.size(), 15);
            std::sort(correlations.begin(), correlations.end());
            EXPECT_GT(correlations[correlations.size() / 2], 0.5f);
        }
}

// an encoder that cannot keep the chroma resolution gets 4:2:0 and the conversion is reported
TEST_F(WatermarkTest, EncodersWithoutTheSourceChromaReportTheConversion) {
    const auto input = tempDir / "chroma-444.mkv";
    writeVideoFixture(input, false, AVCOL_TRC_BT709, true, nullptr, AV_PIX_FMT_YUV444P, AVCOL_RANGE_MPEG, {}, true);
    auto settings = makeVideoSettings(input.string());
    settings.watermarkInterval = 50;
    settings.encodeOptions = "-c:v libsvtav1 -preset 12";
    settings.encodeOutputPath = (tempDir / "chroma-444-av1.mkv").string();
    auto video = initVideo(settings);
    testing::internal::CaptureStdout();
    const int frames = embedVideo(video.get());
    const std::string log = testing::internal::GetCapturedStdout();
    EXPECT_EQ(frames, 100);
    EXPECT_NE(log.find("Encoder 'libsvtav1' cannot keep this chroma resolution: the source chroma (yuv444p) is converted to yuv420p."), std::string::npos) << log;
    video.reset();
    EXPECT_EQ(decodeVideoPlanes(settings.encodeOutputPath).format, AV_PIX_FMT_YUV420P);
}

// odd dimensions need an output format without subsampling in that direction, detection reads only luma and accepts them
TEST_F(WatermarkTest, OddDimensionsNeedAMatchingChromaLayout) {
    constexpr int interval = 7;
    for (const auto format : {AV_PIX_FMT_YUV444P, AV_PIX_FMT_YUV420P}) {
        SCOPED_TRACE(av_get_pix_fmt_name(format));
        const auto input = tempDir / std::format("odd-{}.mkv", static_cast<int>(format));
        const auto output = tempDir / std::format("odd-out-{}.mkv", static_cast<int>(format));
        writeVideoFixture(input, false, AVCOL_TRC_BT709, true, nullptr, format, AVCOL_RANGE_MPEG, {}, true, {321, 181});
        auto settings = makeVideoSettings(input.string());
        settings.watermarkInterval = interval;
        settings.encodeOptions = "-c:v ffv1";
        settings.encodeOutputPath = output.string();
        auto video = initVideo(settings);
        if (format == AV_PIX_FMT_YUV420P) {
            EXPECT_THROW(embedVideo(video.get()), std::runtime_error);
            EXPECT_FALSE(fs::exists(output));
        } else {
            ASSERT_EQ(embedVideo(video.get()), 100);
            video.reset();
            const auto decoded = decodeVideoPlanes(output);
            EXPECT_EQ(decoded.format, AV_PIX_FMT_YUV444P);
            EXPECT_EQ(decoded.width, 321);
            EXPECT_EQ(decoded.height, 181);
            expectFixturePlanes(decoded, interval);
        }
        settings.encodeOutputPath.clear();
        auto detection = initVideo(settings);
        int frames = 0;
        const auto correlations = capturedCorrelations(detection.get(), frames);
        EXPECT_EQ(frames, 100);
        EXPECT_EQ(correlations.size(), 15);
    }
}

#if defined(_USE_GPU_)
// 4:2:2 and 4:4:4 sources with the default hardware encoder: NVENC CUDA surfaces keep the chroma when the GPU encodes it (a failed open retries with 4:2:0),
// D3D11/QSV tries YUY2/AYUV surfaces, every required conversion is reported
TEST_F(WatermarkTest, HardwareEncodingKeepsOrReportsTheChroma) {
    session.reset();
    const auto devices = getAvailableDevices();
    for (size_t deviceIndex = 0; deviceIndex < devices.size(); ++deviceIndex) {
        ASSERT_TRUE(initializeEnvironment(static_cast<int>(deviceIndex)));
        SCOPED_TRACE(devices[deviceIndex]);
        for (const auto format : {AV_PIX_FMT_YUV444P, AV_PIX_FMT_YUV422P, AV_PIX_FMT_YUV444P10LE, AV_PIX_FMT_YUV422P10LE}) {
            SCOPED_TRACE(av_get_pix_fmt_name(format));
            const auto input = tempDir / std::format("hardware-{}.mkv", static_cast<int>(format));
            writeVideoFixture(input, av_pix_fmt_desc_get(format)->comp[0].depth > 8, AVCOL_TRC_BT709, true, nullptr, format, AVCOL_RANGE_MPEG, {}, true);
            auto settings = makeVideoSettings(input.string());
            settings.useHwEncoder = true;
            settings.watermarkInterval = 7;
            settings.encodeOptions.clear();
            settings.encodeOutputPath = (tempDir / std::format("hardware-{}-out.mkv", static_cast<int>(format))).string();
            auto video = initVideo(settings);
            testing::internal::CaptureStdout();
            const int frames = embedVideo(video.get());
            const std::string log = testing::internal::GetCapturedStdout();
            EXPECT_EQ(frames, 100);
            const AVPixelFormat output = video->outputFormat;
            video.reset();
            EXPECT_EQ(decodeVideoPlanes(settings.encodeOutputPath).format, output);
            if (output == AV_PIX_FMT_YUV420P)
                EXPECT_NE(log.find(std::format(": the source chroma ({}) is converted to yuv420p.", av_get_pix_fmt_name(format))), std::string::npos) << log;
            else
                EXPECT_EQ(output, av_pix_fmt_desc_get(format)->log2_chroma_w == 0 ? AV_PIX_FMT_YUV444P : AV_PIX_FMT_YUV422P);
        }
    }
}
#endif

#if defined(_USE_CUDA_)
// an encoder option that changes the bitstream format is reported from the encoded stream: NVENC -highbitdepth writes 10-bit HEVC from the 8-bit frames
TEST_F(WatermarkTest, EncodedFormatChangesAreReported) {
    const auto input = tempDir / "depth-source.mkv";
    writeVideoFixture(input, false, AVCOL_TRC_BT709, true, "libx264");
    for (const bool promoted : {true, false}) {
        auto settings = makeVideoSettings(input.string());
        settings.useHwEncoder = true;
        settings.watermarkInterval = 50;
        settings.encodeOptions = promoted ? "-c:v hevc_nvenc -highbitdepth 1" : "-c:v hevc_nvenc -profile:v main10";
        settings.encodeOutputPath = (tempDir / std::format("depth-{}.mkv", promoted)).string();
        auto video = initVideo(settings);
        testing::internal::CaptureStdout();
        const int frames = embedVideo(video.get());
        const std::string log = testing::internal::GetCapturedStdout();
        EXPECT_EQ(frames, 100);
        EXPECT_EQ(log.find("The encoded video is yuv420p10le while the watermarked frames are yuv420p") != std::string::npos, promoted) << log;
    }
}

// lossless NVENC from 4:4:4 CUDA surfaces keeps every chroma sample, from NVDEC (HEVC) and from CPU decoding (FFV1)
TEST_F(WatermarkTest, NvencKeeps444ChromaFromGpuAndCpuDecoding) {
    constexpr int interval = 7;
    for (const bool hevc : {true, false}) {
        SCOPED_TRACE(hevc ? "HEVC source" : "FFV1 source");
        const auto input = tempDir / (hevc ? "nvenc-444.mkv" : "nvenc-444-ffv1.mkv");
        if (hevc)
            writeVideoFixture(input, false, AVCOL_TRC_BT709, true, "libx265", AV_PIX_FMT_YUV444P, AVCOL_RANGE_MPEG,
                {
                    {"preset",      "ultrafast"         },
                    {"x265-params", "lossless=1:pools=2"}
            },
                true);
        else
            writeVideoFixture(input, false, AVCOL_TRC_BT709, true, nullptr, AV_PIX_FMT_YUV444P, AVCOL_RANGE_MPEG, {}, true);
        auto settings = makeVideoSettings(input.string());
        settings.useHwDecoder = settings.useHwEncoder = true;
        settings.watermarkInterval = interval;
        settings.encodeOptions = "-c:v hevc_nvenc -tune lossless";
        settings.encodeOutputPath = (tempDir / (hevc ? "nvenc-444-out.mkv" : "nvenc-444-ffv1-out.mkv")).string();
        auto video = initVideo(settings);
        if (!hevc)
            ASSERT_FALSE(video->useHwDecoder);
        ASSERT_EQ(embedVideo(video.get()), 100);
        if (video->outputFormat != AV_PIX_FMT_YUV444P)
            GTEST_SKIP() << "NVENC on this GPU does not encode HEVC 4:4:4";
        video.reset();
        const auto decoded = decodeVideoPlanes(settings.encodeOutputPath);
        EXPECT_EQ(decoded.format, AV_PIX_FMT_YUV444P);
        expectFixturePlanes(decoded, interval);
    }
}
#endif

// video embedding: covers the encoder/muxer setup, the filter graph, the encode
// thread and the frame conversion kernels
TEST_F(WatermarkTest, EmbedsIntoVideoAndDetectsFromTheEncodedFile) {
    const fs::path output = tempDir / "watermarked.mkv";
    VideoSettings embedSettings = makeVideoSettings(shortVideoFixture().string());
    embedSettings.encodeOutputPath = output.string();

    VideoHandle embedSession = initVideo(embedSettings);
    const int embeddedFrames = embedVideo(embedSession.get());
    embedSession.reset(); // close the muxer so the file is complete before we read it back
    ASSERT_EQ(embeddedFrames, 100);
    ASSERT_TRUE(fs::exists(output));
    ASSERT_GT(fs::file_size(output), 0U);

    VideoSettings detectSettings = makeVideoSettings(output.string());
    VideoHandle detectSession = initVideo(detectSettings);
    int detectedFrames = 0;
    const std::vector<float> correlations = capturedCorrelations(detectSession.get(), detectedFrames);

    EXPECT_EQ(detectedFrames, embeddedFrames) << "the encoder must not add or drop frames";
    ASSERT_EQ(correlations.size(), static_cast<size_t>(detectedFrames));
    // flat/fade frames carry no watermark, check the clip by its median frame (hopefully not flat too)
    std::vector<float> sorted = correlations;
    std::sort(sorted.begin(), sorted.end());
    EXPECT_GT(sorted[sorted.size() / 2], 0.5f);
}

TEST_F(WatermarkTest, EmbedsOnlyOnTheRequestedVideoInterval) {
    const fs::path output = tempDir / "interval.mkv";
    VideoSettings embedSettings = makeVideoSettings(shortVideoFixture().string());
    embedSettings.watermarkInterval = 50;
    embedSettings.encodeOutputPath = output.string();

    VideoHandle embedSession = initVideo(embedSettings);
    const int embeddedFrames = embedVideo(embedSession.get());
    embedSession.reset();
    ASSERT_GT(embeddedFrames, 0);

    VideoSettings detectSettings = makeVideoSettings(output.string());
    detectSettings.watermarkInterval = 50;
    VideoHandle detectSession = initVideo(detectSettings);
    int detectedFrames = 0;
    const std::vector<float> correlations = capturedCorrelations(detectSession.get(), detectedFrames);

    // detection checks exactly the frames that were watermarked
    EXPECT_EQ(correlations.size(), static_cast<size_t>((detectedFrames + 49) / 50));
    std::vector<float> sorted = correlations;
    std::sort(sorted.begin(), sorted.end());
    EXPECT_GT(sorted[sorted.size() / 2], 0.5f);
}

TEST_F(WatermarkTest, RefusesToOverwriteTheInputVideo) {
    VideoSettings settings = makeVideoSettings(shortVideoFixture().string());
    settings.encodeOutputPath = shortVideoFixture().string();
    VideoHandle session = initVideo(settings);
    EXPECT_THROW(embedVideo(session.get()), std::runtime_error);
}

// A muxer header failure must preserve the destination and remove temporary output
TEST_F(WatermarkTest, FailedVideoHeaderPreservesExistingOutput) {
    const fs::path output = tempDir / "existing.mp4";
    const std::string sentinel = "existing destination must survive";
    {
        std::ofstream file(output, std::ios::binary);
        file << sentinel;
    }
    auto settings = makeVideoSettings(shortVideoFixture().string());
    settings.encodeOptions = "-c:v libx264 -preset ultrafast -tag:v BAD!";
    settings.encodeOutputPath = output.string();
    auto video = initVideo(settings);
    EXPECT_THROW(embedVideo(video.get()), std::runtime_error);
    const auto bytes = readBytes(output);
    EXPECT_EQ(std::string(bytes.begin(), bytes.end()), sentinel);
    EXPECT_EQ(std::distance(fs::directory_iterator(tempDir), fs::directory_iterator{}), 1);
}

// A minimal WAV exercises missing-video validation without an external fixture
TEST_F(WatermarkTest, AudioOnlyInputReportsNoVideo) {
    const fs::path input = tempDir / "audio.wav";
    std::array<uint8_t, 844> wave{};
    const auto put16 = [&](size_t offset, uint16_t value) {
        wave[offset] = static_cast<uint8_t>(value);
        wave[offset + 1] = static_cast<uint8_t>(value >> 8);
    };
    const auto put32 = [&](size_t offset, uint32_t value) {
        for (int byte = 0; byte < 4; ++byte)
            wave[offset + byte] = static_cast<uint8_t>(value >> (byte * 8));
    };
    std::memcpy(wave.data(), "RIFF", 4);
    put32(4, static_cast<uint32_t>(wave.size() - 8));
    std::memcpy(wave.data() + 8, "WAVEfmt ", 8);
    put32(16, 16);
    put16(20, 1);
    put16(22, 1);
    put32(24, 8000);
    put32(28, 8000);
    put16(32, 1);
    put16(34, 8);
    std::memcpy(wave.data() + 36, "data", 4);
    put32(40, 800);
    {
        std::ofstream file(input, std::ios::binary);
        file.write(reinterpret_cast<const char*>(wave.data()), wave.size());
    }
    EXPECT_THROW(initVideo(makeVideoSettings(input.string())), std::runtime_error);
}

// An existing directory forces replacement to fail and tests destructor cleanup
TEST_F(WatermarkTest, OutputReplacementFailurePreservesTheDestinationAndCleansTemporaryFiles) {
    const fs::path destination = tempDir / "existing-directory.mp4";
    fs::create_directory(destination);
    {
        std::ofstream file(destination / "sentinel");
        file << "keep";
    }
    {
        OutputFile file;
        file.prepare(destination.string());
        {
            std::ofstream temporary(video_utils::pathFromUtf8(file.path()));
            temporary << "new output";
        }
        EXPECT_THROW(file.commit(), std::system_error);
        EXPECT_TRUE(fs::exists(destination / "sentinel"));
    }
    EXPECT_EQ(std::distance(fs::directory_iterator(tempDir), fs::directory_iterator{}), 1);
}

// MP4 faststart must finish reopening the temporary file before it is published
TEST_F(WatermarkTest, ReplacesExistingMp4OnlyAfterFaststartFinalization) {
    const auto input = tempDir / "input.mkv";
    const auto output = tempDir / "existing.mp4";
    writeVideoFixture(input, false, AVCOL_TRC_BT709);
    {
        std::ofstream sentinel(output);
        sentinel << "old destination";
    }
    auto settings = makeVideoSettings(input.string());
    settings.encodeOptions = "-c:v libx264 -preset ultrafast -movflags +faststart";
    settings.encodeOutputPath = output.string();
    auto video = initVideo(settings);
    ASSERT_EQ(embedVideo(video.get()), 1);
    video.reset();
    EXPECT_NEAR(firstVideoLuma(output).first, 128, 1);
    EXPECT_EQ(std::distance(fs::directory_iterator(tempDir), fs::directory_iterator{}), 2);
}

// Equivalent HDR samples should produce matching SDR output at both input depths
TEST_F(WatermarkTest, ToneMapsBothEightAndTenBitPqAndHlg) {
    for (const auto transfer : {AVCOL_TRC_SMPTE2084, AVCOL_TRC_ARIB_STD_B67}) {
        std::array<int, 2> luma{};
        for (const bool deep : {false, true}) {
            const auto input = tempDir / std::format("hdr-{}-{}.mkv", static_cast<int>(transfer), deep);
            const auto output = tempDir / std::format("sdr-{}-{}.mkv", static_cast<int>(transfer), deep);
            writeVideoFixture(input, deep, transfer);
            auto settings = makeVideoSettings(input.string());
            settings.encodeOptions = "-c:v ffv1";
            settings.encodeOutputPath = output.string();
            auto video = initVideo(settings);
            ASSERT_EQ(embedVideo(video.get()), 1);
            video.reset();
            const auto [sample, tag] = firstVideoLuma(output);
            EXPECT_EQ(tag, AVCOL_TRC_BT709);
            luma[deep ? 1 : 0] = sample;
        }
        EXPECT_NE(luma[0], 128) << "HDR samples require an actual conversion before SDR tagging";
        EXPECT_NEAR(luma[0], luma[1], 1) << "transfer " << static_cast<int>(transfer);
    }
}

// synthetic packets test muxer buffering, payloads do not need to be decodable
TEST(VideoMuxTest, EmptyAudioStreamDoesNotRetainTheWholeVideo) {
    using namespace video_utils;
    auto input = std::unique_ptr<AVFormatContext, decltype(&avformat_free_context)>(avformat_alloc_context(), avformat_free_context);
    ASSERT_NE(input, nullptr);
    auto* audio = avformat_new_stream(input.get(), nullptr);
    ASSERT_NE(audio, nullptr);
    audio->codecpar->codec_type = AVMEDIA_TYPE_AUDIO;
    audio->codecpar->codec_id = AV_CODEC_ID_PCM_S16LE;
    audio->codecpar->sample_rate = 8000;
    audio->codecpar->bits_per_coded_sample = 16;
    av_channel_layout_default(&audio->codecpar->ch_layout, 1);
    audio->time_base = {1, 8000};
    AVFormatContext* rawOutput = nullptr;
    ASSERT_EQ(avformat_alloc_output_context2(&rawOutput, nullptr, "matroska", nullptr), 0);
    auto output = std::unique_ptr<AVFormatContext, decltype(&avformat_free_context)>(rawOutput, avformat_free_context);
    auto* video = avformat_new_stream(output.get(), nullptr);
    ASSERT_NE(video, nullptr);
    video->time_base = {1, 1000};
    video->codecpar->codec_type = AVMEDIA_TYPE_VIDEO;
    video->codecpar->codec_id = AV_CODEC_ID_MPEG4;
    video->codecpar->width = video->codecpar->height = 16;
    AuxiliaryMux mux;
    AuxiliaryMuxSetup setup;
    setup.input = input.get();
    setup.output = output.get();
    setup.outputPath = "test.mkv";
    const int64_t defaultInterleaveDelta = output->max_interleave_delta;
    std::string error;
    ASSERT_TRUE(mux.configure(setup, error)) << error;
    EXPECT_EQ(output->max_interleave_delta, defaultInterleaveDelta);
    ASSERT_GT(output->max_interleave_delta, 0);
    ASSERT_EQ(avio_open_dyn_buf(&output->pb), 0);
    ASSERT_GE(avformat_write_header(output.get(), nullptr), 0);
    avio_flush(output->pb);
    const auto headerBytes = avio_tell(output->pb);
    AVPacketPtr packet(av_packet_alloc());
    ASSERT_NE(packet, nullptr);
    for (int i = 0; i < 120; ++i) {
        ASSERT_EQ(av_new_packet(packet.get(), 100000), 0);
        std::memset(packet->data, 0, packet->size);
        packet->stream_index = video->index;
        packet->pts = packet->dts = i * 1000;
        packet->duration = 1000;
        packet->flags = AV_PKT_FLAG_KEY;
        ASSERT_GE(av_interleaved_write_frame(output.get(), packet.get()), 0);
    }
    avio_flush(output->pb);
    // Most packets should be written before the trailer, even with the default limit
    EXPECT_GT(avio_tell(output->pb) - headerBytes, 10000000);
    EXPECT_GE(av_write_trailer(output.get()), 0);
    uint8_t* bytes = nullptr;
    avio_close_dyn_buf(output->pb, &bytes);
    av_free(bytes);
    output->pb = nullptr;
}

TEST_F(WatermarkTest, RejectsAnEncoderThatDisagreesWithTheSelectedBackend) {
    VideoSettings settings = makeVideoSettings(shortVideoFixture().string());
    settings.useHwEncoder = false;
    settings.encodeOptions = "-c:v hevc_nvenc"; // hardware encoder while the pipeline is set to software
    settings.encodeOutputPath = (tempDir / "mismatch.mkv").string();
    VideoHandle session = initVideo(settings);
    EXPECT_THROW(embedVideo(session.get()), std::runtime_error);
}

TEST_F(WatermarkTest, RejectsAnEncodeOptionsStringWithoutAnEncoder) {
    VideoSettings settings = makeVideoSettings(shortVideoFixture().string());
    settings.encodeOptions = "-preset ultrafast -crf 20";
    settings.encodeOutputPath = (tempDir / "nocodec.mkv").string();
    VideoHandle session = initVideo(settings);
    EXPECT_THROW(embedVideo(session.get()), std::runtime_error);
}

TEST_F(WatermarkTest, RejectsHybridHardwareEncodersWhenCpuEncodingIsSelected) {
    for (const char* codec : {"h264_qsv", "h264_mf"}) {
        auto settings = makeVideoSettings(shortVideoFixture().string());
        settings.encodeOptions = std::string("-c:v ") + codec;
        settings.encodeOutputPath = (tempDir / "hybrid-mismatch.mkv").string();
        auto video = initVideo(settings);
        EXPECT_THROW(embedVideo(video.get()), std::runtime_error);
    }
}

#if defined(_USE_EIGEN_)
TEST_F(WatermarkTest, EigenRejectsHardwareVideoEncodingBeforeOpeningInput) {
    auto settings = makeVideoSettings("missing-input.mkv");
    settings.useHwEncoder = true;
    EXPECT_THROW(initVideo(settings), std::invalid_argument);
}
#endif

// Repair invalid DTS values without changing timestamps that are already valid
TEST(VideoTimestampTest, RepairsOnlyInvalidDecodeTimestamps) {
    int64_t lastDts = AV_NOPTS_VALUE;
    AVPacket packet{};
    packet.pts = 100;
    packet.dts = 90;
    EXPECT_FALSE(video_utils::enforceMonotonicDts(&packet, lastDts));
    EXPECT_EQ(lastDts, 90);
    packet.pts = 101;
    packet.dts = 90;
    EXPECT_TRUE(video_utils::enforceMonotonicDts(&packet, lastDts));
    EXPECT_EQ(packet.dts, 91);
    EXPECT_EQ(packet.pts, 101);
    packet.pts = 91;
    packet.dts = 105;
    EXPECT_TRUE(video_utils::enforceMonotonicDts(&packet, lastDts));
    EXPECT_EQ(packet.dts, 92);
    EXPECT_EQ(packet.pts, 92);
}

TEST(VideoMuxTest, CopiesMatroskaAttachmentsAndDropsThemForMp4) {
    auto input = std::unique_ptr<AVFormatContext, decltype(&avformat_free_context)>(avformat_alloc_context(), avformat_free_context);
    ASSERT_NE(input, nullptr);
    AVStream* attachment = avformat_new_stream(input.get(), nullptr);
    ASSERT_NE(attachment, nullptr);
    attachment->codecpar->codec_type = AVMEDIA_TYPE_ATTACHMENT;
    attachment->codecpar->codec_id = AV_CODEC_ID_TTF;
    av_dict_set(&attachment->metadata, "filename", "font.ttf", 0);
    av_dict_set(&attachment->metadata, "mimetype", "application/x-truetype-font", 0);

    for (const char* extension : {"mkv", "mp4"}) {
        AVFormatContext* rawOutput = nullptr;
        ASSERT_EQ(avformat_alloc_output_context2(&rawOutput, nullptr, nullptr, (std::string("test.") + extension).c_str()), 0);
        auto output = std::unique_ptr<AVFormatContext, decltype(&avformat_free_context)>(rawOutput, avformat_free_context);
        video_utils::AuxiliaryMux mux;
        video_utils::AuxiliaryMuxSetup setup;
        setup.input = input.get();
        setup.output = output.get();
        setup.outputPath = std::string("test.") + extension;
        std::string error;
        ASSERT_TRUE(mux.configure(setup, error)) << error;
        if (std::string_view(extension) == "mkv") {
            ASSERT_EQ(output->nb_streams, 1u);
            EXPECT_EQ(output->streams[0]->codecpar->codec_id, AV_CODEC_ID_TTF);
            EXPECT_STREQ(av_dict_get(output->streams[0]->metadata, "filename", nullptr, 0)->value, "font.ttf");
        } else {
            EXPECT_EQ(output->nb_streams, 0u);
        }
    }
}

// MP4 cannot store SubRip, the events are converted to mov_text. Events that cannot be converted
// are dropped with one log line per stream and the others go on
TEST(VideoMuxTest, ConvertsTextSubtitlesAndDropsEventsItCannotConvert) {
    auto input = std::unique_ptr<AVFormatContext, decltype(&avformat_free_context)>(avformat_alloc_context(), avformat_free_context);
    ASSERT_NE(input, nullptr);
    AVStream* subtitles = avformat_new_stream(input.get(), nullptr);
    ASSERT_NE(subtitles, nullptr);
    subtitles->codecpar->codec_type = AVMEDIA_TYPE_SUBTITLE;
    subtitles->codecpar->codec_id = AV_CODEC_ID_SUBRIP;
    subtitles->time_base = AVRational{1, 1000};

    AVFormatContext* rawOutput = nullptr;
    ASSERT_EQ(avformat_alloc_output_context2(&rawOutput, nullptr, nullptr, "test.mp4"), 0);
    auto output = std::unique_ptr<AVFormatContext, decltype(&avformat_free_context)>(rawOutput, avformat_free_context);
    std::vector<int> emittedSizes;
    std::vector<std::string> logLines;
    video_utils::AuxiliaryMux mux;
    video_utils::AuxiliaryMuxSetup setup;
    setup.input = input.get();
    setup.output = output.get();
    setup.outputPath = "test.mp4";
    setup.videoWidth = 1280;
    setup.videoHeight = 720;
    setup.log = [&logLines](const std::string& line) { logLines.push_back(line); };
    setup.sink = [&emittedSizes](AVPacket* packet) {
        emittedSizes.push_back(packet->size);
        return true;
    };
    std::string error;
    ASSERT_TRUE(mux.configure(setup, error)) << error;
    ASSERT_EQ(output->nb_streams, 1u);
    EXPECT_EQ(output->streams[0]->codecpar->codec_id, AV_CODEC_ID_MOV_TEXT);

    const auto route = [&mux, &error](const std::string& text, const int64_t pts) {
        video_utils::AVPacketPtr packet(av_packet_alloc());
        if (!packet || av_new_packet(packet.get(), static_cast<int>(text.size())) < 0)
            return false;
        std::memcpy(packet->data, text.data(), text.size());
        packet->stream_index = 0;
        packet->pts = pts;
        packet->dts = pts;
        packet->duration = 1000;
        error.clear();
        return mux.routePacket(packet.get(), error);
    };
    ASSERT_TRUE(route("Hello subtitles", 0)) << error;
    ASSERT_EQ(emittedSizes.size(), 1u);
    const size_t setupLogLines = logLines.size();

    // mov_text stores at most 64 KiB per event: bigger events are dropped, logged once, and the next event still converts
    EXPECT_TRUE(route(std::string(100000, 'x'), 2000)) << error;
    EXPECT_TRUE(route(std::string(100000, 'y'), 3000)) << error;
    EXPECT_EQ(emittedSizes.size(), 1u);
    ASSERT_EQ(logLines.size(), setupLogLines + 1);
    EXPECT_NE(logLines.back().find("dropped"), std::string::npos) << logLines.back();
    ASSERT_TRUE(route("After the dropped events", 3500)) << error;
    EXPECT_EQ(emittedSizes.size(), 2u);

    // a failing output (sink) must fail the call
    mux.setSink([](AVPacket*) { return false; });
    EXPECT_FALSE(route("Another event", 4000));
    EXPECT_FALSE(error.empty());
}

// Encoder option parsing keeps codec options separate from pipeline-owned settings
TEST(EncodeOptionsTest, ExtractsTheEncoderAndForwardsTheRest) {
    const video_utils::ParsedEncodeOptions parsed = video_utils::parseEncodeOptions("-c:v libx265 -preset fast -crf 23");
    EXPECT_EQ(parsed.codecName, "libx265");
    EXPECT_EQ(dictValue(parsed, "preset"), "fast");
    EXPECT_EQ(dictValue(parsed, "crf"), "23");
}

TEST(EncodeOptionsTest, KeepsNegativeAndFractionalValuesAsValues) {
    const video_utils::ParsedEncodeOptions parsed = video_utils::parseEncodeOptions("-c:v libx265 -b:v 0 -qcomp -0.5 -rc-lookahead -1");
    EXPECT_EQ(parsed.codecName, "libx265");
    EXPECT_EQ(dictValue(parsed, "qcomp"), "-0.5");
    EXPECT_EQ(dictValue(parsed, "rc-lookahead"), "-1");
}

TEST(EncodeOptionsTest, IgnoresOptionsThePipelineOwns) {
    const video_utils::ParsedEncodeOptions parsed = video_utils::parseEncodeOptions("-c:v libx265 -map 0 -pix_fmt yuv444p -c:a aac");
    EXPECT_EQ(parsed.codecName, "libx265");
    EXPECT_EQ(dictValue(parsed, "pix_fmt"), "") << "pixel format is chosen by the pipeline, not the user";
    EXPECT_EQ(dictValue(parsed, "map"), "");
    EXPECT_FALSE(parsed.ignored.empty());
}

TEST(EncodeOptionsTest, CannotDisableTheMuxInterleaveLimit) {
    const auto parsed = video_utils::parseEncodeOptions("-c:v libx264 -max_interleave_delta 0");
    EXPECT_EQ(dictValue(parsed, "max_interleave_delta"), "");
    EXPECT_EQ(parsed.ignored, std::vector<std::string>{"-max_interleave_delta"});
}

TEST(EncodeOptionsTest, FlagsColourOptionsThatClashWithTheSourceMetadata) {
    const video_utils::ParsedEncodeOptions parsed = video_utils::parseEncodeOptions("-c:v libx265 -color_range pc");
    EXPECT_EQ(parsed.overrides.size(), 1U);
}

TEST(EncodeOptionsTest, DropsTrailingFlagsThatHaveNoValue) {
    const video_utils::ParsedEncodeOptions parsed = video_utils::parseEncodeOptions("-c:v libx265 -preset");
    EXPECT_EQ(parsed.codecName, "libx265");
    EXPECT_EQ(parsed.valueless.size(), 1U);
}

TEST(EncodeOptionsTest, RespectsQuotedValues) {
    const video_utils::ParsedEncodeOptions parsed = video_utils::parseEncodeOptions("-c:v libx265 -x265-params \"keyint=50:min-keyint=50\"");
    EXPECT_EQ(dictValue(parsed, "x265-params"), "keyint=50:min-keyint=50");
}

TEST(EncodeOptionsTest, ParsesBothNumericAndFourCcCodecTags) {
    EXPECT_EQ(video_utils::parseEncodeOptions("-c:v libx265 -tag:v hvc1").codecTag, "hvc1");
    EXPECT_EQ(video_utils::codecTagFromString("hvc1"), 0x31637668U); // little endian 'h','v','c','1'
    EXPECT_EQ(video_utils::codecTagFromString("0x31637668"), 0x31637668U);
}

// the UI benchmark and batch workers: images are preloaded on another thread (on the GPU backends they are uploaded on the compute stream)
// while the session embeds and detects (CUDA records its graphs meanwhile). Every run must work, the next one on a new thread too
TEST(WatermarkThreadingTest, EmbedsAndDetectsWhileAnotherThreadPreloadsImages) {
    for (int run = 0; run < 2; ++run) {
        std::string error;
        std::thread worker([&] {
            if (!initializeEnvironment(0)) {
                error = "no device";
                return;
            }
            const int device = getCurrentDeviceIndex();
            // a small image keeps the other thread uploading all the time
            std::atomic<bool> stop = false;
            std::future<void> preloads = std::async(std::launch::async, [&] {
                while (!stop)
                    preloadImageFromDisk("samples/images/18x18.png", device, false);
            });
            try {
                ImageHandle session = createImageSession(defaultPassword, defaultP, defaultPsnr);
                loadImage(session.get(), colorImage.string());
                // every p change records new graphs
                for (int round = 0; round < 5; ++round) {
                    for (const int p : {3, 5, 7, 9}) {
                        updateSessionParams(session.get(), p, defaultPsnr);
                        embedImage(session.get());
                        finish();
                        prepareDetectionImage(session.get());
                        detectEmbeddedBuffer(session.get());
                    }
                }
            } catch (const std::exception& e) { error = e.what(); }
            stop = true;
            try {
                preloads.get();
            } catch (const std::exception& e) {
                if (error.empty())
                    error = e.what();
            }
        });
        worker.join();
        EXPECT_EQ(error, "") << "run " << run;
    }
}
