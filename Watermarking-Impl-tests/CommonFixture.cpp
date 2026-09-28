#include "../Watermarking-Impl/AuxiliaryMux.hpp"
#include "../Watermarking-Impl/AvUtil.hpp"
#include "../Watermarking-Impl/EncodeOptions.hpp"
#include "../Watermarking-Impl/half_float.hpp"
#include "../Watermarking-Impl/WatermarkCrypto.hpp"
#if defined(_USE_OPENCL_)
#include "../Watermarking-Impl/OclArray.hpp"
#include "../Watermarking-Impl/opencl_utils.hpp"
#include <cstdlib>
#include <optional>
#elif defined(_USE_CUDA_)
#include "../Watermarking-Impl/cuda_utils.hpp"
#include "../Watermarking-Impl/CudaArray.hpp"
#include "../Watermarking-Impl/CudaStreamManager.hpp"
#include "../Watermarking-Impl/nvjpeg_utils.hpp"
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
#include <filesystem>
#include <format>
#include <fstream>
#include <future>
#include <gtest/gtest.h>
#include <iomanip>
#include <iterator>
#include <limits>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
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
#include "libavutil/dict.h"
}

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
// small (1.9 MB / 693 frame) clip, it also tests the 10-bit to 8-bit filter graph
const fs::path shortVideo = "samples/videos/sample_1080p_10bit.mkv";

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

// writes "displayed" (row-major interleaved, 1 or 3 channels) as a quality 100, 4:4:4 JPEG with the EXIF orientation tag, its pixels stored so that a viewer applying the
// orientation shows "displayed"
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

// writes "pixels" (row-major interleaved; 1, 3 or 4 components: gray, RGB, CMYK) as a quality 90 JPEG, the first component sampled "sampling" times the others
// in both directions (1: 4:4:4, 2: 4:2:0)
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

ReductionPathResult runReductionPath() {
    ImageHandle reductionSession = createImageSession(defaultPassword, defaultP, defaultPsnr);
    loadImage(reductionSession.get(), colorImage.string());
    ReductionPathResult result;
    result.pixels = embedAndRead(reductionSession.get());
    prepareDetectionImage(reductionSession.get());
    result.correlation = detectEmbeddedBuffer(reductionSession.get());
    return result;
}
#endif
} // namespace

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
TEST(OpenCLReductionTest, AvailableDevicesMatchForcedPortableResults) {
    PortableReductionOverrideGuard restoreOverride;
    const std::vector<std::string> devices = getAvailableDevices();
    ASSERT_FALSE(devices.empty());

    for (size_t deviceIndex = 0; deviceIndex < devices.size(); ++deviceIndex) {
        setPortableReductionOverride(false);
        ASSERT_TRUE(initializeEnvironment(static_cast<int>(deviceIndex))) << devices[deviceIndex];
        buildOpenCLKernels(); // every documented prediction order must compile on every device
        const cl::Program selectedProgram = cl_utils::OpenCLKernelCache<defaultP>::getProgram();
        const cl_utils::ReductionMode selectedMode = cl_utils::reductionMode(selectedProgram);
        const ReductionPathResult selected = runReductionPath();
        EXPECT_TRUE(std::isfinite(selected.correlation)) << devices[deviceIndex];
        EXPECT_GT(selected.correlation, 0.5f) << devices[deviceIndex];

        if (selectedMode == cl_utils::ReductionMode::Portable)
            continue;

        setPortableReductionOverride(true);
        const cl::Program portableProgram = cl_utils::OpenCLKernelCache<defaultP>::getProgram();
        ASSERT_EQ(cl_utils::reductionMode(portableProgram), cl_utils::ReductionMode::Portable) << devices[deviceIndex];
        const ReductionPathResult portable = runReductionPath();
        EXPECT_EQ(selected.pixels.width, portable.pixels.width) << devices[deviceIndex];
        EXPECT_EQ(selected.pixels.height, portable.pixels.height) << devices[deviceIndex];
        EXPECT_EQ(selected.pixels.channels, portable.pixels.channels) << devices[deviceIndex];
        EXPECT_EQ(selected.pixels.pixels, portable.pixels.pixels) << devices[deviceIndex];
        EXPECT_NEAR(selected.correlation, portable.correlation, 1.0e-4f) << devices[deviceIndex];
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

TEST_F(WatermarkTest, RejectsUndocumentedPredictionOrder) {
    ImageHandle badSession = createImageSession(defaultPassword, 4, defaultPsnr);
    EXPECT_THROW(loadImage(badSession.get(), colorImage.string()), std::invalid_argument);
}

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

// every EXIF orientation of one displayed image (40x72: partial GPU tiles) loads upright, with the embedded pixels and the original preview of orientation 1.
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

// JPEG decoding: CImg (libjpeg) on the Eigen and OpenCL builds, nvJPEG on CUDA, within +-3 of libjpeg (different IDCT and upsampling rounding).
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
    VideoSettings settings = makeVideoSettings(shortVideo.string());
    settings.psnr = defaultPsnr;
    settings.encodeOptions.clear();

    VideoHandle video = initVideo(settings);
    EXPECT_THROW(detectVideo(video.get()), std::runtime_error);
}

// video embedding: covers the encoder/muxer setup, the filter graph, the encode
// thread and the frame conversion kernels
TEST_F(WatermarkTest, EmbedsIntoVideoAndDetectsFromTheEncodedFile) {
    const fs::path output = tempDir / "watermarked.mkv";
    VideoSettings embedSettings = makeVideoSettings(shortVideo.string());
    embedSettings.encodeOutputPath = output.string();

    VideoHandle embedSession = initVideo(embedSettings);
    const int embeddedFrames = embedVideo(embedSession.get());
    embedSession.reset(); // close the muxer so the file is complete before we read it back
    ASSERT_GT(embeddedFrames, 0);
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
    VideoSettings embedSettings = makeVideoSettings(shortVideo.string());
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
    VideoSettings settings = makeVideoSettings(shortVideo.string());
    settings.encodeOutputPath = shortVideo.string();
    VideoHandle session = initVideo(settings);
    EXPECT_THROW(embedVideo(session.get()), std::runtime_error);
}

TEST_F(WatermarkTest, RejectsAnEncoderThatDisagreesWithTheSelectedBackend) {
    VideoSettings settings = makeVideoSettings(shortVideo.string());
    settings.useHwEncoder = false;
    settings.encodeOptions = "-c:v hevc_nvenc"; // hardware encoder while the pipeline is set to software
    settings.encodeOutputPath = (tempDir / "mismatch.mkv").string();
    VideoHandle session = initVideo(settings);
    EXPECT_THROW(embedVideo(session.get()), std::runtime_error);
}

TEST_F(WatermarkTest, RejectsAnEncodeOptionsStringWithoutAnEncoder) {
    VideoSettings settings = makeVideoSettings(shortVideo.string());
    settings.encodeOptions = "-preset ultrafast -crf 20";
    settings.encodeOutputPath = (tempDir / "nocodec.mkv").string();
    VideoHandle session = initVideo(settings);
    EXPECT_THROW(embedVideo(session.get()), std::runtime_error);
}

// encode option parsing
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
