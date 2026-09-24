#include "../Watermarking-Impl/AuxiliaryMux.hpp"
#include "../Watermarking-Impl/AvUtil.hpp"
#include "../Watermarking-Impl/EncodeOptions.hpp"
#include "../Watermarking-Impl/WatermarkCrypto.hpp"
#if defined(_USE_OPENCL_)
#include "../Watermarking-Impl/opencl_utils.hpp"
#include <cstdlib>
#include <optional>
#endif
#include "WatermarkCore.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <future>
#include <gtest/gtest.h>
#include <iomanip>
#include <iterator>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>
#include <WatermarkTypes.hpp>

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
// small (1.9 MB / 693 frame) clip, it also tests the 10-bit to 8-bit filter graph
const fs::path shortVideo = "samples/videos/sample_1080p_10bit.mkv";

std::string hexDigest(const std::array<uint8_t, 32>& digest) {
    std::ostringstream result;
    result << std::hex << std::setfill('0');
    for (const uint8_t byte : digest)
        result << std::setw(2) << static_cast<unsigned int>(byte);
    return result.str();
}

SessionPixelData embedAndRead(ImageSession* session, const MaskMethod method) {
    embedImage(session, method);
    finish();
    return getSessionPixelData(session);
}

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

std::array<ReductionPathResult, 2> runReductionPath() {
    std::array<ReductionPathResult, 2> results;
    const std::array methods = {MaskMethod::NVF, MaskMethod::ME};
    for (size_t index = 0; index < methods.size(); ++index) {
        ImageHandle reductionSession = createImageSession(defaultPassword, defaultP, defaultPsnr);
        loadImage(reductionSession.get(), colorImage.string());
        embedImage(reductionSession.get(), methods[index]);
        finish();
        results[index].pixels = getSessionPixelData(reductionSession.get());
        prepareDetectionImage(reductionSession.get(), methods[index]);
        results[index].correlation = detectEmbeddedBuffer(reductionSession.get(), methods[index]);
    }
    return results;
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
        const auto selected = runReductionPath();

        for (size_t method = 0; method < selected.size(); ++method) {
            EXPECT_TRUE(std::isfinite(selected[method].correlation)) << devices[deviceIndex];
            EXPECT_GT(selected[method].correlation, 0.5f) << devices[deviceIndex];
        }

        if (selectedMode == cl_utils::ReductionMode::Portable)
            continue;

        setPortableReductionOverride(true);
        const cl::Program portableProgram = cl_utils::OpenCLKernelCache<defaultP>::getProgram();
        ASSERT_EQ(cl_utils::reductionMode(portableProgram), cl_utils::ReductionMode::Portable) << devices[deviceIndex];
        const auto portable = runReductionPath();

        for (size_t method = 0; method < selected.size(); ++method) {
            EXPECT_EQ(selected[method].pixels.width, portable[method].pixels.width) << devices[deviceIndex];
            EXPECT_EQ(selected[method].pixels.height, portable[method].pixels.height) << devices[deviceIndex];
            EXPECT_EQ(selected[method].pixels.channels, portable[method].pixels.channels) << devices[deviceIndex];
            EXPECT_EQ(selected[method].pixels.pixels, portable[method].pixels.pixels) << devices[deviceIndex];
            EXPECT_NEAR(selected[method].correlation, portable[method].correlation, 1.0e-4f) << devices[deviceIndex];
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

TEST_F(WatermarkTest, EmbedsAndDetectsBothMasks) {
    for (const MaskMethod method : {MaskMethod::NVF, MaskMethod::ME}) {
        embedImage(session.get(), method);
        finish();
        prepareDetectionImage(session.get(), method);
        const float correlation = detectEmbeddedBuffer(session.get(), method);
        EXPECT_TRUE(std::isfinite(correlation));
        EXPECT_GT(correlation, 0.5f);
    }
}

TEST_F(WatermarkTest, SavesReloadsAndDetectsFromTemporaryDirectory) {
    embedImage(session.get(), MaskMethod::ME);
    finish();

    const fs::path requestedPath = tempDir / "result.png";
    const fs::path savedPath = tempDir / "resultW_ME.png";
    saveImage(session.get(), requestedPath.string(), MaskMethod::ME);
    ASSERT_TRUE(fs::exists(savedPath));

    ImageHandle diskSession = createImageSession(defaultPassword, defaultP, defaultPsnr);
    loadImage(diskSession.get(), savedPath.string());
    const float diskCorrelation = detectLoadedImage(diskSession.get(), MaskMethod::ME);
    EXPECT_TRUE(std::isfinite(diskCorrelation));
    EXPECT_GT(diskCorrelation, 0.65f);
}

TEST_F(WatermarkTest, ReusesSessionAcrossSameSizeRgbAndGrayImages) {
    const SessionPixelData rgb = embedAndRead(session.get(), MaskMethod::NVF);
    ASSERT_EQ(rgb.channels, 3);

    loadImage(session.get(), grayImage.string());
    const SessionPixelData gray = embedAndRead(session.get(), MaskMethod::NVF);
    EXPECT_EQ(gray.width, rgb.width);
    EXPECT_EQ(gray.height, rgb.height);
    EXPECT_EQ(gray.channels, 1);
    EXPECT_EQ(gray.pixels.size(), static_cast<size_t>(gray.width) * gray.height);
}

TEST_F(WatermarkTest, SameInputsAreDeterministicAcrossSessions) {
    const SessionPixelData first = embedAndRead(session.get(), MaskMethod::NVF);

    ImageHandle secondSession = createImageSession(defaultPassword, defaultP, defaultPsnr);
    loadImage(secondSession.get(), colorImage.string());
    const SessionPixelData second = embedAndRead(secondSession.get(), MaskMethod::NVF);
    EXPECT_EQ(first.pixels, second.pixels);
}

TEST_F(WatermarkTest, ExportedImageRemainsStableAcrossSessionReuse) {
    embedImage(session.get(), MaskMethod::NVF);
    finish();
    ExportHandle exported = createReusableExportBuffer();
    exportForSave(session.get(), exported.get(), MaskMethod::NVF);
    const fs::path first = tempDir / "firstW_NVF.png";
    const fs::path second = tempDir / "secondW_NVF.png";
    const fs::path third = tempDir / "thirdW_NVF.png";
    flushToDiskAsync(exported.get(), (tempDir / "first.png").string(), MaskMethod::NVF);

    updateSessionParams(session.get(), defaultP, 30.0f);
    embedImage(session.get(), MaskMethod::NVF);
    finish();
    flushToDiskAsync(exported.get(), (tempDir / "second.png").string(), MaskMethod::NVF);

    std::ifstream firstFile(first, std::ios::binary);
    std::ifstream secondFile(second, std::ios::binary);
    ASSERT_TRUE(firstFile && secondFile);
    const std::vector<char> firstBytes(std::istreambuf_iterator<char>{firstFile}, {});
    const std::vector<char> secondBytes(std::istreambuf_iterator<char>{secondFile}, {});
    EXPECT_EQ(firstBytes, secondBytes);

    exportForSave(session.get(), exported.get(), MaskMethod::NVF);
    flushToDiskAsync(exported.get(), (tempDir / "third.png").string(), MaskMethod::NVF);
    std::ifstream thirdFile(third, std::ios::binary);
    ASSERT_TRUE(thirdFile);
    const std::vector<char> thirdBytes(std::istreambuf_iterator<char>{thirdFile}, {});
    EXPECT_NE(firstBytes, thirdBytes);
}

TEST_F(WatermarkTest, DifferentPasswordsProduceDifferentWatermarks) {
    const SessionPixelData first = embedAndRead(session.get(), MaskMethod::NVF);

    ImageHandle secondSession = createImageSession("a_different_password", defaultP, defaultPsnr);
    loadImage(secondSession.get(), colorImage.string());
    const SessionPixelData second = embedAndRead(secondSession.get(), MaskMethod::NVF);
    EXPECT_NE(first.pixels, second.pixels);
}

TEST_F(WatermarkTest, PsnrOnlyUpdatePreservesTheDeterministicWatermark) {
    const SessionPixelData original = embedAndRead(session.get(), MaskMethod::NVF);

    updateSessionParams(session.get(), defaultP, 30.0f);
    const SessionPixelData stronger = embedAndRead(session.get(), MaskMethod::NVF);
    EXPECT_NE(original.pixels, stronger.pixels);

    updateSessionParams(session.get(), defaultP, defaultPsnr);
    const SessionPixelData restored = embedAndRead(session.get(), MaskMethod::NVF);
    EXPECT_EQ(original.pixels, restored.pixels);
}

TEST_F(WatermarkTest, SupportsEveryDocumentedPredictionOrder) {
    for (const int predictionOrder : {3, 5, 7, 9}) {
        ImageHandle pSession = createImageSession(defaultPassword, predictionOrder, defaultPsnr);
        loadImage(pSession.get(), colorImage.string());
        const SessionPixelData output = embedAndRead(pSession.get(), MaskMethod::ME);
        EXPECT_EQ(output.pixels.size(), static_cast<size_t>(output.width) * output.height * output.channels) << "p=" << predictionOrder;
        prepareDetectionImage(pSession.get(), MaskMethod::ME);
        const float correlation = detectEmbeddedBuffer(pSession.get(), MaskMethod::ME);
        EXPECT_TRUE(std::isfinite(correlation)) << "p=" << predictionOrder;
        EXPECT_GT(correlation, 0.5f) << "p=" << predictionOrder;
    }
}

TEST_F(WatermarkTest, RejectsUndocumentedPredictionOrder) {
    ImageHandle badSession = createImageSession(defaultPassword, 4, defaultPsnr);
    EXPECT_THROW(loadImage(badSession.get(), colorImage.string()), std::invalid_argument);
}

TEST_F(WatermarkTest, PreservesTheAlphaChannelWhenSaving) {
    ASSERT_TRUE(fs::exists(alphaImage)) << alphaImage;
    ImageHandle alphaSession = createImageSession(defaultPassword, defaultP, defaultPsnr);
    loadImage(alphaSession.get(), alphaImage.string(), true);
    const OriginalPixelData original = takeOriginalPixelData(alphaSession.get());
    ASSERT_EQ(original.channels, 4);
    embedImage(alphaSession.get(), MaskMethod::ME);
    finish();

    const fs::path requested = tempDir / "alpha.png";
    const fs::path saved = tempDir / "alphaW_ME.png";
    saveImage(alphaSession.get(), requested.string(), MaskMethod::ME);
    ASSERT_TRUE(fs::exists(saved));

    ExportHandle exported = createReusableExportBuffer();
    exportForSave(alphaSession.get(), exported.get(), MaskMethod::ME);
    const fs::path exportedSaved = tempDir / "alpha_exportW_ME.png";
    auto saveTask = std::async(std::launch::async, flushToDiskAsync, exported.get(), (tempDir / "alpha_export.png").string(), MaskMethod::ME);
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
        const float correlation = detectLoadedImage(reloaded.get(), MaskMethod::ME);
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
        embedImage(image.get(), MaskMethod::ME);
        prepareDetectionImage(image.get(), MaskMethod::ME);
        return detectEmbeddedBuffer(image.get(), MaskMethod::ME);
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
            embedImage(image.get(), MaskMethod::ME);
            prepareDetectionImage(image.get(), MaskMethod::ME);
            const float correlation = detectEmbeddedBuffer(image.get(), MaskMethod::ME);
            EXPECT_TRUE(std::isfinite(correlation)) << devices[deviceIndex] << " p=" << order;
            EXPECT_GT(correlation, 0.75f) << devices[deviceIndex] << " p=" << order;
        }
    }
}
#endif

TEST(PreviewPixelConversionTest, HandlesPlanarRgbAndGrayWithPaddedRows) {
    for (const auto [width, height] : {
             std::pair{5,  3 },
             std::pair{8,  8 },
             std::pair{13, 11},
             std::pair{16, 31},
             std::pair{17, 35}
    }) {
        for (const int channels : {1, 3}) {
            SessionPixelData source;
            source.width = width;
            source.height = height;
            source.channels = channels;
            const size_t planeSize = static_cast<size_t>(source.width) * source.height;
            source.pixels.resize(planeSize * channels);
            for (int channel = 0; channel < channels; ++channel)
                for (int col = 0; col < source.width; ++col)
                    for (int row = 0; row < source.height; ++row)
                        source.pixels[static_cast<size_t>(channel) * planeSize + static_cast<size_t>(col) * source.height + row] = static_cast<uint8_t>(channel * 50 + col * 7 + row);

            const size_t stride = static_cast<size_t>(source.width) * channels + 3;
            std::vector<uint8_t> preview(stride * source.height, 0xA5);
            copySessionPixelsForPreview(source, preview.data(), stride);
            for (int row = 0; row < source.height; ++row) {
                for (int col = 0; col < source.width; ++col)
                    for (int channel = 0; channel < channels; ++channel)
                        EXPECT_EQ(preview[static_cast<size_t>(row) * stride + col * channels + channel], static_cast<uint8_t>(channel * 50 + col * 7 + row));
                for (size_t padding = static_cast<size_t>(source.width) * channels; padding < stride; ++padding)
                    EXPECT_EQ(preview[static_cast<size_t>(row) * stride + padding], 0xA5);
            }
        }
    }
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
