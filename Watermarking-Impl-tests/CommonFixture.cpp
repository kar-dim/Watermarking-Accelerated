#include "../Watermarking-Impl/EncodeOptions.hpp"
#include "../Watermarking-Impl/WatermarkCrypto.hpp"
#include "WatermarkCore.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <gtest/gtest.h>
#include <iomanip>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>
#include <WatermarkTypes.hpp>

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
    loadImage(alphaSession.get(), alphaImage.string());
    embedImage(alphaSession.get(), MaskMethod::ME);
    finish();

    const fs::path requested = tempDir / "alpha.png";
    const fs::path saved = tempDir / "alphaW_ME.png";
    saveImage(alphaSession.get(), requested.string(), MaskMethod::ME);
    ASSERT_TRUE(fs::exists(saved));

    // reloading 4 channel PNG must still detect, and the alpha channel must not bleed into the colour planes
    ImageHandle reloaded = createImageSession(defaultPassword, defaultP, defaultPsnr);
    loadImage(reloaded.get(), saved.string());
    const float correlation = detectLoadedImage(reloaded.get(), MaskMethod::ME);
    EXPECT_TRUE(std::isfinite(correlation));
    EXPECT_GT(correlation, 0.5f);
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
