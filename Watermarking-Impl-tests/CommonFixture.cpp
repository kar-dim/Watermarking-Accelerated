#include "../Watermarking-Impl/WatermarkCrypto.hpp"
#include "WatermarkCore.hpp"
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <gtest/gtest.h>
#include <iomanip>
#include <sstream>
#include <string>
#include <system_error>
#include <WatermarkTypes.hpp>

using namespace WatermarkCore;
namespace fs = std::filesystem;

namespace {
constexpr int defaultP = 3;
constexpr float defaultPsnr = 40.0f;
constexpr const char* defaultPassword = "random_watermark_password";
const fs::path colorImage = "samples/images/512.png";
const fs::path grayImage = "samples/images/512_gray.jpg";

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
    }
}

TEST_F(WatermarkTest, RejectsHighBitDepthVideoDetection) {
    VideoSettings settings{};
    settings.videoFile = "samples/videos/sample_1080p_10bit.mkv";
    settings.watermarkPassword = defaultPassword;
    settings.p = defaultP;
    settings.psnr = defaultPsnr;
    settings.watermarkInterval = 1;
    settings.useHwDecoder = false;
    settings.useHwEncoder = false;

    VideoHandle video = initVideo(settings);
    EXPECT_THROW(detectVideo(video.get()), std::runtime_error);
}
