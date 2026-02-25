#include "WatermarkEngine.hpp"
#include <filesystem>
#include <gtest/gtest.h>
#include <string>
#include <WatermarkTypes.hpp>

using namespace WatermarkEngine;

class WatermarkTest : public ::testing::Test {
  protected:
    const std::string imageFile = "../../Watermarking-CLI/samples/images/4k.png";
    const std::string watermarkPath = "../../Watermarking-CLI/samples/w_4k.dat";
    const std::string imageNvfPath = "../../Watermarking-CLI/samples/images/4kW_NVF.png";
    const std::string imageMePath = "../../Watermarking-CLI/samples/images/4kW_ME.png";
    const int p = 3;
    const float psnr = 40.0f;

    ImageHandle session{nullptr};

    void SetUp() override {
        WatermarkEngine::initializeEnvironment(0);
        session = WatermarkEngine::openImage(imageFile, watermarkPath, p, psnr);
    }

    void TearDown() override {
        std::filesystem::remove(imageNvfPath);
        std::filesystem::remove(imageMePath);
    }
};

// test embedding
 TEST_F(WatermarkTest, EmbedWatermark) {
    ASSERT_NE(session, nullptr) << "Session failed to initialize!";
    EXPECT_NO_THROW(WatermarkEngine::embedImage(session.get(), MaskMethod::NVF));
    EXPECT_NO_THROW(WatermarkEngine::embedImage(session.get(), MaskMethod::ME));
}

// test correlation
TEST_F(WatermarkTest, DetectWatermark) {
    // nvf
    WatermarkEngine::embedImage(session.get(), MaskMethod::NVF);
    WatermarkEngine::prepareDetectionImage(session.get(), MaskMethod::NVF);
    const float corrNvf = WatermarkEngine::detectEmbeddedBuffer(session.get(), MaskMethod::NVF);
    // me
    WatermarkEngine::embedImage(session.get(), MaskMethod::ME);
    WatermarkEngine::prepareDetectionImage(session.get(), MaskMethod::ME);
    const float corrMe = WatermarkEngine::detectEmbeddedBuffer(session.get(), MaskMethod::ME);
    // ME algorithm should (generally) have higher or equal correlation
    EXPECT_GE(corrMe, corrNvf);
}

// test saving to disk
TEST_F(WatermarkTest, SaveToDisk) {
    // nvf
    WatermarkEngine::embedImage(session.get(), MaskMethod::NVF);
    WatermarkEngine::saveImage(session.get(), imageFile, MaskMethod::NVF);
    EXPECT_TRUE(std::filesystem::exists(imageNvfPath)) << "NVF image was not saved to disk!";
    // me
    WatermarkEngine::embedImage(session.get(), MaskMethod::ME);
    WatermarkEngine::saveImage(session.get(), imageFile, MaskMethod::ME);
    EXPECT_TRUE(std::filesystem::exists(imageMePath)) << "ME image was not saved to disk!";
    // test if the saved image can be loaded and detected correctly
    ImageHandle diskSession = WatermarkEngine::openImage(imageMePath, watermarkPath, p, psnr);
    const float diskCorr = WatermarkEngine::detectLoadedImage(diskSession.get(), MaskMethod::ME);
    EXPECT_GT(diskCorr, 0.80f) << "The saved image lost too much watermark data after saving to disk, OR was not embedded correctly!";
}