#include "gtest/gtest.h"

#include "buffer.hpp"
#include "MaskDiskConfig.h"
#include "test_common.hpp"
#include "utils.hpp"
#include "WatermarkBase.hpp"
#include <arrayfire.h>
#include <iostream>
#include <memory>
#include <string>

#if defined(_USE_OPENCL_)
#include <exception>
#endif

using std::cout;
using std::string;

/*!
 *  \brief  Main Test class used for testing watermarking algorithms
 *  \author Dimitris Karatzas
 */

class GpuFixture : public CommonFixture {
  protected:
    void SetUp() override {
        CommonFixture::SetUp();
        watermarkObj = Utils::createWatermarkObject(rows, cols, watermarkPath, p, psnr);
    }

    static void SetUpTestSuite() {
#if defined(_USE_OPENCL_)
        static constexpr int openclDevice = 1;
        try {
            af::setDevice(openclDevice);
        } catch (const std::exception&) {
            cout << "NOTE: Invalid OpenCL device specified, using default 0\n";
            af::setDevice(0);
        }
#endif
    }

    static void TearDownTestSuite() { CommonFixture::TearDownTestSuite(); }

    ImageBuffer embedAndConvertToGray(MASK_TYPE maskType) override {
        float strength = 0.0f;
        ImageOutputBuffer watermarkedImage;
        return Utils::rgb2gray(embedWatermark(watermarkedImage, strength, maskType));
    }

    void calculateMSE(const ImageBuffer& diskRgb, const ImageOutputBuffer& watermark) override {
        EXPECT_EQ(diskRgb.elements(), watermark.elements()) << "Expected disk image elements to match original";
        const float mse = af::sum<float>(af::abs(diskRgb - watermark)) / diskRgb.elements();
        EXPECT_LE(mse, mseThreshold);
    }

    // helper method to embed watermark in the image
    ImageOutputBuffer embedWatermark(ImageOutputBuffer& output, float& strength, MASK_TYPE maskType) override {
        watermarkObj->makeWatermark(buf.image, buf.rgbImage, output, strength, maskType);
        EXPECT_GT(output.elements(), 0);
        EXPECT_FALSE(af::anyTrue<bool>(af::isNaN(output) | af::isInf(output)));
        return output;
    }

    // helper methhod to embed watermark for both mask types (we can't check strength values, they were on VRAM)
    void testEmbedding(ImageOutputBuffer& output) override {
        float strength = 0.0f;
        embedWatermark(output, strength, NVF);
        embedWatermark(output, strength, ME);
    }
};

TEST_F(GpuFixture, EmbedWatermark) {
    ImageOutputBuffer output;
    testEmbedding(output);
}

TEST_F(GpuFixture, DetectWatermark) { EXPECT_GE(calculateCorrelation(ME), calculateCorrelation(NVF)); }

TEST_F(GpuFixture, SaveToDisk) {
    ImageOutputBuffer watermark;
    for (const auto& config : strategies)
        testSaveToDisk(watermark, config.strategy, config.label, config.outputFile);
}