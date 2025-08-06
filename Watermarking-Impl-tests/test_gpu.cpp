#include "gtest/gtest.h"

#include "buffer.hpp"
#include "constants.h"
#include "FileDeleter.h"
#include "MaskDiskConfig.h"
#include "test_common.hpp"
#include "utils.hpp"
#include "WatermarkBase.hpp"
#include <arrayfire.h>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

using std::cout;
using std::string;

/*!
 *  \brief  Main Test class used for testing watermarking algorithms
 *  \author Dimitris Karatzas
 */

class GpuFixture : public CommonFixture 
{
protected:
	//load the input image and initialize watermark object
    void SetUp() override 
    {
        CommonFixture::SetUp();
        watermarkObj = Utils::createWatermarkObject(static_cast<unsigned int>(image.dims(0)), static_cast<unsigned int>(image.dims(1)), watermarkPath, p, psnr);
    }

#if defined(_USE_OPENCL_)
    static void SetUpTestSuite() 
    {
        static constexpr int openclDevice = 0;
        try {
            af::setDevice(openclDevice);
        }
        catch (const std::exception&) {
            std::cout << "NOTE: Invalid OpenCL device specified, using default 0\n";
            af::setDevice(0);
        }
    }
#endif

	//helper method to save the watermarked image to disk and check if it matches the expected MSE threshold
    void testSaveToDisk(MASK_TYPE mask, const std::string& label, const std::string& outputFileName)
    {
        float strength = 0.0f;
		FileDeleter cleanup(outputFileName); //delete the file after the test
        const BufferType watermark = embedWatermark(image, rgbImage, strength, mask);
        Utils::saveImage(imageFile, label, watermark, alphaChannel);
        BufferType diskRgb, diskImage;
        std::optional<BufferAlphaType> diskAlpha;
        Utils::loadImage(diskRgb, diskImage, outputFileName, diskAlpha);
        EXPECT_EQ(diskRgb.elements(), watermark.elements()) << "Expected disk image (" << label << ") elements to match original";
        const float mse = af::sum<float>(af::abs(diskRgb - watermark)) / diskRgb.elements();
        EXPECT_LE(mse, mseThreshold);
    }
};

TEST_F(GpuFixture, EmbedWatermark)
{
    testEmbedding();
}

TEST_F(GpuFixture, DetectWatermark)
{
    float strengthNvf = 0.0f, strengthMe = 0.0f;
    const BufferType watermarkedNVFgray = af::rgb2gray(embedWatermark(image, rgbImage, strengthNvf, NVF), Constants::rPercent, Constants::gPercent, Constants::bPercent);
    const BufferType watermarkedMEgray = af::rgb2gray(embedWatermark(image, rgbImage, strengthMe, ME), Constants::rPercent, Constants::gPercent, Constants::bPercent);
    const float correlationNvf = watermarkObj->detectWatermark(watermarkedNVFgray, NVF);
    const float correlationMe = watermarkObj->detectWatermark(watermarkedMEgray, ME);
    //watermark correlation of Me should be at least as NVF
    EXPECT_GE(correlationMe, correlationNvf);
}

TEST_F(GpuFixture, SaveToDisk)
{
    std::vector<MaskDiskConfig> strategies = 
    {
        { NVF, "W_NVF", "../../Watermarking-Impl/samples/images/4kW_NVF.png" },
        { ME,  "W_ME",  "../../Watermarking-Impl/samples/images/4kW_ME.png" }
    };
    for (const auto& config : strategies)
        testSaveToDisk(config.strategy, config.label, config.outputFile);
}