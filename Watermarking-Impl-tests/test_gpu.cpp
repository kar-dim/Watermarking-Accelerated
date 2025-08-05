#include "gtest/gtest.h"

#include "buffer.hpp"
#include "constants.h"
#include "FileDeleter.h"
#include "MaskDiskConfig.h"
#include "utils.hpp"
#include "WatermarkBase.hpp"
#include <arrayfire.h>
#if defined(_USE_OPENCL_)
#include <exception>
#endif
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

class TestFixture : public ::testing::Test 
{
protected:
    static constexpr float psnr = 40.0f;
    static constexpr float mseThreshold = 10.0f;
    static constexpr int p = 3;
	static constexpr int openclDevice = 1;

    std::unique_ptr<WatermarkBase> watermarkObj;
    BufferType rgbImage, image;
    std::optional<BufferAlphaType> alphaChannel;
	const std::string imageFile = "../../Watermarking-Impl/samples/images/4k.png";
	const std::string watermarkPath = "../../Watermarking-Impl/samples/w_4k.dat";

	//load the input image and initialize watermark object
    void SetUp() override 
    {
#pragma omp parallel
        {}
#if defined(_USE_OPENCL_)
        try {
            af::setDevice(openclDevice);
        }
        catch (const std::exception&) {
            cout << "NOTE: Invalid OpenCL device specified, using default 0" << "\n";
            af::setDevice(0);
        }
#endif
        Utils::loadImage(rgbImage, image, imageFile, alphaChannel);
        watermarkObj = Utils::createWatermarkObject(static_cast<unsigned int>(image.dims(0)), static_cast<unsigned int>(image.dims(1)), watermarkPath, p, psnr);
    }

	//helper method to embed watermark in the image (and check if it is successful based on watermark strength)
    BufferType embedWatermark(BufferType& image, BufferType& outputImage, float& strength, MASK_TYPE maskType) 
    {
        BufferType output = watermarkObj->makeWatermark(image, outputImage, strength, maskType);
        EXPECT_GT(strength, 0.0f) << "Expected strength > 0.0f, but got strength = " << strength;
        return output;
	}

	//helper method to save the watermarked image to disk and check if it matches the expected MSE threshold
    void saveToDiskTest(MASK_TYPE mask, const std::string& label, const std::string& outputFileName)
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
        EXPECT_LE(mse, mseThreshold) << "MSE for " << label << " is too high: " << mse << " , expected less than or equal to: " << mseThreshold;
    }
};

TEST_F(TestFixture, EmbedWatermark) 
{
    float strengthNvf = 0.0f, strengthMe = 0.0f;
    embedWatermark(image, rgbImage, strengthNvf, NVF);
	embedWatermark(image, rgbImage, strengthMe, ME);
	//watermark strength of Me should be at least as strong as NVF
	EXPECT_GE(strengthMe, strengthNvf) << "Expected strengthMe >= strengthNvf, but got strengthMe = " << strengthMe << " and strengthNvf = " << strengthNvf;
}

TEST_F(TestFixture, DetectWatermark) 
{
    float strengthNvf = 0.0f, strengthMe = 0.0f;
    const BufferType watermarkedNVFgray = af::rgb2gray(embedWatermark(image, rgbImage, strengthNvf, NVF), Constants::rPercent, Constants::gPercent, Constants::bPercent);
    const BufferType watermarkedMEgray = af::rgb2gray(embedWatermark(image, rgbImage, strengthMe, ME), Constants::rPercent, Constants::gPercent, Constants::bPercent);
    const float correlationNvf = watermarkObj->detectWatermark(watermarkedNVFgray, NVF);
    const float correlationMe = watermarkObj->detectWatermark(watermarkedMEgray, ME);
    //watermark correlation of Me should be at least as NVF
    EXPECT_GE(correlationMe, correlationNvf) << "Expected correlationMe >= correlationNvf, but got correlationMe = " << correlationMe << " and correlationNvf = " << correlationNvf;
}

TEST_F(TestFixture, SaveToDisk) 
{
    std::vector<MaskDiskConfig> strategies = 
    {
        { NVF, "W_NVF", "../../Watermarking-Impl/samples/images/4kW_NVF.png" },
        { ME,  "W_ME",  "../../Watermarking-Impl/samples/images/4kW_ME.png" }
    };
    for (const auto& config : strategies)
        saveToDiskTest(config.strategy, config.label, config.outputFile);
}