#include "gtest/gtest.h"

#include "buffer.hpp"
#include "constants.h"
#include "FileDeleter.h"
#include "MaskDiskConfig.h"
#include "utils.hpp"
#include "WatermarkBase.hpp"
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

using std::cout;
using std::string;

class CommonFixture : public ::testing::Test
{
protected:
    static constexpr float psnr = 40.0f;
    static constexpr float mseThreshold = 10.0f;
    static constexpr int p = 3;

    std::unique_ptr<WatermarkBase> watermarkObj;
    BufferType rgbImage, image;
    std::optional<BufferAlphaType> alphaChannel;
    const std::string imageFile = "../../Watermarking-Impl/samples/images/4k.png";
    const std::string watermarkPath = "../../Watermarking-Impl/samples/w_4k.dat";
    const std::vector<MaskDiskConfig> strategies =
    {
        { NVF, "W_NVF", "../../Watermarking-Impl/samples/images/4kW_NVF.png" },
        { ME,  "W_ME",  "../../Watermarking-Impl/samples/images/4kW_ME.png" }
    };

    //load the input image
    void SetUp() override
    {
        Utils::loadImage(rgbImage, image, imageFile, alphaChannel);
    }

    //initialize OpenMP once per fixture
    static void SetUpTestSuite()
    {
#pragma omp parallel
        {}
    }

    virtual BufferType embedAndConvertToGray(BufferType image, BufferType rgbImage, MASK_TYPE maskType) = 0;

    virtual void calculateMSE(const BufferType& diskRgb, const BufferType& watermark) = 0;

    //helper method to embed watermark in the image (and check if it is successful based on watermark strength)
    BufferType embedWatermark(BufferType& image, BufferType& outputImage, float& strength, MASK_TYPE maskType)
    {
        BufferType output = watermarkObj->makeWatermark(image, outputImage, strength, maskType);
        EXPECT_GT(strength, 0.0f);
        return output;
    }

    float calculateCorrelation(MASK_TYPE maskType)
    {
        return watermarkObj->detectWatermark(embedAndConvertToGray(image, rgbImage, maskType), maskType);
    }

	//helper methhod to embed watermark for both mask types and check if the strength of ME is at least as strong as NVF
    void testEmbedding() 
    {
        float strengthNvf = 0.0f, strengthMe = 0.0f;
        embedWatermark(image, rgbImage, strengthNvf, NVF);
        embedWatermark(image, rgbImage, strengthMe, ME);
        //for this specific test image we expect the below specific strengths
        EXPECT_NEAR(strengthNvf, 8.4817f, 0.1f);
        EXPECT_NEAR(strengthMe, 316.85f, 0.5f);
    }

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
		calculateMSE(diskRgb, watermark);
    }
};