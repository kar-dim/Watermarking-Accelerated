#include "gtest/gtest.h"

#include "buffer.hpp"
#include "FileDeleter.h"
#include "MaskDiskConfig.h"
#include "utils.hpp"
#include "WatermarkBase.hpp"
#include <memory>
#include <optional>
#include <string>
#include <vector>

class CommonFixture : public ::testing::Test
{
protected:
    static constexpr float psnr = 40.0f;
    static constexpr float mseThreshold = 10.0f;
    static constexpr int p = 3;

    std::unique_ptr<WatermarkBase> watermarkObj;
    ImageBuffer rgbImage, image;
    std::optional<AlphaBuffer> alphaChannel;
    const std::string imageFile = "../../Watermarking-Impl/samples/images/4k.png";
    const std::string watermarkPath = "../../Watermarking-Impl/samples/w_4k.dat";
    inline static const std::vector<MaskDiskConfig> strategies =
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

    //delete the disk images (if exist)
    static void TearDownTestSuite()
    {
        for (const auto& strategy : strategies)
            FileDeleter cleanup(strategy.outputFile);
    }

    virtual ImageBuffer embedAndConvertToGray(MASK_TYPE maskType) = 0;

    virtual void calculateMSE(const ImageBuffer& diskRgb, const ImageBuffer& watermark) = 0;

    //helper method to embed watermark in the image (and check if it is successful based on watermark strength)
    ImageBuffer embedWatermark(ImageBuffer& output, float& strength, MASK_TYPE maskType)
    {
        watermarkObj->makeWatermark(image, rgbImage, output, strength, maskType);
        EXPECT_GT(strength, 0.0f);
        return output;
    }

    float calculateCorrelation(MASK_TYPE maskType)
    {
        return watermarkObj->detectWatermark(embedAndConvertToGray(maskType), maskType);
    }

	//helper methhod to embed watermark for both mask types and check if the strength of ME is at least as strong as NVF
    void testEmbedding(ImageBuffer& output)
    {
        float strengthNvf = 0.0f, strengthMe = 0.0f;
        embedWatermark(output, strengthNvf, NVF);
        embedWatermark(output, strengthMe, ME);
        //for this specific test image we expect the below specific strengths
        EXPECT_NEAR(strengthNvf, 8.4817f, 0.1f);
        EXPECT_NEAR(strengthMe, 316.85f, 4.0f);
    }

    //helper method to save the watermarked image to disk and check if it matches the expected MSE threshold
    void testSaveToDisk(ImageBuffer& watermark, MASK_TYPE mask, const std::string& label, const std::string& outputFileName)
    {
        float strength = 0.0f;
        embedWatermark(watermark, strength, mask);
        Utils::saveImage(imageFile, label, watermark, alphaChannel);
        ImageBuffer diskRgb, diskImage;
        std::optional<AlphaBuffer> diskAlpha;
        Utils::loadImage(diskRgb, diskImage, outputFileName, diskAlpha);
		calculateMSE(diskRgb, watermark);
    }
};