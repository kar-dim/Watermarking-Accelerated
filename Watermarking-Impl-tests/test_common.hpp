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

    //load the input image and initialize watermark object
    void SetUp() override
    {
        Utils::loadImage(rgbImage, image, imageFile, alphaChannel);
#pragma omp parallel
        {}
    }

    //helper method to embed watermark in the image (and check if it is successful based on watermark strength)
    BufferType embedWatermark(BufferType& image, BufferType& outputImage, float& strength, MASK_TYPE maskType)
    {
        BufferType output = watermarkObj->makeWatermark(image, outputImage, strength, maskType);
        EXPECT_GT(strength, 0.0f);
        return output;
    }

	//helper methhod to embed watermark for both mask types and check if the strength of ME is at least as strong as NVF
    void testEmbedding() 
    {
        float strengthNvf = 0.0f, strengthMe = 0.0f;
        embedWatermark(image, rgbImage, strengthNvf, NVF);
        embedWatermark(image, rgbImage, strengthMe, ME);
        //watermark strength of Me should be at least as strong as NVF
        EXPECT_GE(strengthMe, strengthNvf);
    }

    //helper method to save the watermarked image to disk and check if it matches the expected MSE threshold
    //void testSaveToDisk(MASK_TYPE mask, const std::string& label, const std::string& outputFileName)
    //{
    //    float strength = 0.0f;
    //    FileDeleter cleanup(outputFileName); //delete the file after the test
    //    const BufferType watermark = embedWatermark(image, rgbImage, strength, mask);
    //    Utils::saveImage(imageFile, label, watermark, alphaChannel);
    //    BufferType diskRgb, diskImage;
    //    std::optional<BufferAlphaType> diskAlpha;
    //    Utils::loadImage(diskRgb, diskImage, outputFileName, diskAlpha);
    //    EXPECT_EQ(diskRgb.elements(), watermark.elements()) << "Expected disk image (" << label << ") elements to match original";
    //    const float mse = af::sum<float>(af::abs(diskRgb - watermark)) / diskRgb.elements();
    //    EXPECT_LE(mse, mseThreshold) << "MSE for " << label << " is too high: " << mse << " , expected less than or equal to: " << mseThreshold;
    //}
};