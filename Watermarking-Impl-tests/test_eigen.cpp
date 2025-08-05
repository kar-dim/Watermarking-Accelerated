#include "gtest/gtest.h"

#include "buffer.hpp"
#include "cimg_init.h"
#include "constants.h"
#include "eigen_utils.hpp"
#include "FileDeleter.h"
#include "MaskDiskConfig.h"
#include "utils.hpp"
#include "WatermarkBase.hpp"
#include <Eigen/Dense>
#include <iostream>
#include <memory>
#include <omp.h>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using namespace cimg_library;
using namespace Eigen;

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

    std::unique_ptr<WatermarkBase> watermarkObj;
    BufferType rgbImage, image;
    std::optional<BufferAlphaType> alphaChannel;
    const std::string imageFile = "../../Watermarking-Impl/samples/images/4k.png";
    const std::string watermarkPath = "../../Watermarking-Impl/samples/w_4k.dat";

    //load the input image and initialize watermark object
    void SetUp() override 
    {
        omp_set_num_threads(std::max(omp_get_max_threads(), static_cast<int>(std::thread::hardware_concurrency())));
#pragma omp parallel
        {}
        Utils::loadImage(rgbImage, image, imageFile, alphaChannel);
        watermarkObj = Utils::createWatermarkObject(static_cast<unsigned int>(image.getGray().rows()), static_cast<unsigned int>(image.getGray().cols()), watermarkPath, p, psnr);
    }

    //helper method to embed watermark in the image (and check if it is successful based on watermark strength)
    BufferType embedWatermark(BufferType& image, BufferType& outputImage, float& strength, MASK_TYPE maskType) 
    {
        BufferType output = std::move(watermarkObj->makeWatermark(image, outputImage, strength, maskType).getRGB());
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
        EXPECT_EQ(diskRgb.getRGB()[0].size(), watermark.getRGB()[0].size()) << "Expected disk image (" << label << ") elements to match original";
        float mse = 0.0f;
#pragma omp parallel for
        for (int i = 0; i < 3; i++)
            mse += (diskRgb.getRGB()[i] - watermark.getRGB()[i]).abs().sum();
        mse /= (3 * diskRgb.getRGB()[0].size());
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
    const BufferType watermarkedNVFgray(eigen_utils::eigenRgbToGray(embedWatermark(image, rgbImage, strengthNvf, NVF).getRGB(), Constants::rPercent, Constants::gPercent, Constants::bPercent));
    const BufferType watermarkedMEgray(eigen_utils::eigenRgbToGray(embedWatermark(image, rgbImage, strengthMe, ME).getRGB(), Constants::rPercent, Constants::gPercent, Constants::bPercent));
    const float correlationNvf = watermarkObj->detectWatermark(watermarkedNVFgray, NVF);
    const float correlationMe = watermarkObj->detectWatermark(watermarkedMEgray, ME);
    //watermark correlation of Me should be at least as NVF
    EXPECT_GE(correlationMe, correlationNvf) << "Expected correlationMe >= correlationNvf, but got correlationMe = " << correlationMe << " and correlationNvf = " << correlationNvf;
}
TEST_F(TestFixture, SaveToDisk) 
{
    std::vector<MaskDiskConfig> strategies = {
        { NVF, "W_NVF", "../../Watermarking-Impl/samples/images/4kW_NVF.png" },
        { ME,  "W_ME",  "../../Watermarking-Impl/samples/images/4kW_ME.png" }
    };
    for (const auto& config : strategies)
        saveToDiskTest(config.strategy, config.label, config.outputFile);
}