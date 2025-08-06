#include "gtest/gtest.h"

#include "buffer.hpp"
#include "cimg_init.h"
#include "constants.h"
#include "eigen_utils.hpp"
#include "FileDeleter.h"
#include "MaskDiskConfig.h"
#include "test_common.hpp"
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

class EigenFixture : public CommonFixture
{
protected:
    //load the input image and initialize watermark object
    void SetUp() override 
    {
		CommonFixture::SetUp();
        omp_set_num_threads(std::max(omp_get_max_threads(), static_cast<int>(std::thread::hardware_concurrency())));
        watermarkObj = Utils::createWatermarkObject(static_cast<unsigned int>(image.getGray().rows()), static_cast<unsigned int>(image.getGray().cols()), watermarkPath, p, psnr);
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
        EXPECT_EQ(diskRgb.getRGB()[0].size(), watermark.getRGB()[0].size()) << "Expected disk image (" << label << ") elements to match original";
        float mse = 0.0f;
#pragma omp parallel for
        for (int i = 0; i < 3; i++)
            mse += (diskRgb.getRGB()[i] - watermark.getRGB()[i]).abs().sum();
        mse /= (3 * diskRgb.getRGB()[0].size());
        EXPECT_LE(mse, mseThreshold) << "MSE for " << label << " is too high: " << mse << " , expected less than or equal to: " << mseThreshold;
    }
};

TEST_F(EigenFixture, EmbedWatermark)
{
    testEmbedding();
}

TEST_F(EigenFixture, DetectWatermark)
{
    float strengthNvf = 0.0f, strengthMe = 0.0f;
    const BufferType watermarkedNVFgray(eigen_utils::eigenRgbToGray(embedWatermark(image, rgbImage, strengthNvf, NVF).getRGB(), Constants::rPercent, Constants::gPercent, Constants::bPercent));
    const BufferType watermarkedMEgray(eigen_utils::eigenRgbToGray(embedWatermark(image, rgbImage, strengthMe, ME).getRGB(), Constants::rPercent, Constants::gPercent, Constants::bPercent));
    const float correlationNvf = watermarkObj->detectWatermark(watermarkedNVFgray, NVF);
    const float correlationMe = watermarkObj->detectWatermark(watermarkedMEgray, ME);
    //watermark correlation of Me should be at least as NVF
    EXPECT_GE(correlationMe, correlationNvf) << "Expected correlationMe >= correlationNvf, but got correlationMe = " << correlationMe << " and correlationNvf = " << correlationNvf;
}
TEST_F(EigenFixture, SaveToDisk)
{
    std::vector<MaskDiskConfig> strategies = {
        { NVF, "W_NVF", "../../Watermarking-Impl/samples/images/4kW_NVF.png" },
        { ME,  "W_ME",  "../../Watermarking-Impl/samples/images/4kW_ME.png" }
    };
    for (const auto& config : strategies)
        testSaveToDisk(config.strategy, config.label, config.outputFile);
}