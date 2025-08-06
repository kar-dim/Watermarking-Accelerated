#include "gtest/gtest.h"

#include "buffer.hpp"
#include "cimg_init.h"
#include "constants.h"
#include "eigen_rgb_array.hpp"
#include "eigen_utils.hpp"
#include "FileDeleter.h"
#include "MaskDiskConfig.h"
#include "test_common.hpp"
#include "utils.hpp"
#include "WatermarkBase.hpp"
#include <Eigen/Dense>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <thread>

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

    void SetUp() override 
    {
		CommonFixture::SetUp();
        watermarkObj = Utils::createWatermarkObject(static_cast<unsigned int>(image.getGray().rows()), static_cast<unsigned int>(image.getGray().cols()), watermarkPath, p, psnr);
    }

    static void SetUpTestSuite()
    {
        CommonFixture::SetUpTestSuite();
    }

    BufferType embedAndConvertToGray(BufferType image, BufferType rgbImage, MASK_TYPE maskType) override
    {
        float strength = 0.0f;
        BufferType watermarkedImage([](int r, int c) 
            { return EigenArrayRGB { ArrayXXf(r, c), ArrayXXf(r, c), ArrayXXf(r, c) }; } (image.getGray().rows(), image.getGray().cols()));
        embedWatermark(image, rgbImage, watermarkedImage, strength, maskType);
        return eigen_utils::eigenRgbToGray(watermarkedImage.getRGB(), Constants::rPercent, Constants::gPercent, Constants::bPercent);
    }

    void calculateMSE(const BufferType& diskRgb, const BufferType& watermark) override
    {
        EXPECT_EQ(diskRgb.getRGB()[0].size(), watermark.getRGB()[0].size()) << "Expected disk image elements to match original";
        float mse = 0.0f;
#pragma omp parallel for
        for (int i = 0; i < 3; i++)
            mse += (diskRgb.getRGB()[i] - watermark.getRGB()[i]).abs().sum();
        mse /= (3 * diskRgb.getRGB()[0].size());
        EXPECT_LE(mse, mseThreshold);
	}
};

TEST_F(EigenFixture, EmbedWatermark)
{
    BufferType output([](int r, int c)
        { return EigenArrayRGB{ ArrayXXf(r, c), ArrayXXf(r, c), ArrayXXf(r, c) }; } (image.getGray().rows(), image.getGray().cols()));
    testEmbedding(output);
}

TEST_F(EigenFixture, DetectWatermark)
{
    EXPECT_GE(calculateCorrelation(ME), calculateCorrelation(NVF));
}

TEST_F(EigenFixture, SaveToDisk)
{
    for (const auto& config : strategies)
        testSaveToDisk(config.strategy, config.label, config.outputFile);
}