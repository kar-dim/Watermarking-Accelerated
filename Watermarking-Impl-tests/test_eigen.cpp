#include "gtest/gtest.h"

#include "buffer.hpp"
#include "cimg_init.h"
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

    BufferType embedAndConvertToGray(MASK_TYPE maskType) override
    {
        float strength = 0.0f;
        BufferType watermarkedImage(eigen_utils::makeEigenRGB(image.getGray().rows(), image.getGray().cols()));
        embedWatermark(watermarkedImage, strength, maskType);
        return Utils::rgb2gray(watermarkedImage);
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
    BufferType output(eigen_utils::makeEigenRGB(image.getGray().rows(), image.getGray().cols()));
    testEmbedding(output);
}

TEST_F(EigenFixture, DetectWatermark)
{
    EXPECT_GE(calculateCorrelation(ME), calculateCorrelation(NVF));
}

TEST_F(EigenFixture, SaveToDisk)
{
	BufferType watermark(eigen_utils::makeEigenRGB(image.getGray().rows(), image.getGray().cols()));
    for (const auto& config : strategies)
        testSaveToDisk(watermark, config.strategy, config.label, config.outputFile);
}