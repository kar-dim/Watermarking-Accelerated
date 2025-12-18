#include "gtest/gtest.h"

#include "buffer.hpp"
#include "eigen_utils.hpp"
#include "MaskDiskConfig.h"
#include "test_common.hpp"
#include "utils.hpp"
#include "WatermarkBase.hpp"
#include <Eigen/Dense>
#include <iostream>
#include <memory>
#include <string>

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

    //initialize OpenMP once per fixture
    static void SetUpTestSuite()
    {
#pragma omp parallel
        {}
    }

    static void TearDownTestSuite() { CommonFixture::TearDownTestSuite(); }

    ImageBuffer embedAndConvertToGray(MASK_TYPE maskType) override
    {
        float strength = 0.0f;
        ImageBuffer watermarkedImage(eigen_utils::makeEigenRGB(image.getGray().rows(), image.getGray().cols()));
        embedWatermark(watermarkedImage, strength, maskType);
        return Utils::rgb2gray(watermarkedImage);
    }

    void calculateMSE(const ImageBuffer& diskRgb, const ImageBuffer& watermark) override
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
    ImageBuffer output(eigen_utils::makeEigenRGB(image.getGray().rows(), image.getGray().cols()));
    testEmbedding(output);
}

TEST_F(EigenFixture, DetectWatermark)
{
    EXPECT_GE(calculateCorrelation(ME), calculateCorrelation(NVF));
}

TEST_F(EigenFixture, SaveToDisk)
{
    ImageBuffer watermark(eigen_utils::makeEigenRGB(image.getGray().rows(), image.getGray().cols()));
    for (const auto& config : strategies)
        testSaveToDisk(watermark, config.strategy, config.label, config.outputFile);
}