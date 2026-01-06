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
        watermarkObj = Utils::createWatermarkObject(rows, cols, watermarkPath, p, psnr);
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
        ImageOutputBuffer watermarkedImage(eigen_utils::makeEigenRGBu8(rows, cols));
        embedWatermark(watermarkedImage, strength, maskType);
        return Utils::rgb2gray(Utils::castToFloat(watermarkedImage));
    }

    void calculateMSE(const ImageBuffer& diskRgb, const ImageOutputBuffer& watermark) override
    {
        EXPECT_EQ(diskRgb.getRGB()[0].size(), watermark.getRGB()[0].size()) << "Expected disk image elements to match original";
        float mse = 0.0f;
#pragma omp parallel for
        for (int i = 0; i < 3; i++)
            mse += (diskRgb.getRGB()[i].cast<uint8_t>() - watermark.getRGB()[i]).abs().sum();
        mse /= (3 * diskRgb.getRGB()[0].size());
        EXPECT_LE(mse, mseThreshold);
	}

    //helper method to embed watermark in the image (and check if it is successful based on watermark strength)
    ImageOutputBuffer embedWatermark(ImageOutputBuffer& output, float& strength, MASK_TYPE maskType) override
    {
        watermarkObj->makeWatermark(buf.image, buf.rgbImage, output, strength, maskType);
        EXPECT_GT(strength, 0.0f);
        return output;
    }

    //helper methhod to embed watermark for both mask types and check if the strength of ME is at least as strong as NVF
    void testEmbedding(ImageOutputBuffer& output) override
    {
        float strengthNvf = 0.0f, strengthMe = 0.0f;
        embedWatermark(output, strengthNvf, NVF);
        embedWatermark(output, strengthMe, ME);
        //for this specific test image we expect the below specific strengths
        EXPECT_NEAR(strengthNvf, 8.4817f, 0.1f);
        EXPECT_NEAR(strengthMe, 316.85f, 4.0f);
    }
};

TEST_F(EigenFixture, EmbedWatermark)
{
    ImageOutputBuffer output(eigen_utils::makeEigenRGBu8(rows, cols));
    testEmbedding(output);
}

TEST_F(EigenFixture, DetectWatermark)
{
    EXPECT_GE(calculateCorrelation(ME), calculateCorrelation(NVF));
}

TEST_F(EigenFixture, SaveToDisk)
{
    ImageOutputBuffer watermark(eigen_utils::makeEigenRGBu8(rows, cols));
    for (const auto& config : strategies)
        testSaveToDisk(watermark, config.strategy, config.label, config.outputFile);
}