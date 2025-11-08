#include "gtest/gtest.h"

#include "buffer.hpp"
#include "MaskDiskConfig.h"
#include "test_common.hpp"
#include "utils.hpp"
#include "WatermarkBase.hpp"
#include <arrayfire.h>
#include <iostream>
#include <memory>
#include <string>

#if defined(_USE_OPENCL_)
#include <exception>
#endif

using std::cout;
using std::string;

/*!
 *  \brief  Main Test class used for testing watermarking algorithms
 *  \author Dimitris Karatzas
 */

class GpuFixture : public CommonFixture 
{
protected:

    void SetUp() override 
    {
        CommonFixture::SetUp();
        watermarkObj = Utils::createWatermarkObject(static_cast<unsigned int>(image.dims(0)), static_cast<unsigned int>(image.dims(1)), watermarkPath, p, psnr);
    }

    static void SetUpTestSuite() 
    {
#if defined(_USE_OPENCL_)
        static constexpr int openclDevice = 1;
        try {
            af::setDevice(openclDevice);
        }
        catch (const std::exception&) {
            cout << "NOTE: Invalid OpenCL device specified, using default 0\n";
            af::setDevice(0);
        }
#endif
    }

    static void TearDownTestSuite() { CommonFixture::TearDownTestSuite(); }

    ImageBuffer embedAndConvertToGray(MASK_TYPE maskType) override
    {
        float strength = 0.0f;
        ImageBuffer watermarkedImage;
        return Utils::rgb2gray(embedWatermark(watermarkedImage, strength, maskType));
    }

    void calculateMSE(const ImageBuffer& diskRgb, const ImageBuffer& watermark) override
    {
        EXPECT_EQ(diskRgb.elements(), watermark.elements()) << "Expected disk image elements to match original";
        const float mse = af::sum<float>(af::abs(diskRgb - watermark)) / diskRgb.elements();
        EXPECT_LE(mse, mseThreshold);
    }
};

TEST_F(GpuFixture, EmbedWatermark)
{
    ImageBuffer output;
    testEmbedding(output);
}

TEST_F(GpuFixture, DetectWatermark)
{
    EXPECT_GE(calculateCorrelation(ME), calculateCorrelation(NVF));
}

TEST_F(GpuFixture, SaveToDisk)
{
    ImageBuffer watermark;
    for (const auto& config : strategies)
        testSaveToDisk(watermark, config.strategy, config.label, config.outputFile);
}