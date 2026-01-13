#include "gtest/gtest.h"

#include "buffer.hpp"
#include "FileDeleter.h"
#include "ImageFileBuffer.hpp"
#include "MaskDiskConfig.h"
#include "utils.hpp"
#include "WatermarkBase.hpp"
#include <memory>
#include <string>
#include <vector>

class CommonFixture : public ::testing::Test {
  protected:
    static constexpr float psnr = 40.0f;
    static constexpr float mseThreshold = 10.0f;
    static constexpr int p = 3;

    std::unique_ptr<WatermarkBase> watermarkObj;
    ImageFileBuffer buf;
    unsigned int rows, cols;
    const std::string imageFile = "../../Watermarking-Impl/samples/images/4k.png";
    const std::string watermarkPath = "../../Watermarking-Impl/samples/w_4k.dat";
    inline static const std::vector<MaskDiskConfig> strategies = {{NVF, "W_NVF", "../../Watermarking-Impl/samples/images/4kW_NVF.png"},
                                                                  {ME, "W_ME", "../../Watermarking-Impl/samples/images/4kW_ME.png"}};

    // load the input image
    void SetUp() override {
        Utils::loadImage(buf, imageFile);
        rows = static_cast<unsigned int>(buf.rows);
        cols = static_cast<unsigned int>(buf.cols);
    }

    // delete the disk images (if exist)
    static void TearDownTestSuite() {
        for (const auto& strategy : strategies)
            FileDeleter cleanup(strategy.outputFile);
    }

    virtual ImageBuffer embedAndConvertToGray(MASK_TYPE maskType) = 0;

    virtual void calculateMSE(const ImageBuffer& diskRgb, const ImageOutputBuffer& watermark) = 0;

    // helper method to embed watermark in the image (and check if it is successful based on watermark strength)
    virtual ImageOutputBuffer embedWatermark(ImageOutputBuffer& output, float& strength, MASK_TYPE maskType) = 0;

    // helper methhod to embed watermark for both mask types and check if the strength of ME is at least as strong as NVF
    virtual void testEmbedding(ImageOutputBuffer& output) = 0;

    float calculateCorrelation(MASK_TYPE maskType) { return watermarkObj->detectWatermark(embedAndConvertToGray(maskType), maskType); }

    // helper method to save the watermarked image to disk and check if it matches the expected MSE threshold
    void testSaveToDisk(ImageOutputBuffer& watermark, MASK_TYPE mask, const std::string& label, const std::string& outputFileName) {
        float strength = 0.0f;
        embedWatermark(watermark, strength, mask);
        Utils::saveImage(imageFile, label, watermark, buf.alphaChannel);
        ImageFileBuffer diskBuf;
        Utils::loadImage(diskBuf, outputFileName);
        calculateMSE(diskBuf.rgbImage, watermark);
    }
};