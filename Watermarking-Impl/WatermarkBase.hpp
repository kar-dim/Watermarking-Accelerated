#pragma once

#include "buffer.hpp"
#include <cmath>
#include <fstream>
#if defined(_USE_GPU_)
#include <vector>
#endif
#include <stdexcept>
#include <string>

enum MASK_TYPE { ME, NVF };

/*!
 *  \brief  Functions for watermark computation and detection, Base class.
 *  \author Dimitris Karatzas
 */
class WatermarkBase {
  private:
    static inline float computeStrengthFactor(const float psnr) { return 255.0f / std::sqrt(std::pow(10.0f, psnr / 10.0f)); }

  public:
    WatermarkBase(const unsigned int rows, const unsigned int cols, const std::string& randomMatrixPath, const float psnr)
        : baseRows(rows), baseCols(cols), randomMatrix(loadRandomMatrix(randomMatrixPath)), strengthFactor(computeStrengthFactor(psnr)) {}

    WatermarkBase(const unsigned int rows, const unsigned int cols, const ImageBuffer& randomMatrix, const float strengthFactor)
        : baseRows(rows), baseCols(cols), randomMatrix(randomMatrix), strengthFactor(strengthFactor) {}

    //delete copy and move operations we don't wannt them
    WatermarkBase(const WatermarkBase&) = delete;
    WatermarkBase(WatermarkBase&&) = delete;
    WatermarkBase& operator=(const WatermarkBase&) = delete;
    WatermarkBase& operator=(WatermarkBase&&) = delete;

    virtual ~WatermarkBase() = default;

    // main watermark embedding method
    // it embeds the watermark computed from "inputGrayImage" (always grayscale, 2D)
    // into a new array "output" based on "inputImage" (RGB or grayscale always u8)
    virtual void makeWatermark(const ImageBuffer& inputGrayImage, const ImageBuffer& inputImage, ImageOutputBuffer& output, float& watermarkStrength, const MASK_TYPE maskType) = 0;

    // the main mask detector function
    virtual float detectWatermark(const ImageBuffer& inputImage, const MASK_TYPE maskType) = 0;

  protected:
    template <unsigned int ALIGN> static constexpr unsigned int align(const unsigned int x) { return (x + (ALIGN - 1)) & ~(ALIGN - 1); }
    static constexpr bool maskCalcRequired = true;
    static constexpr bool maskCalcNotRequired = false;
    unsigned int baseRows, baseCols;
    ImageBuffer randomMatrix;
    float strengthFactor;

    // helper method to load the random noise matrix W from the file specified.
    ImageBuffer loadRandomMatrix(const std::string& randomMatrixPath) const {
        std::ifstream randomMatrixStream(randomMatrixPath.c_str(), std::ios::binary);
        if (!randomMatrixStream.is_open())
            throw std::runtime_error(std::string("Error opening '" + randomMatrixPath + "' file for Random noise W array\n"));
        randomMatrixStream.seekg(0, std::ios::end);
        const size_t totalBytes = static_cast<size_t>(randomMatrixStream.tellg());
        randomMatrixStream.seekg(0, std::ios::beg);
        const size_t expectedBytes = static_cast<size_t>(baseRows) * baseCols * sizeof(float);
        if (expectedBytes != totalBytes)
            throw std::runtime_error(std::string("Error: W file total elements != image dimensions! W file total elements: " + std::to_string(totalBytes / (sizeof(float))) +
                                                 ", Image width: " + std::to_string(baseCols) + ", Image height: " + std::to_string(baseRows) + "\n"));
#if defined(_USE_GPU_)
        std::vector<float> watermarkBuffer(baseRows * baseCols);
        randomMatrixStream.read(reinterpret_cast<char*>(watermarkBuffer.data()), totalBytes);
        return af::transpose(af::array(baseCols, baseRows, watermarkBuffer.data()));
#elif defined(_USE_EIGEN_)
        Eigen::ArrayXXf watermark(baseCols, baseRows);
        randomMatrixStream.read(reinterpret_cast<char*>(watermark.data()), totalBytes);
        return ImageBuffer(watermark.transpose());
#endif
    }
};