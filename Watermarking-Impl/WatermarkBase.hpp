#pragma once

#include "buffer.hpp"
#include "include/WatermarkTypes.hpp"
#include <cmath>
#include <span>
#include <string>
#include <vector>

/*!
 *  \brief  Functions for watermark computation and detection, Base class.
 *  \author Dimitris Karatzas
 */
class WatermarkBase {
  protected:
    // creates the backend watermark buffer from the half precision bits of the generated watermark
    using WatermarkLoader = WatermarkBuffer (*)(std::span<const uint16_t>, const int, const int);

    template <int ALIGNMENT>
    static constexpr int alignUp(const int x) {
        static_assert(ALIGNMENT > 0 && (ALIGNMENT & (ALIGNMENT - 1)) == 0, "ALIGNMENT must be a power of 2");
        return (x + (ALIGNMENT - 1)) & ~(ALIGNMENT - 1);
    }
    int baseRows, baseCols, totalPixels;
    WatermarkBuffer randomMatrix;
    float strengthFactor;
    float strengthNumerator;

  public:
    WatermarkBase(const int rows, const int cols, const std::string& watermarkPassword, const float psnr, WatermarkLoader loader)
        : baseRows(rows), baseCols(cols), totalPixels(baseRows * baseCols), randomMatrix(generateRandomMatrix(watermarkPassword, loader)), strengthFactor(computeStrengthFactor(psnr)),
          strengthNumerator(strengthFactor * std::sqrt(static_cast<float>(totalPixels))) {}

    // delete copy and move operations we don't wannt them
    WatermarkBase(const WatermarkBase&) = delete;
    WatermarkBase(WatermarkBase&&) = delete;
    WatermarkBase& operator=(const WatermarkBase&) = delete;
    WatermarkBase& operator=(WatermarkBase&&) = delete;

    virtual ~WatermarkBase() = default;

    // layout of the 8-bit embedding output, column-major like every image plane of the backends, or row-major (the layout of video frames)
    enum class Layout { ColMajor, RowMajor };

    // RGB embedding: the watermark computed from the luma "inputGrayImage" is added to each channel of the 8-bit column-major planar
    // "inputImage", the output is column-major planar
    virtual void makeWatermark(const ImageBuffer& inputGrayImage, const ImageOutputBuffer& inputImage, ImageOutputBuffer& output, const MaskMethod maskType) = 0;

    // grayscale embedding (grayscale images, video luma): the watermark is added to "inputGrayImage", the output is written in
    // "outputLayout" (row-major video frames do NOT transpose later)
    virtual void makeWatermark(const ImageBuffer& inputGrayImage, ImageOutputBuffer& output, const MaskMethod maskType, const Layout outputLayout) = 0;

    // the main mask detector function
    virtual float detectWatermark(const ImageBuffer& inputImage, const MaskMethod maskType) = 0;

    // PSNR affects only embedding strength. Updating it must not regenerate the
    // deterministic watermark or rebuild prediction workspaces.
    void updatePsnr(const float psnr) {
        strengthFactor = computeStrengthFactor(psnr);
        strengthNumerator = strengthFactor * std::sqrt(static_cast<float>(totalPixels));
    }

  private:
    static inline float computeStrengthFactor(const float psnr) { return 255.0f / std::sqrt(std::pow(10.0f, psnr / 10.0f)); }

    // helper method to generate the watermark based on the given seed, using a parallelized approach with OpenMP for very fast generation
    // it generates secure random values based on ChaCha20 and uses Box-Muller transform to conver to gaussian random
    // NOTE: keep the implementation in .cpp file, NVCC (cuda) hangs when it reads this code in the header
    WatermarkBuffer generateRandomMatrix(const std::string& watermarkPassword, WatermarkLoader loader) const;
};
