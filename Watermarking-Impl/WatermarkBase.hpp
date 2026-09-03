#pragma once

#include "buffer.hpp"
#include "include/WatermarkTypes.hpp"
#include <cmath>
#include <string>
#include <vector>

/*!
 *  \brief  Functions for watermark computation and detection, Base class.
 *  \author Dimitris Karatzas
 */
class WatermarkBase {
  protected:
    using WatermarkLoader = ImageBuffer (*)(const std::vector<float>&, const int, const int);

    template <int ALIGNMENT>
    static constexpr int alignUp(const int x) {
        static_assert(ALIGNMENT > 0 && (ALIGNMENT & (ALIGNMENT - 1)) == 0, "ALIGNMENT must be a power of 2");
        return (x + (ALIGNMENT - 1)) & ~(ALIGNMENT - 1);
    }
    int baseRows, baseCols, totalPixels;
    ImageBuffer randomMatrix;
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

    // main watermark embedding method
    // it embeds the watermark computed from "inputGrayImage" (always grayscale, 2D)
    // into a new array "output" based on "inputImage" (RGB or grayscale always u8)
    virtual void makeWatermark(const ImageBuffer& inputGrayImage, const ImageBuffer& inputImage, ImageOutputBuffer& output, const MaskMethod maskType) = 0;

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
    ImageBuffer generateRandomMatrix(const std::string& watermarkPassword, WatermarkLoader loader) const;
};
