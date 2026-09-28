#pragma once

#include "buffer.hpp"
#include <cmath>
#include <string>
#include <utility>

/*!
 *  \brief  Functions for watermark computation and detection, Base class.
 *  \author Dimitris Karatzas
 */
class WatermarkBase {
  protected:
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
    // "watermark": the rows x cols watermark of the password, from generateWatermark (or from an object of the same size)
    WatermarkBase(const int rows, const int cols, WatermarkBuffer watermark, const float psnr)
        : baseRows(rows), baseCols(cols), totalPixels(baseRows * baseCols), randomMatrix(std::move(watermark)), strengthFactor(computeStrengthFactor(psnr)),
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
    virtual void makeWatermark(const ImageBuffer& inputGrayImage, const ImageOutputBuffer& inputImage, ImageOutputBuffer& output) = 0;

    // grayscale embedding (grayscale images, video luma): the watermark is added to "inputGrayImage", the output is written in
    // "outputLayout" (row-major video frames do NOT transpose later)
    virtual void makeWatermark(const ImageBuffer& inputGrayImage, ImageOutputBuffer& output, const Layout outputLayout) = 0;

    // the main mask detector function
    virtual float detectWatermark(const ImageBuffer& inputImage) = 0;

    // the watermark: rows x cols standard normal values (rounded to half precision) generated from the password with ChaCha20 and the
    // Box-Muller transform. The GPU backends generate it on the device, the bits are the same on every backend
    // NOTE: keep the implementation in .cpp file, NVCC (cuda) hangs when it reads the generation code in the header
    static WatermarkBuffer generateWatermark(const std::string& watermarkPassword, const int rows, const int cols);

    // the watermark depends only on the password and the size: a new object of the same size (another p) takes it instead of
    // generating it again. This object must not be used afterwards
    bool hasSize(const int rows, const int cols) const { return baseRows == rows && baseCols == cols; }
    WatermarkBuffer releaseWatermark() { return std::move(randomMatrix); }

    // PSNR affects only embedding strength. Updating it must not regenerate the
    // deterministic watermark or rebuild prediction workspaces.
    void updatePsnr(const float psnr) {
        strengthFactor = computeStrengthFactor(psnr);
        strengthNumerator = strengthFactor * std::sqrt(static_cast<float>(totalPixels));
    }

  private:
    static inline float computeStrengthFactor(const float psnr) { return 255.0f / std::sqrt(std::pow(10.0f, psnr / 10.0f)); }
};
