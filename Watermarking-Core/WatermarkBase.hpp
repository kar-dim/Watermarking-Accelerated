#pragma once

#include "buffer.hpp"
#include "CheckedSize.hpp"
#include <cmath>
#include <string>
#include <utility>

/*!
 *  \brief  Base class for watermark computation and detection
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
    // rows x cols watermark of password, from generateWatermark or matching object
    WatermarkBase(const int rows, const int cols, WatermarkBuffer watermark, const float psnr)
        : baseRows(rows), baseCols(cols), totalPixels(InternalUtils::checkedElements(rows, cols)), randomMatrix(std::move(watermark)), strengthFactor(computeStrengthFactor(psnr)),
          strengthNumerator(strengthFactor * std::sqrt(static_cast<float>(totalPixels))) {}

    WatermarkBase(const WatermarkBase&) = delete;
    WatermarkBase(WatermarkBase&&) = delete;
    WatermarkBase& operator=(const WatermarkBase&) = delete;
    WatermarkBase& operator=(WatermarkBase&&) = delete;

    virtual ~WatermarkBase() = default;

    // layout of 8-bit embedding output, column-major for images, row-major for video
    enum class Layout { ColMajor, RowMajor };

    // RGB embedding: watermark from luma added to each channel of 8-bit planar image
    virtual void makeWatermark(const ImageBuffer& inputGrayImage, const ImageOutputBuffer& inputImage, ImageOutputBuffer& output) = 0;

    // grayscale embedding: watermark added to luma and written in outputLayout
    virtual void makeWatermark(const ImageBuffer& inputGrayImage, ImageOutputBuffer& output, const Layout outputLayout) = 0;

    virtual float detectWatermark(const ImageBuffer& inputImage) = 0;

    // watermark: rows x cols standard normal values generated with ChaCha20 and Box-Muller
    // implementation lives in cpp file, nvcc hangs when parsing generation in header
    static WatermarkBuffer generateWatermark(const std::string& watermarkPassword, const int rows, const int cols);

    // watermark reused across objects of same size and password, source invalidated
    bool hasSize(const int rows, const int cols) const { return baseRows == rows && baseCols == cols; }
    WatermarkBuffer releaseWatermark() { return std::move(randomMatrix); }

    // PSNR affects only strength, updates do not regenerate watermark or workspaces
    void updatePsnr(const float psnr) {
        strengthFactor = computeStrengthFactor(psnr);
        strengthNumerator = strengthFactor * std::sqrt(static_cast<float>(totalPixels));
    }

  private:
    static inline float computeStrengthFactor(const float psnr) { return 255.0f / std::sqrt(std::pow(10.0f, psnr / 10.0f)); }
};
