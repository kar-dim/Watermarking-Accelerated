#pragma once
#include "buffer.hpp"
#include "include/WatermarkTypes.hpp"
#include "WatermarkBase.hpp"
#include <arrayfire.h>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <vector>

/*!
 *  \brief  Functions for watermark computation and detection, Base GPU class.
 *			GPU implementations must inherit from this class.
 *  \author Dimitris Karatzas
 */
template <int p>
class WatermarkGPU : public WatermarkBase {
  public:
    WatermarkGPU<p>(const unsigned int rows, const unsigned int cols, const uint32_t watermarkSeed, const float psnr)
        : WatermarkBase(rows, cols, watermarkSeed, psnr, initializeRandomMatrix), strengthNumerator(strengthFactor * std::sqrt(static_cast<float>(this->totalPixels))) {}

    ~WatermarkGPU<p>() override = default;

    void makeWatermark(const ImageBuffer& inputGrayImage, const ImageBuffer& inputImage, ImageOutputBuffer& output, const MaskMethod maskType) override {
        output = computeStrengthenedWatermark(inputGrayImage, inputImage, maskType);
    }

    float detectWatermark(const ImageBuffer& inputImage, const MaskMethod maskType) override {
        const af::array errorSequenceW = computePredictionErrorData(inputImage, false);
        const af::array mask = maskType == MaskMethod::ME ? computePredictionErrorMask(errorSequenceW) : computeCustomMask(inputImage);
        mask.eval(); // we make sure mask is calculated, else arrayfire panics and deep copies!
        const float correlation = computeCorrelation(errorSequenceW, mask);
        return std::isfinite(correlation) ? correlation : 0.0f;
    }

    // helper method to unlock multiple af::arrays (return memory to ArrayFire)
    template <std::same_as<af::array>... Args>
    static void unlockArrays(const Args&... arrays) {
        (arrays.unlock(), ...);
    }

    // helper method to display an af::array in a window
    static void displayArray(const af::array& array, const int width = 1600, const int height = 900) {
        af::Window window(width, height);
        while (!window.close())
            window.image(array);
    }

  private:
    // initialize the watermark random matrix into an arrayfire array (copy from host to GPU VRAM)
    // clang-format off
    static af::array initializeRandomMatrix(const std::vector<float>& watermarkVec, const unsigned int rows, const unsigned int cols) { 
        return af::transpose(af::array(cols, rows, watermarkVec.data())); 
    }
    // clang-format on

  protected:
    static constexpr int localSize = (p * p) - 1;

    float strengthNumerator;
    af::array coefficients = af::array(localSize, f32);
    af::array stopFlag = af::constant(0, 1, s32);

    // computes u = a * (M * W) where a=strength, M=mask calculated and W is the random noise matrix
    virtual af::array computeStrengthenedWatermark(const ImageBuffer& inputGrayImage, const ImageBuffer& inputImage, const MaskMethod maskType) const = 0;

    // computes custom Mask
    virtual af::array computeCustomMask(const af::array& image) const = 0;

    // computes error sequence, used in prediction error mask
    virtual af::array computeErrorSequence(const af::array& image, const bool calculateAbs) const = 0;

    // computes error sequence between two inputs, used in correlation calculation. calculateAbs is always false
    virtual af::array computeErrorSequence(const af::array& inputA, const af::array& inputB) const = 0;

    // Used in both creation and detection of the watermark,
    // calculates error sequence and prediction error filter (coefficients)
    virtual af::array computePredictionErrorData(const af::array& image, const bool calculateAbs) const = 0;

    // helper method used in detectors
    virtual float computeCorrelation(const af::array& e_u, const af::array& e_z) const = 0;

    // compute prediction error mask as: abs(e) / max(abs(e)) where e is the error sequence. Used in ME mask type detector
    virtual af::array computePredictionErrorMask(const af::array& errorSequence) const = 0;
};