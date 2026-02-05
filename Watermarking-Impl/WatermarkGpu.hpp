#pragma once
#include "buffer.hpp"
#include "WatermarkBase.hpp"
#include <arrayfire.h>
#include <cmath>
#include <concepts>
#include <string>
#include <utility>

/*!
 *  \brief  Functions for watermark computation and detection, Base GPU class.
 *			GPU implementations must inherit from this class.
 *  \author Dimitris Karatzas
 */
template <int p> class WatermarkGPU : public WatermarkBase {
  public:
    WatermarkGPU<p>(const unsigned int rows, const unsigned int cols, const std::string& randomMatrixPath, const float psnr) : WatermarkBase(rows, cols, randomMatrixPath, psnr) {}

    WatermarkGPU<p>(const unsigned int rows, const unsigned int cols, const ImageBuffer& randomMatrix, const float strengthFactor) : WatermarkBase(rows, cols, randomMatrix, strengthFactor) {}

    ~WatermarkGPU<p>() override = default;

    void makeWatermark(const ImageBuffer& inputGrayImage, const ImageBuffer& inputImage, ImageOutputBuffer& output, float& watermarkStrength, const MASK_TYPE maskType) override {
        output = computeStrengthenedWatermark(inputGrayImage, inputImage, watermarkStrength, maskType);
    }

    float detectWatermark(const ImageBuffer& inputImage, const MASK_TYPE maskType) override {
        const af::array errorSequenceW = computePredictionErrorData(inputImage, false);
        const af::array mask = maskType == ME ? computePredictionErrorMask<true>(errorSequenceW) : computeCustomMask(inputImage);
        const float correlation = computeCorrelation(computeErrorSequence(mask, randomMatrix), errorSequenceW);
        return std::isfinite(correlation) ? correlation : 0.0f;
    }

    // helper method to unlock multiple af::arrays (return memory to ArrayFire)
    template <std::same_as<af::array>... Args> static void unlockArrays(const Args&... arrays) { (arrays.unlock(), ...); }

    // helper method to display an af::array in a window
    static void displayArray(const af::array& array, const int width = 1600, const int height = 900) {
        af::Window window(width, height);
        while (!window.close())
            window.image(array);
    }

  protected:
    static constexpr int pSquared = p * p;
    static constexpr int pad = p / 2;
    static constexpr int localSize = pSquared - 1;
    static constexpr int localSizeSq = localSize * localSize;

    af::array coefficients = af::array(localSize, f32);
    af::array stopFlag = af::constant(0, 1, s32);

    virtual af::array computeStrengthenedWatermark(const ImageBuffer& inputGrayImage, const ImageBuffer& inputImage, float& watermarkStrength, const MASK_TYPE maskType) const = 0;

    // computes custom Mask
    virtual af::array computeCustomMask(const af::array& image) const = 0;

    // computes error sequence, used in prediction error mask
    virtual af::array computeErrorSequence(const af::array& image, const bool calculateAbs) const = 0;

    // computes error sequence between two inputs, used in correlation calculation. calculateAbs is always false
    virtual af::array computeErrorSequence(const af::array& inputA, const af::array& inputB) const = 0;

    // Used in both creation and detection of the watermark.
    // Calculates error sequence and prediction error filter (coefficients)
    virtual af::array computePredictionErrorData(const af::array& image, const bool calculateAbs) const = 0;

    // helper method used in detectors
    virtual float computeCorrelation(const af::array& e_u, const af::array& e_z) const = 0;

    // compute prediction error mask
    template <bool CALC_ABS> af::array computePredictionErrorMask(const af::array& errorSequence) const {
        const af::array& input = CALC_ABS ? af::abs(errorSequence) : errorSequence;
        return input / (af::max(af::flat(input)) + 1.0e-6f);
    }

    // helper method to sum the incomplete RxPartial and rxPartial arrays which were produced from the custom "me" kernel
    // and to transform them to the correct size, so that they can be used by the system solver
    std::pair<af::array, af::array> transformCorrelationArrays(const af::array& RxPartial, const af::array& rxPartial) const {
        // reduction sum of blocks
        // all [p^2-1,1] blocks will be summed in rx
        // all [((p^2-1)(p^2))/2] vector blocks will be summed in Rx
        const auto totalBlocks = rxPartial.elements() / localSize;
        const auto RxStride = RxPartial.elements() / totalBlocks;
        const af::array rx = af::sum(af::moddims(rxPartial, localSize, totalBlocks), 1);
        const af::array Rx = af::sum(af::moddims(RxPartial, RxStride, totalBlocks), 1);
        return std::make_pair(Rx, rx);
    }
};