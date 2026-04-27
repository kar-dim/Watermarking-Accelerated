#pragma once
#include "buffer.hpp"
#include "include/WatermarkTypes.hpp"
#include "WatermarkBase.hpp"
#include <cmath>
#include <string>
#include <vector>

#if defined(_USE_CUDA_)
#include "CudaStreamManager.hpp"
#include "GpuArray.hpp"
#include <cuda_runtime.h>
#elif defined(_USE_OPENCL_)
#include <arrayfire.h>
#include <concepts>
#endif

/*!
 *  \brief  Functions for watermark computation and detection, Base GPU class.
 *			GPU implementations must inherit from this class.
 *  \author Dimitris Karatzas
 */
template <int p>
class WatermarkGPU : public WatermarkBase {
  public:
#if defined(_USE_CUDA_)
    WatermarkGPU<p>(const unsigned int rows, const unsigned int cols, const std::string& watermarkPassword, const float psnr)
        : WatermarkBase(rows, cols, watermarkPassword, psnr, initializeRandomMatrix), strengthNumerator(strengthFactor * std::sqrt(static_cast<float>(this->totalPixels))),
          stream_(CudaStreamManager::getInstance().getComputeStream()), coefficients(localSize, stream_), stopFlag(GpuArray<int32_t>::zeros(1, stream_)) {}
#elif defined(_USE_OPENCL_)
    WatermarkGPU<p>(const unsigned int rows, const unsigned int cols, const std::string& watermarkPassword, const float psnr)
        : WatermarkBase(rows, cols, watermarkPassword, psnr, initializeRandomMatrix), strengthNumerator(strengthFactor * std::sqrt(static_cast<float>(this->totalPixels))) {}
#endif

    ~WatermarkGPU<p>() override = default;

    void makeWatermark(const ImageBuffer& inputGrayImage, const ImageBuffer& inputImage, ImageOutputBuffer& output, const MaskMethod maskType) override {
        output = computeStrengthenedWatermark(inputGrayImage, inputImage, maskType);
    }

    float detectWatermark(const ImageBuffer& inputImage, const MaskMethod maskType) override {
        auto errorSequenceW = computePredictionErrorData(inputImage, false);
        auto mask = maskType == MaskMethod::ME ? computePredictionErrorMask(errorSequenceW) : computeCustomMask(inputImage);
#if defined(_USE_OPENCL_)
        mask.eval();
#endif
        const float correlation = computeCorrelation(errorSequenceW, mask);
        return std::isfinite(correlation) ? correlation : 0.0f;
    }

#if defined(_USE_OPENCL_)
    template <std::same_as<af::array>... Args>
    static void unlockArrays(const Args&... arrays) {
        (arrays.unlock(), ...);
    }

    static void displayArray(const af::array& array, const int width = 1600, const int height = 900) {
        af::Window window(width, height);
        while (!window.close())
            window.image(array);
    }
#endif

  private:
#if defined(_USE_CUDA_)
    static ImageBuffer initializeRandomMatrix(const std::vector<float>& watermarkVec, const unsigned int rows, const unsigned int cols) {
        return GpuArray<float>(rows, cols, watermarkVec.data(), CudaStreamManager::getInstance().getComputeStream());
    }
#elif defined(_USE_OPENCL_)
    // clang-format off
    static af::array initializeRandomMatrix(const std::vector<float>& watermarkVec, const unsigned int rows, const unsigned int cols) {
        return af::array(rows, cols, watermarkVec.data());
    }
    // clang-format on
#endif

  protected:
    static constexpr int localSize = (p * p) - 1;
    float strengthNumerator;

#if defined(_USE_CUDA_)
    cudaStream_t stream_;
    GpuArray<float> coefficients;
    GpuArray<int32_t> stopFlag;

    virtual ImageOutputBuffer computeStrengthenedWatermark(const ImageBuffer& inputGrayImage, const ImageBuffer& inputImage, const MaskMethod maskType) const = 0;
    virtual ImageBuffer computeCustomMask(const ImageBuffer& image) const = 0;
    virtual ImageBuffer computeErrorSequence(const ImageBuffer& image, const bool calculateAbs) const = 0;
    virtual ImageBuffer computePredictionErrorData(const ImageBuffer& image, const bool calculateAbs) const = 0;
    virtual float computeCorrelation(const ImageBuffer& e_u, const ImageBuffer& e_z) const = 0;
    virtual ImageBuffer computePredictionErrorMask(const ImageBuffer& errorSequence) const = 0;
#elif defined(_USE_OPENCL_)
    af::array coefficients = af::array(localSize, f32);
    af::array stopFlag = af::constant(0, 1, s32);

    // computes u = x + [a * (M * W)] where x = input image, a = strength, M = computed mask, and W is the random noise matrix
    virtual af::array computeStrengthenedWatermark(const ImageBuffer& inputGrayImage, const ImageBuffer& inputImage, const MaskMethod maskType) const = 0;

    // computes custom Mask
    virtual af::array computeCustomMask(const af::array& image) const = 0;

    // computes error sequence, used in prediction error mask
    virtual af::array computeErrorSequence(const af::array& image, const bool calculateAbs) const = 0;

    // Used in both creation and detection of the watermark,
    // calculates error sequence and prediction error filter (coefficients)
    virtual af::array computePredictionErrorData(const af::array& image, const bool calculateAbs) const = 0;

    // helper method used in detectors
    virtual float computeCorrelation(const af::array& e_u, const af::array& e_z) const = 0;

    // compute prediction error mask as: abs(e) / max(abs(e)) where e is the error sequence. Used in ME mask type detector
    virtual af::array computePredictionErrorMask(const af::array& errorSequence) const = 0;
#endif
};