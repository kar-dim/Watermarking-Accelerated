#pragma once
#include "WatermarkBase.hpp"
#include <memory>
#include <string>

/*!
 *  \brief  Creates the watermark object of the backend (CUDA, OpenCL or Eigen) for a prediction window size
 *  \author Dimitris Karatzas
 */
namespace InternalUtils {
// the watermark depends only on the password and the size: a "previous" object of the same size gives its watermark to the new one (a p
// or channel count change) instead of generating it again. The previous object is destroyed before the new one allocates its buffers
std::unique_ptr<WatermarkBase> createWatermarkObject(
    const unsigned int height, const unsigned int width, const std::string& watermarkPassword, const int p, const float psnr, std::unique_ptr<WatermarkBase> previous = nullptr);
} // namespace InternalUtils
