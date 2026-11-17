#pragma once
#include "WatermarkBase.hpp"
#include <memory>
#include <string>

/*!
 *  \brief  Creates the watermark object of the backend (CUDA, OpenCL or Eigen) for a prediction window size
 *  \author Dimitris Karatzas
 */
namespace InternalUtils {
std::unique_ptr<WatermarkBase> createWatermarkObject(unsigned int height, unsigned int width, const std::string& watermarkPassword, int p, float psnr);
} // namespace InternalUtils
