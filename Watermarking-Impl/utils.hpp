#pragma once

#include "buffer.hpp"
#include "common_utils.hpp"
#include "ImageFileBuffer.hpp"
#include "WatermarkBase.hpp"
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

/*!
 *  \brief  Helper functions dealing with GPU/Eigen types, image loading/saving, and watermark object creation (internal)
 *  \author Dimitris Karatzas
 */
namespace InternalUtils {
ImageFileBuffer loadImage(const std::string& imageFile, bool captureOriginal = false);
void saveImage(const std::string& imagePath, const std::string& suffix, const ImageOutputBuffer& watermark, const std::optional<Gray8BufferIO>& alphaChannel);
ImageBuffer castToFloatGray(const ImageOutputBuffer& buffer, const bool isRGB);
void rotate(FloatBufferIO& img, int orientation);
// the watermark depends only on the password and the size: a "previous" object of the same size gives its watermark to the new one (a p
// or channel count change) instead of generating it again. The previous object is destroyed before the new one allocates its buffers
std::unique_ptr<WatermarkBase> createWatermarkObject(
    const unsigned int height, const unsigned int width, const std::string& watermarkPassword, const int p, const float psnr, std::unique_ptr<WatermarkBase> previous = nullptr);
} // namespace InternalUtils
