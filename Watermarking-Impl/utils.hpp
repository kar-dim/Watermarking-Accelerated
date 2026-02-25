#pragma once

#include "buffer.hpp"
#include "ImageFileBuffer.hpp"
#include "WatermarkBase.hpp"
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

/*!
 *  \brief  Helper functions dealing with ArrayFire/Eigen types, image loading/saving, and watermark object creation (internal)
 *  \author Dimitris Karatzas
 */
namespace InternalUtils {
void loadImage(ImageFileBuffer& buf, const std::string& imageFile);
void saveImage(const std::string& imagePath, const std::string& suffix, const ImageOutputBuffer& watermark, const std::optional<Gray8BufferIO>& alphaChannel);
ImageBuffer rgb2gray(const ImageBuffer& rgbImage);
ImageBuffer castToFloat(const ImageOutputBuffer& buffer);
void rotate(FloatBufferIO& img, uint16_t orientation);
std::unique_ptr<WatermarkBase> createWatermarkObject(const unsigned int height, const unsigned int width, const std::string& randomMatrixPath, const int p, const float psnr);
} // namespace InternalUtils