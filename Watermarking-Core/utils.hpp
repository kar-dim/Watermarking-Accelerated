#pragma once

#include "buffer.hpp"
#include "ImageFileBuffer.hpp"
#include "include/WatermarkCore.hpp"
#include <cstdint>
#include <string>
#include <vector>

/*!
 *  \brief  Helper functions dealing with GPU/Eigen types and image loading/saving (internal)
 *  \author Dimitris Karatzas
 */
namespace InternalUtils {
WatermarkCore::PreloadedHandle preloadImageSerial(const std::string& imagePath, int deviceIndex = -1);
ImageFileBuffer loadImage(const std::string& imageFile, bool captureOriginal = false, bool parallel = true);
void saveImage(const std::string& imagePath, const std::string& suffix, const ImageOutputBuffer& watermark, const std::vector<uint8_t>& alphaChannel, bool parallel = true);
ImageBuffer castToFloatGray(const ImageOutputBuffer& buffer, const bool isRGB);
// images loading at once in a batch (one image decodes on one thread, the rest of the CPU encodes the saves), also the nvJPEG decoders
int batchLoadThreads();
} // namespace InternalUtils
