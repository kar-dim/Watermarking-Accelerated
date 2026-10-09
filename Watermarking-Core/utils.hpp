#pragma once

#include "buffer.hpp"
#include "ImageFileBuffer.hpp"
#include <cstdint>
#include <string>
#include <vector>

/*!
 *  \brief  Helper functions dealing with GPU/Eigen types and image loading/saving (internal)
 *  \author Dimitris Karatzas
 */
namespace InternalUtils {
ImageFileBuffer loadImage(const std::string& imageFile, bool captureOriginal = false);
void saveImage(const std::string& imagePath, const std::string& suffix, const ImageOutputBuffer& watermark, const std::vector<uint8_t>& alphaChannel);
ImageBuffer castToFloatGray(const ImageOutputBuffer& buffer, const bool isRGB);
// images loading at once in a batch (one image decodes on one thread, the rest of the CPU encodes the saves), also the nvJPEG decoders
int batchLoadThreads();
} // namespace InternalUtils
