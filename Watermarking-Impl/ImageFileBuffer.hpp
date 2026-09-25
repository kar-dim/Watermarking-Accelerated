#pragma once
#include "buffer.hpp"
#include <cstdint>
#include <optional>
#include <vector>

/*!
 *  \brief  Helper struct to hold image buffers and metadata loaded from file (JPEG, PNG, TIFF, etc).
 *  \author Dimitris Karatzas
 */
struct ImageFileBuffer {
    // 8-bit RGB image (empty for grayscale images) and the float luma the watermark is computed from
    ImageOutputBuffer rgbImage;
    ImageBuffer image;
    std::optional<Gray8BufferIO> alphaChannel;
    unsigned int rows = 0, cols = 0;
    bool isRGB = false;
    std::vector<uint8_t> originalPreview;
    int previewChannels = 0;
};
