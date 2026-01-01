#pragma once
#include "buffer.hpp"
#include <optional>

/*!
 *  \brief  Helper struct to hold image buffers and metadata loaded from file (JPEG, PNG, TIFF, etc).
 *  \author Dimitris Karatzas
 */
struct ImageFileBuffer
{
	ImageBuffer rgbImage;
	ImageBuffer image;
	std::optional<AlphaBuffer> alphaChannel;
	unsigned int rows = 0, cols = 0;
	bool isRGB = false;
};