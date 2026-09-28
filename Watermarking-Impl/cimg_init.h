#pragma once
#define cimg_display 0 // disable display features to save build time and binary size
#define cimg_use_png
#define cimg_use_jpeg
#define cimg_use_openmp
#define cimg_use_webp
#define cimg_use_tiff
#include <CImg.h>
#include <cstdint>
// the 8-bit image of the loaders and savers (planar, row-major)
using Gray8BufferIO = cimg_library::CImg<uint8_t>;
