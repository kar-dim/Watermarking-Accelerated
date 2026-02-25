#include "buffer.hpp"
#include "ImageFileBuffer.hpp"
#include "include/common_utils.hpp"
#include "TinyEXIF.h"
#include "utils.hpp"
#include "WatermarkBase.hpp"
#include <cstdint>
#include <fstream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#if defined(_USE_OPENCL_)
#include "WatermarkOCL.hpp"
#elif defined(_USE_CUDA_)
#include "WatermarkCuda.cuh"
#elif defined(_USE_EIGEN_)
#include <algorithm>
#include <cctype>
#include "cimg_init.h"
#include "eigen_rgb_array.hpp"
#include "eigen_utils.hpp"
#include "WatermarkEigen.hpp"
#endif

using std::string;
using namespace CommonUtils;

void InternalUtils::saveImage(const string& imagePath, const string& suffix, const ImageOutputBuffer& watermark, const std::optional<Gray8BufferIO>& alphaChannel) {
#if defined(_USE_EIGEN_)
    const string watermarkedFile = CommonUtils::addSuffixBeforeExtension(imagePath, suffix);
    string extension = watermarkedFile.substr(watermarkedFile.find_last_of('.') + 1);
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
    const auto cimgToSave = watermark.isRGB() ? eigen_utils::eigenRgbToCimg(watermark.getRGB(), alphaChannel) : eigen_utils::eigenGrayToCimg(watermark.getGray());
    if (extension == "png")
        cimgToSave.save_png(watermarkedFile.c_str());
    else if (extension == "bmp")
        cimgToSave.save_bmp(watermarkedFile.c_str());
    else if (extension == "jpg" || extension == "jpeg")
        cimgToSave.save_jpeg(watermarkedFile.c_str());
    else if (extension == "webp")
        cimgToSave.save_webp(watermarkedFile.c_str());
    else if (extension == "tif" || extension == "tiff")
        cimgToSave.save_tiff(watermarkedFile.c_str(), 20); // 20 = LZW compression
    else
        throw std::runtime_error("Unsupported image format: " + extension);
#elif defined(_USE_GPU_)
    const ImageBuffer& arrayToSave = alphaChannel.has_value() ? af::join(2, watermark, *alphaChannel).as(u8) : watermark.as(u8);
    af::saveImageNative(CommonUtils::addSuffixBeforeExtension(imagePath, suffix).c_str(), arrayToSave);
#endif
}

std::unique_ptr<WatermarkBase> InternalUtils::createWatermarkObject(const unsigned int height, const unsigned int width, const string& randomMatrixPath, const int p, const float psnr) {
#if defined(_USE_OPENCL_)
    switch (p) {
    case 3: return std::make_unique<WatermarkOCL<3>>(height, width, randomMatrixPath, psnr); break;
    case 5: return std::make_unique<WatermarkOCL<5>>(height, width, randomMatrixPath, psnr); break;
    case 7: return std::make_unique<WatermarkOCL<7>>(height, width, randomMatrixPath, psnr); break;
    case 9: return std::make_unique<WatermarkOCL<9>>(height, width, randomMatrixPath, psnr); break;
#elif defined(_USE_CUDA_)
    switch (p) {
    case 3: return std::make_unique<WatermarkCuda<3>>(height, width, randomMatrixPath, psnr); break;
    case 5: return std::make_unique<WatermarkCuda<5>>(height, width, randomMatrixPath, psnr); break;
    case 7: return std::make_unique<WatermarkCuda<7>>(height, width, randomMatrixPath, psnr); break;
    case 9: return std::make_unique<WatermarkCuda<9>>(height, width, randomMatrixPath, psnr); break;
#elif defined(_USE_EIGEN_)
    switch (p) {
    case 3: return std::make_unique<WatermarkEigen<3>>(height, width, randomMatrixPath, psnr); break;
    case 5: return std::make_unique<WatermarkEigen<5>>(height, width, randomMatrixPath, psnr); break;
    case 7: return std::make_unique<WatermarkEigen<7>>(height, width, randomMatrixPath, psnr); break;
    case 9: return std::make_unique<WatermarkEigen<9>>(height, width, randomMatrixPath, psnr); break;
#endif
    default: throw std::invalid_argument("Unsupported value for p. Allowed p values: 3, 5, 7, 9");
    }
}

// helper method to rotate an image based on EXIF metadata
void InternalUtils::rotate(FloatBufferIO& img, const uint16_t orientation) {
#if defined(_USE_GPU_)
    switch (orientation) {
    case 2: img = af::flip(img, 1); break;
    case 3:
        img = af::flip(img, 0);
        img = af::flip(img, 1);
        break;
    case 4: img = af::flip(img, 0); break;
    case 5:
        img = af::flip(img, 1);
        img = af::reorder(img, 1, 0, 2);
        img = af::flip(img, 0);
        break;
    case 6:
        img = af::reorder(img, 1, 0, 2);
        img = af::flip(img, 1);
        break;
    case 7:
        img = af::flip(img, 1);
        img = af::reorder(img, 1, 0, 2);
        img = af::flip(img, 1);
        break;
    case 8:
        img = af::reorder(img, 1, 0, 2);
        img = af::flip(img, 0);
        break;
    default: break;
    }
#else
    switch (orientation) {
    case 2: img.mirror('x'); break;
    case 3: img.rotate(180); break;
    case 4: img.mirror('y'); break;
    case 5:
        img.mirror('x');
        img.rotate(270);
        break;
    case 6: img.rotate(90); break;
    case 7:
        img.mirror('x');
        img.rotate(90);
        break;
    case 8: img.rotate(270); break;
    default: break;
    }
#endif
}

void InternalUtils::loadImage(ImageFileBuffer& buf, const string& imageFile) {
    auto& [rgbImage, image, alphaChannel, rows, cols, isRGB] = buf;
    // read file exif data for orientation
    std::ifstream fileStream(imageFile, std::ifstream::binary);
    TinyEXIF::EXIFInfo exif(fileStream);

#if defined(_USE_GPU_)
    rgbImage = af::loadImageNative(imageFile.c_str()).as(f32);
    InternalUtils::rotate(rgbImage, exif.Orientation);
    switch (rgbImage.dims(2)) {
    case 1: image = rgbImage; break;
    case 3: image = InternalUtils::rgb2gray(rgbImage); break;
    case 4: {
        const af::array alpha = rgbImage(af::span, af::span, 3);
        alphaChannel.emplace(alpha.as(u8)); // we want the alpha channel as 8-bit image later for saving the image with alpha
        rgbImage = rgbImage(af::span, af::span, af::seq(0, 2)) * (af::tile(alpha, 1, 1, 3) != 0);
        image = InternalUtils::rgb2gray(rgbImage);
        break;
    }
    default: throw std::runtime_error("Invalid image dimensions");
    }
    rows = static_cast<unsigned int>(image.dims(0));
    cols = static_cast<unsigned int>(image.dims(1));
    isRGB = rgbImage.dims(2) == 3;
    af::sync();
#elif defined(_USE_EIGEN_)
    auto cimgRgb = FloatBufferIO(imageFile.c_str());
    InternalUtils::rotate(cimgRgb, exif.Orientation);

    switch (cimgRgb.spectrum()) {
    case 1:
        rgbImage = eigen_utils::cimgToEigenGray(cimgRgb);
        image = rgbImage;
        break;
    case 3:
        rgbImage = eigen_utils::cimgToEigenRgb(cimgRgb);
        image = rgb2gray(rgbImage);
        break;
    case 4: {
        alphaChannel.emplace(cimgRgb.get_shared_channel(3)); // CImg copies the float into uint8 here
        auto rgbView = cimgRgb.get_shared_channels(0, 2);
        eigen_utils::cimgAlphaZero(rgbView, *alphaChannel);
        rgbImage = eigen_utils::cimgToEigenRgb(rgbView);
        image = rgb2gray(rgbImage);
        break;
    }
    default: throw std::runtime_error("Invalid image dimensions");
    }
    rows = image.getGray().rows();
    cols = image.getGray().cols();
    isRGB = rgbImage.isRGB();
#endif
}

ImageBuffer InternalUtils::rgb2gray(const ImageBuffer& rgbImage) {
    constexpr float rPercent = 0.299f;
    constexpr float gPercent = 0.587f;
    constexpr float bPercent = 0.114f;
#if defined(_USE_GPU_)
    return af::rgb2gray(rgbImage, rPercent, gPercent, bPercent);
#elif defined(_USE_EIGEN_)
    const auto& rgb = rgbImage.getRGB();
    return ((rgb[0] * rPercent) + (rgb[1] * gPercent) + (rgb[2] * bPercent)).eval();
#endif
}

ImageBuffer InternalUtils::castToFloat(const ImageOutputBuffer& buffer) {
#if defined(_USE_GPU_)
    return buffer.as(f32);
#else
    if (buffer.isRGB()) {
        const auto& rgbU8 = buffer.getRGB();
        EigenArrayRGB rgbFloat;
#pragma omp parallel for
        for (int channel = 0; channel < 3; channel++)
            rgbFloat[channel] = rgbU8[channel].cast<float>();
        return ImageBuffer(rgbFloat);
    } else
        return ImageBuffer(buffer.getGray().cast<float>());
#endif
}
