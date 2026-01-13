#include "buffer.hpp"
#include "utils.hpp"
#include "ImageFileBuffer.hpp"
#include "WatermarkBase.hpp"
#include <format>
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

string Utils::addSuffixBeforeExtension(const string& file, const string& suffix) {
    const auto dot = file.find_last_of('.');
    Utils::checkError(dot == string::npos || dot == file.size() - 1, "Filename has no valid extension: " + file);
    return file.substr(0, dot) + suffix + file.substr(dot);
}

void Utils::saveImage(const string& imagePath, const string& suffix, const ImageOutputBuffer& watermark, const std::optional<AlphaBuffer>& alphaChannel) {
#if defined(_USE_EIGEN_)
    const string watermarkedFile = Utils::addSuffixBeforeExtension(imagePath, suffix);
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
    af::saveImageNative(addSuffixBeforeExtension(imagePath, suffix).c_str(), arrayToSave);
#endif
}

std::unique_ptr<WatermarkBase> Utils::createWatermarkObject(const unsigned int height, const unsigned int width, const string& randomMatrixPath, const int p, const float psnr) {
#if defined(_USE_OPENCL_)
    switch (p) {
    case 3: return std::make_unique<WatermarkOCL<3>>(height, width, randomMatrixPath, psnr); break;
#elif defined(_USE_CUDA_)
    switch (p) {
    case 3: return std::make_unique<WatermarkCuda<3>>(height, width, randomMatrixPath, psnr); break;
    case 5: return std::make_unique<WatermarkCuda<5>>(height, width, randomMatrixPath, psnr); break;
#elif defined(_USE_EIGEN_)
    switch (p) {
    case 3: return std::make_unique<WatermarkEigen<3>>(height, width, randomMatrixPath, psnr); break;
    case 5: return std::make_unique<WatermarkEigen<5>>(height, width, randomMatrixPath, psnr); break;
    case 7: return std::make_unique<WatermarkEigen<7>>(height, width, randomMatrixPath, psnr); break;
    case 9: return std::make_unique<WatermarkEigen<9>>(height, width, randomMatrixPath, psnr); break;
#endif
    default: throw std::invalid_argument("Unsupported value for p. Allowed OpenCL p values: 3, CUDA: 3, 5, CPU: 3, 5, 7, 9");
    }
}

void Utils::checkError(const bool isError, const string& errorMsg) {
    if (isError)
        throw std::runtime_error(errorMsg);
};

// helper method to calculate execution time in FPS or in seconds
string Utils::formatExecutionTime(const bool showFps, const double seconds) { return showFps ? std::format("FPS: {:.2f} FPS", 1.0 / seconds) : std::format("{:.6f} seconds", seconds); }

void Utils::loadImage(ImageFileBuffer& buf, const string& imageFile) {
    auto& [rgbImage, image, alphaChannel, rows, cols, isRGB] = buf;
#if defined(_USE_GPU_)
    rgbImage = af::loadImageNative(imageFile.c_str()).as(f32);
    switch (rgbImage.dims(2)) {
    case 1: image = rgbImage; break;
    case 3: image = rgb2gray(rgbImage); break;
    case 4: {
        const af::array alpha = rgbImage(af::span, af::span, 3);
        alphaChannel.emplace(alpha.as(u8)); // we want the alpha channel as 8-bit image later for saving the image with alpha
        rgbImage = rgbImage(af::span, af::span, af::seq(0, 2)) * (af::tile(alpha, 1, 1, 3) != 0);
        image = rgb2gray(rgbImage);
        break;
    }
    default: throw std::runtime_error("Invalid image dimensions");
    }
    rows = static_cast<unsigned int>(image.dims(0));
    cols = static_cast<unsigned int>(image.dims(1));
    isRGB = rgbImage.dims(2) == 3;
    af::sync();
#elif defined(_USE_EIGEN_)
    auto cimgRgb = cimg_library::CImg<float>(imageFile.c_str());
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

ImageBuffer Utils::rgb2gray(const ImageBuffer& rgbImage) {
#if defined(_USE_GPU_)
    return af::rgb2gray(rgbImage, rPercent, gPercent, bPercent);
#elif defined(_USE_EIGEN_)
    const auto& rgb = rgbImage.getRGB();
    return ((rgb[0] * rPercent) + (rgb[1] * gPercent) + (rgb[2] * bPercent)).eval();
#endif
}

ImageBuffer Utils::castToFloat(const ImageOutputBuffer& buffer) {
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
