#include "buffer.hpp"
#include "common_utils.hpp"
#include "ImageFileBuffer.hpp"
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
#include "CudaStreamManager.hpp"
#include "GpuArray.hpp"
#include "WatermarkCuda.cuh"
#include <algorithm>
#include <cctype>
#include <vector>
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

// save a CImg image selecting the correct encoder by file extension (shared by CUDA and Eigen builds)
#if defined(_USE_CUDA_) || defined(_USE_EIGEN_)
namespace {
void saveCimgByExtension(const Gray8BufferIO& cimgToSave, const string& path) {
    string extension = path.substr(path.find_last_of('.') + 1);
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
    if (extension == "png")
        cimgToSave.save_png(path.c_str());
    else if (extension == "bmp")
        cimgToSave.save_bmp(path.c_str());
    else if (extension == "jpg" || extension == "jpeg")
        cimgToSave.save_jpeg(path.c_str());
    else if (extension == "webp")
        cimgToSave.save_webp(path.c_str());
    else if (extension == "tif" || extension == "tiff")
        cimgToSave.save_tiff(path.c_str(), 20);
    else
        throw std::runtime_error("Unsupported image format: " + extension);
}
} // namespace
#endif

// CUDA helpers for CImg <-> GPU array conversion (column-major)
#if defined(_USE_CUDA_)
namespace {
GpuArray<float> cimgGrayToGpu(const FloatBufferIO& img, cudaStream_t stream) {
    const int rows = img.height();
    const int cols = img.width();
    std::vector<float> colMajor(rows * cols);
    for (int c = 0; c < cols; c++)
        for (int r = 0; r < rows; r++)
            colMajor[r + c * rows] = img(c, r);
    return GpuArray<float>(rows, cols, colMajor.data(), stream);
}

GpuArray<float> cimgRgbToGpu(const FloatBufferIO& img, cudaStream_t stream) {
    const int rows = img.height();
    const int cols = img.width();
    const int planeSize = rows * cols;
    std::vector<float> colMajor(planeSize * 3);
    for (int ch = 0; ch < 3; ch++)
        for (int c = 0; c < cols; c++)
            for (int r = 0; r < rows; r++)
                colMajor[r + c * rows + ch * planeSize] = img(c, r, 0, ch);
    return GpuArray<float>(rows, cols, 3, colMajor.data(), stream);
}

GpuArray<float> cimgRgbToGpuGray(const FloatBufferIO& img, cudaStream_t stream) {
    constexpr float rW = 0.299f, gW = 0.587f, bW = 0.114f;
    const int rows = img.height();
    const int cols = img.width();
    std::vector<float> colMajor(rows * cols);
    for (int c = 0; c < cols; c++)
        for (int r = 0; r < rows; r++)
            colMajor[r + c * rows] = img(c, r, 0, 0) * rW + img(c, r, 0, 1) * gW + img(c, r, 0, 2) * bW;
    return GpuArray<float>(rows, cols, colMajor.data(), stream);
}
} // namespace
#endif

void InternalUtils::saveImage(const string& imagePath, const string& suffix, const ImageOutputBuffer& watermark, const std::optional<Gray8BufferIO>& alphaChannel) {
#if defined(_USE_CUDA_)
    const string watermarkedFile = CommonUtils::addSuffixBeforeExtension(imagePath, suffix);
    const int rows = watermark.rows();
    const int cols = watermark.cols();
    const int channels = watermark.channels();
    const int planeSize = rows * cols;
    std::vector<uint8_t> hostData(watermark.size());
    watermark.toHost(hostData.data());
    const bool hasAlpha = alphaChannel.has_value();
    Gray8BufferIO output(cols, rows, 1, hasAlpha ? channels + 1 : channels);
    for (int ch = 0; ch < channels; ch++)
        for (int c = 0; c < cols; c++)
            for (int r = 0; r < rows; r++)
                output(c, r, 0, ch) = hostData[r + c * rows + ch * planeSize];
    if (hasAlpha)
        for (int c = 0; c < cols; c++)
            for (int r = 0; r < rows; r++)
                output(c, r, 0, channels) = (*alphaChannel)(c, r);
    saveCimgByExtension(output, watermarkedFile);
#elif defined(_USE_EIGEN_)
    const string watermarkedFile = CommonUtils::addSuffixBeforeExtension(imagePath, suffix);
    const auto cimgToSave = watermark.isRGB() ? eigen_utils::eigenRgbToCimg(watermark.getRGB(), alphaChannel) : eigen_utils::eigenGrayToCimg(watermark.getGray());
    saveCimgByExtension(cimgToSave, watermarkedFile);
#elif defined(_USE_OPENCL_)
    const ImageBuffer& arrayToSave = alphaChannel.has_value() ? af::join(2, watermark, *alphaChannel).as(u8) : watermark.as(u8);
    af::saveImageNative(CommonUtils::addSuffixBeforeExtension(imagePath, suffix).c_str(), arrayToSave);
#endif
}

std::unique_ptr<WatermarkBase> InternalUtils::createWatermarkObject(const unsigned int height, const unsigned int width, const string& watermarkPassword, const int p, const float psnr) {
#if defined(_USE_OPENCL_)
    switch (p) {
    case 3: return std::make_unique<WatermarkOCL<3>>(height, width, watermarkPassword, psnr); break;
    case 5: return std::make_unique<WatermarkOCL<5>>(height, width, watermarkPassword, psnr); break;
    case 7: return std::make_unique<WatermarkOCL<7>>(height, width, watermarkPassword, psnr); break;
    case 9: return std::make_unique<WatermarkOCL<9>>(height, width, watermarkPassword, psnr); break;
#elif defined(_USE_CUDA_)
    switch (p) {
    case 3: return std::make_unique<WatermarkCuda<3>>(height, width, watermarkPassword, psnr); break;
    case 5: return std::make_unique<WatermarkCuda<5>>(height, width, watermarkPassword, psnr); break;
    case 7: return std::make_unique<WatermarkCuda<7>>(height, width, watermarkPassword, psnr); break;
    case 9: return std::make_unique<WatermarkCuda<9>>(height, width, watermarkPassword, psnr); break;
#elif defined(_USE_EIGEN_)
    switch (p) {
    case 3: return std::make_unique<WatermarkEigen<3>>(height, width, watermarkPassword, psnr); break;
    case 5: return std::make_unique<WatermarkEigen<5>>(height, width, watermarkPassword, psnr); break;
    case 7: return std::make_unique<WatermarkEigen<7>>(height, width, watermarkPassword, psnr); break;
    case 9: return std::make_unique<WatermarkEigen<9>>(height, width, watermarkPassword, psnr); break;
#endif
    default: throw std::invalid_argument("Unsupported value for p. Allowed p values: 3, 5, 7, 9");
    }
}

void InternalUtils::rotate(FloatBufferIO& img, const uint16_t orientation) {
#if defined(_USE_OPENCL_)
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

ImageFileBuffer InternalUtils::loadImage(const string& imageFile) {
    ImageFileBuffer buf;
    auto& [rgbImage, image, alphaChannel, rows, cols, isRGB] = buf;

    std::ifstream fileStream(imageFile, std::ifstream::binary);
    TinyEXIF::EXIFInfo exif(fileStream);

#if defined(_USE_CUDA_)
    auto cimgRgb = FloatBufferIO(imageFile.c_str());
    InternalUtils::rotate(cimgRgb, exif.Orientation);
    auto stream = CudaStreamManager::getInstance().getComputeStream();

    switch (cimgRgb.spectrum()) {
    case 1:
        image = cimgGrayToGpu(cimgRgb, stream);
        rows = cimgRgb.height();
        cols = cimgRgb.width();
        break;
    case 3:
        rgbImage = cimgRgbToGpu(cimgRgb, stream);
        image = cimgRgbToGpuGray(cimgRgb, stream);
        rows = cimgRgb.height();
        cols = cimgRgb.width();
        isRGB = true;
        break;
    case 4: {
        alphaChannel.emplace(cimgRgb.get_shared_channel(3));
        auto rgbView = cimgRgb.get_shared_channels(0, 2);
        // zero RGB where alpha is zero
        for (int y = 0; y < cimgRgb.height(); y++)
            for (int x = 0; x < cimgRgb.width(); x++)
                if ((*alphaChannel)(x, y) == 0)
                    for (int ch = 0; ch < 3; ch++)
                        rgbView(x, y, 0, ch) = 0.0f;
        rgbImage = cimgRgbToGpu(rgbView, stream);
        image = cimgRgbToGpuGray(rgbView, stream);
        rows = cimgRgb.height();
        cols = cimgRgb.width();
        isRGB = true;
        break;
    }
    default: throw std::runtime_error("Invalid image dimensions");
    }
    cudaStreamSynchronize(stream);
#elif defined(_USE_OPENCL_)
    rgbImage = af::loadImageNative(imageFile.c_str()).as(f32);
    InternalUtils::rotate(rgbImage, exif.Orientation);
    switch (rgbImage.dims(2)) {
    case 1: image = rgbImage; break;
    case 3: image = InternalUtils::rgb2gray(rgbImage); break;
    case 4: {
        const af::array alpha = rgbImage(af::span, af::span, 3);
        alphaChannel.emplace(alpha.as(u8));
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
        alphaChannel.emplace(cimgRgb.get_shared_channel(3));
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
    return buf;
}

ImageBuffer InternalUtils::rgb2gray(const ImageBuffer& rgbImage) {
    constexpr float rPercent = 0.299f;
    constexpr float gPercent = 0.587f;
    constexpr float bPercent = 0.114f;
#if defined(_USE_CUDA_)
    const unsigned int totalPixels = rgbImage.rows() * rgbImage.cols();
    std::vector<float> hostRgb(rgbImage.size());
    rgbImage.toHost(hostRgb.data());
    std::vector<float> gray(totalPixels);
    for (unsigned int i = 0; i < totalPixels; i++)
        gray[i] = hostRgb[i] * rPercent + hostRgb[i + totalPixels] * gPercent + hostRgb[i + 2 * totalPixels] * bPercent;
    return GpuArray<float>(rgbImage.rows(), rgbImage.cols(), gray.data(), CudaStreamManager::getInstance().getComputeStream());
#elif defined(_USE_OPENCL_)
    return af::rgb2gray(rgbImage, rPercent, gPercent, bPercent);
#elif defined(_USE_EIGEN_)
    const auto& rgb = rgbImage.getRGB();
    return ((rgb[0] * rPercent) + (rgb[1] * gPercent) + (rgb[2] * bPercent)).eval();
#endif
}

ImageBuffer InternalUtils::castToFloatGray(const ImageOutputBuffer& buffer, const bool isRGB) {
    constexpr float rPercent = 0.299f;
    constexpr float gPercent = 0.587f;
    constexpr float bPercent = 0.114f;
#if defined(_USE_CUDA_)
    auto stream = CudaStreamManager::getInstance().getComputeStream();
    const unsigned int totalPixels = buffer.rows() * buffer.cols();
    std::vector<uint8_t> hostU8(buffer.size());
    buffer.toHost(hostU8.data());
    std::vector<float> hostF(totalPixels);
    if (isRGB) {
        for (unsigned int i = 0; i < totalPixels; i++)
            hostF[i] = static_cast<float>(hostU8[i]) * rPercent + static_cast<float>(hostU8[i + totalPixels]) * gPercent + static_cast<float>(hostU8[i + 2 * totalPixels]) * bPercent;
    } else {
        for (unsigned int i = 0; i < buffer.size(); i++)
            hostF[i] = static_cast<float>(hostU8[i]);
    }
    return GpuArray<float>(buffer.rows(), buffer.cols(), hostF.data(), stream);
#elif defined(_USE_OPENCL_)
    return isRGB ? af::rgb2gray(buffer.as(f32), rPercent, gPercent, bPercent) : buffer.as(f32);
#else
    if (isRGB) {
        const auto& rgbU8 = buffer.getRGB();
        return ImageBuffer((rgbU8[0].cast<float>() * rPercent + rgbU8[1].cast<float>() * gPercent + rgbU8[2].cast<float>() * bPercent).eval());
    } else {
        return ImageBuffer(buffer.getGray().cast<float>());
    }
#endif
}
