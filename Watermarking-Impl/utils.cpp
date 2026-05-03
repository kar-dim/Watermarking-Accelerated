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
#include "OclQueueManager.hpp"
#include "OclArray.hpp"
#include "opencl_utils.hpp"
#include "WatermarkOCL.hpp"
#include <algorithm>
#include <cctype>
#include <vector>
#elif defined(_USE_CUDA_)
#include "CudaStreamManager.hpp"
#include "CudaArray.hpp"
#include "WatermarkCuda.cuh"
#include "cuda_utils.hpp"
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

// save a CImg image selecting the correct encoder by file extension
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

// standard ITU-R 601 Luma coefficients for RGB to Grayscale conversion
constexpr float kRW = 0.299f;
constexpr float kGW = 0.587f;
constexpr float kBW = 0.114f;
} // namespace

// GPU helpers for CImg (row-major) <-> GPU array (column-major) conversion
#if defined(_USE_GPU_)
namespace {

#if defined(_USE_CUDA_)
// For CUDA: upload the raw row-major CImg data to a temp GPU buffer, then transpose in-place on GPU.
// This avoids CPU-side std::vector allocation and uses a coalesced tiled transpose kernel.
ImageBuffer cimgGrayToGpu(const FloatBufferIO& img, cudaStream_t stream) {
    const int rows = img.height();
    const int cols = img.width();
    CudaArray<float> rowMajor(rows, cols, img.data(), stream);
    CudaArray<float> colMajor(rows, cols, stream);
    cuda_utils::launchRowMajorToColMajorFloatKernel(rowMajor.data(), colMajor.data(), cols, rows, 1, stream);
    return colMajor;
}

ImageBuffer cimgRgbToGpu(const FloatBufferIO& img, cudaStream_t stream) {
    const int rows = img.height();
    const int cols = img.width();
    CudaArray<float> rowMajor(rows, cols, 3, img.data(), stream);
    CudaArray<float> colMajor(rows, cols, 3, stream);
    cuda_utils::launchRowMajorToColMajorFloatKernel(rowMajor.data(), colMajor.data(), cols, rows, 3, stream);
    return colMajor;
}

ImageBuffer cimgRgbToGpuGray(const FloatBufferIO& img, cudaStream_t stream) {
    const int rows = img.height();
    const int cols = img.width();
    const int planeSize = rows * cols;
    std::vector<float> gray(planeSize);
    const float* R = img.data();
    const float* G = R + planeSize;
    const float* B = G + planeSize;
    for (int i = 0; i < planeSize; i++)
        gray[i] = R[i] * kRW + G[i] * kGW + B[i] * kBW;
    CudaArray<float> rowMajorGray(rows, cols, gray.data(), stream);
    CudaArray<float> colMajorGray(rows, cols, stream);
    cuda_utils::launchRowMajorToColMajorFloatKernel(rowMajorGray.data(), colMajorGray.data(), cols, rows, 1, stream);
    return colMajorGray;
}

#elif defined(_USE_OPENCL_)
// For OCL: upload raw row-major CImg data to GPU, then transpose with a coalesced tiled kernel (mirrors CUDA path).
ImageBuffer cimgGrayToGpu(const FloatBufferIO& img, cl_command_queue queue) {
    const int rows = img.height();
    const int cols = img.width();
    OclArray<float> rowMajor(rows, cols, img.data(), queue);
    OclArray<float> colMajor(rows, cols, queue);
    auto& q = OclQueueManager::getInstance().getQueue();
    cl_utils::launchRowMajorToColMajorFloat(rowMajor.clBuffer(), colMajor.clBuffer(), cols, rows, 1, q);
    return colMajor;
}

ImageBuffer cimgRgbToGpu(const FloatBufferIO& img, cl_command_queue queue) {
    const int rows = img.height();
    const int cols = img.width();
    OclArray<float> rowMajor(rows, cols, 3, img.data(), queue);
    OclArray<float> colMajor(rows, cols, 3, queue);
    auto& q = OclQueueManager::getInstance().getQueue();
    cl_utils::launchRowMajorToColMajorFloat(rowMajor.clBuffer(), colMajor.clBuffer(), cols, rows, 3, q);
    return colMajor;
}

ImageBuffer cimgRgbToGpuGray(const FloatBufferIO& img, cl_command_queue queue) {
    const int rows = img.height();
    const int cols = img.width();
    const int planeSize = rows * cols;
    std::vector<float> gray(planeSize);
    const float* R = img.data();
    const float* G = R + planeSize;
    const float* B = G + planeSize;
    for (int i = 0; i < planeSize; i++)
        gray[i] = R[i] * kRW + G[i] * kGW + B[i] * kBW;
    OclArray<float> rowMajorGray(rows, cols, gray.data(), queue);
    OclArray<float> colMajorGray(rows, cols, queue);
    auto& q = OclQueueManager::getInstance().getQueue();
    cl_utils::launchRowMajorToColMajorFloat(rowMajorGray.clBuffer(), colMajorGray.clBuffer(), cols, rows, 1, q);
    return colMajorGray;
}
#endif

} // namespace
#endif

void InternalUtils::saveImage(const string& imagePath, const string& suffix, const ImageOutputBuffer& watermark, const std::optional<Gray8BufferIO>& alphaChannel) {
    const string watermarkedFile = CommonUtils::addSuffixBeforeExtension(imagePath, suffix);
#if defined(_USE_GPU_)
    const int rows = watermark.getRows();
    const int cols = watermark.getCols();
    const int channels = watermark.getChannels();
    const bool hasAlpha = alphaChannel.has_value();
#if defined(_USE_CUDA_)
    auto stream = CudaStreamManager::getInstance().getComputeStream();
    CudaArray<uint8_t> rowMajor(rows, cols, channels, stream);
    cuda_utils::launchColMajorToRowMajorU8Kernel(watermark.data(), rowMajor.data(), cols, rows, channels, stream);
#elif defined(_USE_OPENCL_)
    auto& mgr = OclQueueManager::getInstance();
    OclArray<uint8_t> rowMajor(rows, cols, channels, mgr.getQueueRaw());
    cl_utils::launchColMajorToRowMajorU8(watermark.clBuffer(), rowMajor.clBuffer(), cols, rows, channels, mgr.getQueue());
#endif
    Gray8BufferIO output(cols, rows, 1, hasAlpha ? channels + 1 : channels);
    rowMajor.toHost(output.data());
    if (hasAlpha)
        for (int c = 0; c < cols; c++)
            for (int r = 0; r < rows; r++)
                output(c, r, 0, channels) = (*alphaChannel)(c, r);
    saveCimgByExtension(output, watermarkedFile);
#elif defined(_USE_EIGEN_)
    const auto cimgToSave = watermark.isRGB() ? eigen_utils::eigenRgbToCimg(watermark.getRGB(), alphaChannel) : eigen_utils::eigenGrayToCimg(watermark.getGray());
    saveCimgByExtension(cimgToSave, watermarkedFile);
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
}

ImageFileBuffer InternalUtils::loadImage(const string& imageFile) {
    ImageFileBuffer buf;
    auto& [rgbImage, image, alphaChannel, rows, cols, isRGB] = buf;
    std::ifstream fileStream(imageFile, std::ifstream::binary);
    TinyEXIF::EXIFInfo exif(fileStream); // parse EXIF for orientation
    auto cimgRgb = FloatBufferIO(imageFile.c_str());
    InternalUtils::rotate(cimgRgb, exif.Orientation); // optional rotate (if required)
    rows = cimgRgb.height();
    cols = cimgRgb.width();

#if defined(_USE_CUDA_)
    auto stream = CudaStreamManager::getInstance().getComputeStream();

    switch (cimgRgb.spectrum()) {
    case 1: image = cimgGrayToGpu(cimgRgb, stream); break;
    case 3:
        rgbImage = cimgRgbToGpu(cimgRgb, stream);
        image = cimgRgbToGpuGray(cimgRgb, stream);
        isRGB = true;
        break;
    case 4: {
        alphaChannel.emplace(cimgRgb.get_shared_channel(3));
        auto rgbView = cimgRgb.get_shared_channels(0, 2);
        for (int y = 0; y < cimgRgb.height(); y++)
            for (int x = 0; x < cimgRgb.width(); x++)
                if ((*alphaChannel)(x, y) == 0)
                    for (int ch = 0; ch < 3; ch++)
                        rgbView(x, y, 0, ch) = 0.0f;
        rgbImage = cimgRgbToGpu(rgbView, stream);
        image = cimgRgbToGpuGray(rgbView, stream);
        isRGB = true;
        break;
    }
    default: throw std::runtime_error("Invalid image dimensions");
    }
    cudaStreamSynchronize(stream);
#elif defined(_USE_OPENCL_)
    auto queue = OclQueueManager::getInstance().getQueueRaw();

    switch (cimgRgb.spectrum()) {
    case 1: image = cimgGrayToGpu(cimgRgb, queue); break;
    case 3:
        rgbImage = cimgRgbToGpu(cimgRgb, queue);
        image = cimgRgbToGpuGray(cimgRgb, queue);
        isRGB = true;
        break;
    case 4: {
        alphaChannel.emplace(cimgRgb.get_shared_channel(3));
        auto rgbView = cimgRgb.get_shared_channels(0, 2);
        for (int y = 0; y < cimgRgb.height(); y++)
            for (int x = 0; x < cimgRgb.width(); x++)
                if ((*alphaChannel)(x, y) == 0)
                    for (int ch = 0; ch < 3; ch++)
                        rgbView(x, y, 0, ch) = 0.0f;
        rgbImage = cimgRgbToGpu(rgbView, queue);
        image = cimgRgbToGpuGray(rgbView, queue);
        isRGB = true;
        break;
    }
    default: throw std::runtime_error("Invalid image dimensions");
    }
    clFinish(queue);
#elif defined(_USE_EIGEN_)
    switch (cimgRgb.spectrum()) {
    case 1:
        rgbImage = eigen_utils::cimgToEigenGray(cimgRgb);
        image = rgbImage;
        break;
    case 3: {
        rgbImage = eigen_utils::cimgToEigenRgb(cimgRgb);
        const auto& rgb = rgbImage.getRGB();
        image = ((rgb[0] * kRW) + (rgb[1] * kGW) + (rgb[2] * kBW)).eval();
        break;
    }
    case 4: {
        alphaChannel.emplace(cimgRgb.get_shared_channel(3));
        auto rgbView = cimgRgb.get_shared_channels(0, 2);
        eigen_utils::cimgAlphaZero(rgbView, *alphaChannel);
        rgbImage = eigen_utils::cimgToEigenRgb(rgbView);
        const auto& rgb = rgbImage.getRGB();
        image = ((rgb[0] * kRW) + (rgb[1] * kGW) + (rgb[2] * kBW)).eval();
        break;
    }
    default: throw std::runtime_error("Invalid image dimensions");
    }
    isRGB = rgbImage.isRGB();
#endif
    return buf;
}

ImageBuffer InternalUtils::castToFloatGray(const ImageOutputBuffer& buffer, const bool isRGB) {
#if defined(_USE_CUDA_)
    const int planeSize = buffer.getRows() * buffer.getCols();
    const int channels = isRGB ? 3 : 1;
    auto stream = CudaStreamManager::getInstance().getComputeStream();
    CudaArray<float> gray(buffer.getRows(), buffer.getCols(), stream);
    cuda_utils::launchU8ToFloatGrayKernel(buffer.data(), gray.data(), planeSize, channels, stream);
    return gray;
#elif defined(_USE_OPENCL_)
    const int planeSize = buffer.getRows() * buffer.getCols();
    const int channels = isRGB ? 3 : 1;
    auto& mgr = OclQueueManager::getInstance();
    OclArray<float> gray(buffer.getRows(), buffer.getCols(), mgr.getQueueRaw());
    cl_utils::launchU8ToFloatGray(buffer.clBuffer(), gray.clBuffer(), planeSize, channels, mgr.getQueue());
    return gray;
#else
    if (isRGB) {
        const auto& rgbU8 = buffer.getRGB();
        return ImageBuffer((rgbU8[0].cast<float>() * kRW + rgbU8[1].cast<float>() * kGW + rgbU8[2].cast<float>() * kBW).eval());
    } else {
        return ImageBuffer(buffer.getGray().cast<float>());
    }
#endif
}
