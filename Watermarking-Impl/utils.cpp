#include "buffer.hpp"
#include "ImageFileBuffer.hpp"
#include "TinyEXIF.h"
#include "utils.hpp"
#include "WatermarkBase.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#if defined(_USE_OPENCL_)
#include "OclQueueManager.hpp"
#include "OclArray.hpp"
#include "opencl_utils.hpp"
#include "WatermarkOCL.hpp"
#include <cctype>
#elif defined(_USE_CUDA_)
#include "CudaStreamManager.hpp"
#include "CudaArray.hpp"
#include "CudaCheck.hpp"
#include "WatermarkCuda.cuh"
#include "cuda_utils.hpp"
#include <cctype>
#elif defined(_USE_EIGEN_)
#include <cctype>
#include "cimg_init.h"
#include "eigen_utils.hpp"
#include "WatermarkEigen.hpp"
#endif

using std::string;
using namespace CommonUtils;

namespace {
string addSuffixBeforeExtension(const string& file, const string& suffix) {
    const auto dot = file.find_last_of('.');
    checkError(dot == string::npos || dot == file.size() - 1, "Filename has no valid extension: " + file);
    return file.substr(0, dot) + suffix + file.substr(dot);
}

// save a CImg image selecting the correct encoder by file extension
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

// decode the image file to 8 bits per sample
Gray8BufferIO loadImage8(const string& imageFile) {
    // null for types it does not detect (the loader falls back to the extension)
    const char* detectedType = cimg_library::cimg::ftype(nullptr, imageFile.c_str());
    const string fileType = detectedType ? detectedType : "";
    unsigned int bitsPerValue = 8;
    Gray8BufferIO image;
    if (fileType == "png")
        image.load_png(imageFile.c_str(), &bitsPerValue);
    else if (fileType == "tif")
        image.load_tiff(imageFile.c_str(), 0, 0, 1, &bitsPerValue);
    else
        image.load(imageFile.c_str());
    checkError(bitsPerValue > 8, "Unsupported image: " + std::to_string(bitsPerValue) + " bits per sample, only 8-bit images are supported");
    return image;
}

// zero out RGB channels where alpha is 0, branchless, exploiting CImg's planar layout (RRRR...GGGG...BBBB...)
void cimgAlphaZero(Gray8BufferIO& rgb, const Gray8BufferIO& alpha) {
    const int planeSize = rgb.width() * rgb.height();
    uint8_t* R = rgb.data();
    uint8_t* G = R + planeSize;
    uint8_t* B = G + planeSize;
    const uint8_t* A = alpha.data();
#pragma omp parallel for schedule(static)
    for (int i = 0; i < planeSize; i++) {
        const uint8_t mask = A[i] ? 0xFF : 0;
        R[i] &= mask;
        G[i] &= mask;
        B[i] &= mask;
    }
}

// 8 samples of a plane (the low 8 bytes of the result)
inline __m128i loadBytes(const uint8_t* source) { return _mm_loadl_epi64(reinterpret_cast<const __m128i*>(source)); }

// this interleaves CImg's planar pixels for the preview without an intermediate image
void makeOriginalPreview(const uint8_t* source, uint8_t* preview, const size_t pixelCount, const int channels) {
    const size_t blocks = pixelCount / 8;
    const uint8_t* green = channels >= 3 ? source + pixelCount : source;
    const uint8_t* blue = channels >= 3 ? green + pixelCount : source;
    const uint8_t* alpha = channels == 4 ? blue + pixelCount : source;
    const __m128i zero = _mm_setzero_si128();
    const __m128i rgbMask = _mm_setr_epi8(0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, -1, -1, -1, -1);
    // note: 1000000 is an heuristic threshold, seems good enough
#pragma omp parallel for if (pixelCount > 1000000) schedule(static)
    for (std::ptrdiff_t block = 0; block < static_cast<std::ptrdiff_t>(blocks); ++block) {
        const size_t offset = static_cast<size_t>(block) * 8;
        const __m128i redBytes = loadBytes(source + offset);
        if (channels == 1) {
            _mm_storel_epi64(reinterpret_cast<__m128i*>(preview + offset), redBytes);
        } else {
            const __m128i greenBytes = loadBytes(green + offset);
            const __m128i blueBytes = loadBytes(blue + offset);
            const __m128i rg = _mm_unpacklo_epi8(redBytes, greenBytes);
            const __m128i ba = _mm_unpacklo_epi8(blueBytes, channels == 4 ? loadBytes(alpha + offset) : zero);
            const __m128i first = _mm_unpacklo_epi16(rg, ba);
            const __m128i second = _mm_unpackhi_epi16(rg, ba);
            if (channels == 4) {
                _mm_storeu_si128(reinterpret_cast<__m128i*>(preview + offset * 4), first);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(preview + offset * 4 + 16), second);
            } else {
                const __m128i packedFirst = _mm_shuffle_epi8(first, rgbMask);
                const __m128i packedSecond = _mm_shuffle_epi8(second, rgbMask);
                const __m128i front = _mm_or_si128(packedFirst, _mm_slli_si128(packedSecond, 12));
                _mm_storeu_si128(reinterpret_cast<__m128i*>(preview + offset * 3), front);
                _mm_storel_epi64(reinterpret_cast<__m128i*>(preview + offset * 3 + 16), _mm_srli_si128(packedSecond, 4));
            }
        }
    }
    // tail pixels (not multiple by AVX2 size (8))
    for (size_t pixel = blocks * 8; pixel < pixelCount; ++pixel)
        for (int channel = 0; channel < channels; ++channel)
            preview[pixel * channels + channel] = source[static_cast<size_t>(channel) * pixelCount + pixel];
}
} // namespace

// GPU helpers for CImg (row-major) -> GPU array (column-major) conversion
#if defined(_USE_GPU_)
namespace {
// uploads the 8-bit image (1 or 3 channels) and orients it (EXIF orientation) on the device: column-major 8-bit RGB planes (empty for grayscale) and float luma.
// With "preview", the displayed row-major interleaved pixels (the original image preview) are downloaded into it
std::pair<ImageOutputBuffer, ImageBuffer> uploadOriented(const Gray8BufferIO& img, const int orientation, std::vector<uint8_t>* preview) {
    const int channels = img.spectrum();
    const int srcRows = img.height();
    const int srcCols = img.width();
    const bool swapAxes = orientation >= 5;
    const int rows = swapAxes ? srcCols : srcRows;
    const int cols = swapAxes ? srcRows : srcCols;
#if defined(_USE_CUDA_)
    const cudaStream_t stream = CudaStreamManager::getInstance().getComputeStream();
    const CudaArray<uint8_t> source(srcRows, srcCols, channels, img.data(), stream);
    CudaArray<uint8_t> rgb = channels == 3 ? CudaArray<uint8_t>(rows, cols, 3, stream) : CudaArray<uint8_t>();
    CudaArray<float> gray(rows, cols, stream);
    CudaArray<uint8_t> display = preview ? CudaArray<uint8_t>(rows, cols, channels, stream) : CudaArray<uint8_t>();
    cuda_utils::launchOrientRowMajorToColMajorKernel(source.data(), rgb.data(), gray.data(), display.data(), srcCols, srcRows, channels, orientation, stream);
#elif defined(_USE_OPENCL_)
    auto& mgr = OclQueueManager::getInstance();
    const cl_command_queue queue = mgr.getQueueRaw();
    const OclArray<uint8_t> source(srcRows, srcCols, channels, img.data(), queue);
    OclArray<uint8_t> rgb = channels == 3 ? OclArray<uint8_t>(rows, cols, 3, queue) : OclArray<uint8_t>();
    OclArray<float> gray(rows, cols, queue);
    OclArray<uint8_t> display = preview ? OclArray<uint8_t>(rows, cols, channels, queue) : OclArray<uint8_t>();
    cl_utils::launchOrientRowMajorToColMajor(source.clBuffer(), rgb.clBuffer(), gray.clBuffer(), display.clBuffer(), srcCols, srcRows, channels, orientation, mgr.getQueue());
#endif
    if (preview) {
        preview->resize(display.bytes());
        display.toHost(preview->data());
    }
    return {std::move(rgb), std::move(gray)};
}
} // namespace
#endif

void InternalUtils::saveImage(const string& imagePath, const string& suffix, const ImageOutputBuffer& watermark, const std::optional<Gray8BufferIO>& alphaChannel) {
    const string watermarkedFile = addSuffixBeforeExtension(imagePath, suffix);
#if defined(_USE_GPU_)
    const int rows = watermark.getRows();
    const int cols = watermark.getCols();
    const int channels = watermark.getChannels();
    const bool hasAlpha = alphaChannel.has_value();
#if defined(_USE_CUDA_)
    // async saves can run on another host thread, select the buffer device first
    CUDA_CHECK(cudaSetDevice(watermark.getDeviceIndex()));
    auto stream = CudaStreamManager::getInstance().getComputeStream();
    CudaArray<uint8_t> rowMajor(rows, cols, channels, stream);
    cuda_utils::launchColMajorToRowMajorU8Kernel(watermark.data(), rowMajor.data(), cols, rows, channels, stream);
#elif defined(_USE_OPENCL_)
    auto& mgr = OclQueueManager::getInstance();
    OclArray<uint8_t> rowMajor(rows, cols, channels, mgr.getQueueRaw());
    cl_utils::launchColMajorToRowMajorU8(watermark.clBuffer(), rowMajor.clBuffer(), cols, rows, channels, mgr.getQueue());
#endif
    Gray8BufferIO output(cols, rows, 1, hasAlpha ? 4 : channels);
    rowMajor.toHost(output.data());
    if (hasAlpha)
        std::memcpy(output.data() + (3 * cols * rows), alphaChannel->data(), cols * rows);
    saveCimgByExtension(output, watermarkedFile);
#elif defined(_USE_EIGEN_)
    const auto cimgToSave = watermark.isRGB() ? eigen_utils::eigenRgbToCimg(watermark.getRGB(), alphaChannel) : eigen_utils::eigenGrayToCimg(watermark.getGray());
    saveCimgByExtension(cimgToSave, watermarkedFile);
#endif
}

std::unique_ptr<WatermarkBase> InternalUtils::createWatermarkObject(
    const unsigned int height, const unsigned int width, const string& watermarkPassword, const int p, const float psnr, std::unique_ptr<WatermarkBase> previous) {
    if (p != 3 && p != 5 && p != 7 && p != 9)
        throw std::invalid_argument("Unsupported value for p. Allowed p values: 3, 5, 7, 9");
    if (height < static_cast<unsigned int>(p) || width < static_cast<unsigned int>(p))
        throw std::invalid_argument("Image dimensions must each be at least p pixels");
    const int rows = static_cast<int>(height);
    const int cols = static_cast<int>(width);
    WatermarkBuffer watermark = [&] {
        if (previous && previous->hasSize(rows, cols))
            return previous->releaseWatermark();
        previous.reset();
        return WatermarkBase::generateWatermark(watermarkPassword, rows, cols);
    }();
    previous.reset();
#if defined(_USE_OPENCL_)
    switch (p) {
    case 3: return std::make_unique<WatermarkOCL<3>>(rows, cols, std::move(watermark), psnr); break;
    case 5: return std::make_unique<WatermarkOCL<5>>(rows, cols, std::move(watermark), psnr); break;
    case 7: return std::make_unique<WatermarkOCL<7>>(rows, cols, std::move(watermark), psnr); break;
    case 9: return std::make_unique<WatermarkOCL<9>>(rows, cols, std::move(watermark), psnr); break;
#elif defined(_USE_CUDA_)
    switch (p) {
    case 3: return std::make_unique<WatermarkCuda<3>>(rows, cols, std::move(watermark), psnr); break;
    case 5: return std::make_unique<WatermarkCuda<5>>(rows, cols, std::move(watermark), psnr); break;
    case 7: return std::make_unique<WatermarkCuda<7>>(rows, cols, std::move(watermark), psnr); break;
    case 9: return std::make_unique<WatermarkCuda<9>>(rows, cols, std::move(watermark), psnr); break;
#elif defined(_USE_EIGEN_)
    switch (p) {
    case 3: return std::make_unique<WatermarkEigen<3>>(rows, cols, std::move(watermark), psnr); break;
    case 5: return std::make_unique<WatermarkEigen<5>>(rows, cols, std::move(watermark), psnr); break;
    case 7: return std::make_unique<WatermarkEigen<7>>(rows, cols, std::move(watermark), psnr); break;
    case 9: return std::make_unique<WatermarkEigen<9>>(rows, cols, std::move(watermark), psnr); break;
#endif
    default: throw std::invalid_argument("Unsupported value for p. Allowed p values: 3, 5, 7, 9");
    }
}

void InternalUtils::rotate(Gray8BufferIO& img, const int orientation) {
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

ImageFileBuffer InternalUtils::loadImage(const string& imageFile, const bool captureOriginal) {
    ImageFileBuffer buf;
    auto& rgbImage = buf.rgbImage;
    auto& image = buf.image;
    auto& alphaChannel = buf.alphaChannel;
    auto& rows = buf.rows;
    auto& cols = buf.cols;
    auto& isRGB = buf.isRGB;
    std::ifstream fileStream(imageFile, std::ifstream::binary);
    TinyEXIF::EXIFInfo exif(fileStream); // parse EXIF for orientation
    auto cimgRgb = loadImage8(imageFile);
    const int channels = cimgRgb.spectrum();
    // anything outside the EXIF orientations 1-8 is not rotated
    const int orientation = exif.Orientation >= 1 && exif.Orientation <= 8 ? exif.Orientation : 1;
#if defined(_USE_GPU_)
    // the upload kernel orients 1 and 3 channel images and writes their original preview
    const bool hostImage = channels == 4;
    const int deviceOrientation = hostImage ? 1 : orientation;
    if (hostImage)
        InternalUtils::rotate(cimgRgb, orientation);
    const bool swapAxes = deviceOrientation >= 5;
    rows = swapAxes ? cimgRgb.width() : cimgRgb.height();
    cols = swapAxes ? cimgRgb.height() : cimgRgb.width();
#else
    constexpr bool hostImage = true;
    InternalUtils::rotate(cimgRgb, orientation);
    rows = cimgRgb.height();
    cols = cimgRgb.width();
#endif

    if (captureOriginal && (channels == 1 || channels == 3 || channels == 4)) {
        buf.previewChannels = channels;
        // Convert the CPU pixels for the single image preview
        if (hostImage) {
            const size_t planeSize = static_cast<size_t>(rows) * cols;
            buf.originalPreview.resize(planeSize * channels);
            makeOriginalPreview(cimgRgb.data(), buf.originalPreview.data(), planeSize, channels);
        }
    }

#if defined(_USE_GPU_)
#if defined(_USE_CUDA_)
    auto stream = CudaStreamManager::getInstance().getComputeStream();
#elif defined(_USE_OPENCL_)
    auto stream = OclQueueManager::getInstance().getQueueRaw();
#endif
    std::vector<uint8_t>* devicePreview = captureOriginal && !hostImage ? &buf.originalPreview : nullptr;
    switch (channels) {
    case 1: image = uploadOriented(cimgRgb, deviceOrientation, devicePreview).second; break;
    case 3: {
        auto [rgb, gray] = uploadOriented(cimgRgb, deviceOrientation, devicePreview);
        rgbImage = std::move(rgb);
        image = std::move(gray);
        isRGB = true;
        break;
    }
    case 4: {
        // owning copy here on purpose (avoid dangling shared CImg pointer)
        alphaChannel.emplace(cimgRgb.get_channel(3));
        auto rgbView = cimgRgb.get_shared_channels(0, 2);
        cimgAlphaZero(rgbView, *alphaChannel);
        auto [rgb, gray] = uploadOriented(rgbView, deviceOrientation, nullptr);
        rgbImage = std::move(rgb);
        image = std::move(gray);
        isRGB = true;
        break;
    }
    default: throw std::runtime_error("Invalid image dimensions");
    }
#if defined(_USE_CUDA_)
    CUDA_CHECK(cudaStreamSynchronize(stream));
#elif defined(_USE_OPENCL_)
    clFinish(stream);
#endif
#elif defined(_USE_EIGEN_)
    switch (channels) {
    case 1: image = eigen_utils::cimgToEigenGray(cimgRgb); break;
    case 3: {
        auto [rgb, gray] = eigen_utils::cimgToEigenRgbAndGray(cimgRgb);
        rgbImage = std::move(rgb);
        image = std::move(gray);
        break;
    }
    case 4: {
        // owning copy here on purpose (avoid dangling shared CImg pointer)
        alphaChannel.emplace(cimgRgb.get_channel(3));
        auto rgbView = cimgRgb.get_shared_channels(0, 2);
        cimgAlphaZero(rgbView, *alphaChannel);
        auto [rgb, gray] = eigen_utils::cimgToEigenRgbAndGray(rgbView);
        rgbImage = std::move(rgb);
        image = std::move(gray);
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
        return ImageBuffer((rgbU8[0].cast<float>() * kLumaR + rgbU8[1].cast<float>() * kLumaG + rgbU8[2].cast<float>() * kLumaB).eval());
    } else {
        return ImageBuffer(buffer.getGray().cast<float>());
    }
#endif
}
