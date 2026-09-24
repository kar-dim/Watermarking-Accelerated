#include "buffer.hpp"
#include "common_utils.hpp"
#include "luma_coefficients.hpp"
#include "ImageFileBuffer.hpp"
#include "TinyEXIF.h"
#include "utils.hpp"
#include "WatermarkBase.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <immintrin.h>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#if defined(_USE_OPENCL_)
#include "OclQueueManager.hpp"
#include "OclArray.hpp"
#include "opencl_utils.hpp"
#include "WatermarkOCL.hpp"
#include <cctype>
#elif defined(_USE_CUDA_)
#include "CudaStreamManager.hpp"
#include "CudaArray.hpp"
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

// zero out RGB channels where alpha is 0, branchless, exploiting CImg's planar layout (RRRR...GGGG...BBBB...)
void cimgAlphaZero(FloatBufferIO& rgb, const Gray8BufferIO& alpha) {
    const int planeSize = rgb.width() * rgb.height();
    float* R = rgb.data();
    float* G = R + planeSize;
    float* B = G + planeSize;
    const uint8_t* A = alpha.data();
#pragma omp parallel for schedule(static)
    for (int i = 0; i < planeSize; i++) {
        const float mask = A[i] ? 1.0f : 0.0f;
        R[i] *= mask;
        G[i] *= mask;
        B[i] *= mask;
    }
}

// clamp and truncate eight planar float samples to eight display bytes (used by the preview code)
inline __m128i packPreviewBytes(const float* source) {
    const __m256 values = _mm256_loadu_ps(source);
    const __m256 clamped = _mm256_max_ps(_mm256_setzero_ps(), _mm256_min_ps(values, _mm256_set1_ps(255.0f)));
    const __m256i integers = _mm256_cvttps_epi32(clamped);
    const __m128i halves = _mm_packs_epi32(_mm256_castsi256_si128(integers), _mm256_extracti128_si256(integers, 1));
    return _mm_packus_epi16(halves, _mm_setzero_si128());
}

// this interleaves CImg's planar pixels for the preview without an intermediate image
void makeOriginalPreview(const float* source, uint8_t* preview, const size_t pixelCount, const int channels) {
    const size_t blocks = pixelCount / 8;
    const float* green = channels >= 3 ? source + pixelCount : source;
    const float* blue = channels >= 3 ? green + pixelCount : source;
    const float* alpha = channels == 4 ? blue + pixelCount : source;
    const __m128i zero = _mm_setzero_si128();
    const __m128i rgbMask = _mm_setr_epi8(0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, -1, -1, -1, -1);
    // note: 1000000 is an heuristic threshold, seems good enough
#pragma omp parallel for if (pixelCount > 1000000) schedule(static)
    for (std::ptrdiff_t block = 0; block < static_cast<std::ptrdiff_t>(blocks); ++block) {
        const size_t offset = static_cast<size_t>(block) * 8;
        const __m128i redBytes = packPreviewBytes(source + offset);
        if (channels == 1) {
            _mm_storel_epi64(reinterpret_cast<__m128i*>(preview + offset), redBytes);
        } else {
            const __m128i greenBytes = packPreviewBytes(green + offset);
            const __m128i blueBytes = packPreviewBytes(blue + offset);
            const __m128i rg = _mm_unpacklo_epi8(redBytes, greenBytes);
            const __m128i ba = _mm_unpacklo_epi8(blueBytes, channels == 4 ? packPreviewBytes(alpha + offset) : zero);
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
            preview[pixel * channels + channel] = static_cast<uint8_t>(std::clamp(source[static_cast<size_t>(channel) * pixelCount + pixel], 0.0f, 255.0f));
}
} // namespace

// GPU helpers for CImg (row-major) <-> GPU array (column-major) conversion
#if defined(_USE_GPU_)
namespace {
#if defined(_USE_CUDA_)
// For CUDA: upload the raw row-major CImg data to a temp GPU buffer, then transpose on GPU
ImageBuffer cimgGrayToGpu(const FloatBufferIO& img, cudaStream_t stream) {
    const int rows = img.height();
    const int cols = img.width();
    CudaArray<float> rowMajor(rows, cols, img.data(), stream);
    CudaArray<float> colMajor(rows, cols, stream);
    cuda_utils::launchRowMajorToColMajorFloatKernel(rowMajor.data(), colMajor.data(), cols, rows, 1, stream);
    return colMajor;
}

std::pair<ImageBuffer, ImageBuffer> cimgRgbToGpuAndGray(const FloatBufferIO& img, cudaStream_t stream) {
    const int rows = img.height();
    const int cols = img.width();
    CudaArray<float> rowMajor(rows, cols, 3, img.data(), stream);
    CudaArray<float> rgb(rows, cols, 3, stream);
    CudaArray<float> gray(rows, cols, stream);
    cuda_utils::launchRowMajorToColMajorFloatKernel(rowMajor.data(), rgb.data(), cols, rows, 3, stream);
    cuda_utils::launchRowMajorRGBToColMajorGrayKernel(rowMajor.data(), gray.data(), cols, rows, stream);
    return {std::move(rgb), std::move(gray)};
}

#elif defined(_USE_OPENCL_)
// For OpenCL: upload the raw row-major CImg data to a temp GPU buffer, then transpose on GPU (mirrors CUDA)
ImageBuffer cimgGrayToGpu(const FloatBufferIO& img, cl_command_queue queue) {
    const int rows = img.height();
    const int cols = img.width();
    OclArray<float> rowMajor(rows, cols, img.data(), queue);
    OclArray<float> colMajor(rows, cols, queue);
    auto& q = OclQueueManager::getInstance().getQueue();
    cl_utils::launchRowMajorToColMajorFloat(rowMajor.clBuffer(), colMajor.clBuffer(), cols, rows, 1, q);
    return colMajor;
}

std::pair<ImageBuffer, ImageBuffer> cimgRgbToGpuAndGray(const FloatBufferIO& img, cl_command_queue queue) {
    const int rows = img.height();
    const int cols = img.width();
    OclArray<float> rowMajor(rows, cols, 3, img.data(), queue);
    OclArray<float> rgb(rows, cols, 3, queue);
    OclArray<float> gray(rows, cols, queue);
    auto& q = OclQueueManager::getInstance().getQueue();
    cl_utils::launchRowMajorToColMajorFloat(rowMajor.clBuffer(), rgb.clBuffer(), cols, rows, 3, q);
    cl_utils::launchRowMajorRGBToColMajorGray(rowMajor.clBuffer(), gray.clBuffer(), cols, rows, q);
    return {std::move(rgb), std::move(gray)};
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

std::unique_ptr<WatermarkBase> InternalUtils::createWatermarkObject(const unsigned int height, const unsigned int width, const string& watermarkPassword, const int p, const float psnr) {
    if (p != 3 && p != 5 && p != 7 && p != 9)
        throw std::invalid_argument("Unsupported value for p. Allowed p values: 3, 5, 7, 9");
    if (height < static_cast<unsigned int>(p) || width < static_cast<unsigned int>(p))
        throw std::invalid_argument("Image dimensions must each be at least p pixels");
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
    auto cimgRgb = FloatBufferIO(imageFile.c_str());
    InternalUtils::rotate(cimgRgb, exif.Orientation); // optional rotate (if required)
    rows = cimgRgb.height();
    cols = cimgRgb.width();

    if (captureOriginal && (cimgRgb.spectrum() == 1 || cimgRgb.spectrum() == 3 || cimgRgb.spectrum() == 4)) {
        // Convert the CPU pixels for the single image preview
        const int channels = cimgRgb.spectrum();
        const size_t planeSize = static_cast<size_t>(rows) * cols;
        const float* source = cimgRgb.data();
        buf.previewChannels = channels;
        buf.originalPreview.resize(planeSize * channels);
        makeOriginalPreview(source, buf.originalPreview.data(), planeSize, channels);
    }

#if defined(_USE_GPU_)
#if defined(_USE_CUDA_)
    auto stream = CudaStreamManager::getInstance().getComputeStream();
#elif defined(_USE_OPENCL_)
    auto stream = OclQueueManager::getInstance().getQueueRaw();
#endif
    switch (cimgRgb.spectrum()) {
    case 1: image = cimgGrayToGpu(cimgRgb, stream); break;
    case 3: {
        auto [rgb, gray] = cimgRgbToGpuAndGray(cimgRgb, stream);
        rgbImage = std::move(rgb);
        image = std::move(gray);
        isRGB = true;
        break;
    }
    case 4: {
        // CImg<float> -> CImg<uint8_t> creates an owning copy, so we are safe
        alphaChannel.emplace(cimgRgb.get_shared_channel(3));
        auto rgbView = cimgRgb.get_shared_channels(0, 2);
        cimgAlphaZero(rgbView, *alphaChannel);
        auto [rgb, gray] = cimgRgbToGpuAndGray(rgbView, stream);
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
    switch (cimgRgb.spectrum()) {
    case 1:
        rgbImage = eigen_utils::cimgToEigenGray(cimgRgb);
        image = rgbImage;
        break;
    case 3: {
        auto [rgb, gray] = eigen_utils::cimgToEigenRgbAndGray(cimgRgb);
        rgbImage = std::move(rgb);
        image = std::move(gray);
        break;
    }
    case 4: {
        // Different types, thus it creates an owning 8-bit alpha plane, we are safe
        alphaChannel.emplace(cimgRgb.get_shared_channel(3));
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
