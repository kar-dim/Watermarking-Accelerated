#include "buffer.hpp"
#include "CheckedSize.hpp"
#include "cimg_init.h"
#include "common_utils.hpp"
#include "ImageFileBuffer.hpp"
#include "TinyEXIF.h"
#include "utils.hpp"
#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <zlib.h>

#if defined(_USE_GPU_)
#include <cstring>
#endif

#if defined(_USE_OPENCL_)
#include "OclArray.hpp"
#include "OclQueueManager.hpp"
#include "opencl_utils.hpp"
#elif defined(_USE_CUDA_)
#include "cuda_utils.hpp"
#include "CudaArray.hpp"
#include "CudaCheck.hpp"
#include "CudaStreamManager.hpp"
#include "nvjpeg_utils.hpp"
#elif defined(_USE_EIGEN_)
#include "eigen_utils.hpp"
#endif

/*!
 *  \brief  Image I/O, format conversion, EXIF parsing, and buffer manipulation helper functions
 *  \author Dimitris Karatzas
 */

using std::string;
using namespace CommonUtils;

namespace {
string addSuffixBeforeExtension(const string& file, const string& suffix) {
    const auto dot = file.find_last_of('.');
    checkError(dot == string::npos || dot == file.size() - 1, "Filename has no valid extension: " + file);
    return file.substr(0, dot) + suffix + file.substr(dot);
}

string lowercaseExtension(const string& path) {
    string extension = path.substr(path.find_last_of('.') + 1);
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
    return extension;
}

bool hasJpegExtension(const string& path) {
    const string extension = lowercaseExtension(path);
    return extension == "jpg" || extension == "jpeg";
}

// 24-bit BMP file as CImg's save_bmp but more optimized and faster and without CImg overhead
void saveBmp(const Gray8BufferIO& image, const string& path) {
    constexpr size_t chunkBytes = size_t{1} << 20;
    const unsigned int width = image.width();
    const unsigned int height = image.height();
    const unsigned int align = (4 - (3 * width) % 4) % 4;
    const unsigned int rowBytes = (3 * width) + align;
    const unsigned int bufferSize = rowBytes * height;
    const unsigned int fileSize = 54 + bufferSize;
    const auto putLE32 = [](uint8_t* destination, const unsigned int value) {
        for (int byte = 0; byte < 4; byte++)
            destination[byte] = static_cast<uint8_t>(value >> (8 * byte));
    };
    uint8_t header[54] = {'B', 'M'};
    putLE32(header + 0x02, fileSize);
    header[0x0A] = 0x36;
    header[0x0E] = 0x28;
    putLE32(header + 0x12, width);
    putLE32(header + 0x16, height);
    header[0x1A] = 1;
    header[0x1C] = 24;
    putLE32(header + 0x22, bufferSize);
    header[0x27] = 0x1;
    header[0x2B] = 0x1;

    std::ofstream file(path, std::ofstream::binary);
    checkError(!file.write(reinterpret_cast<const char*>(header), sizeof(header)), "Unable to write the image file: " + path);
    const int channels = image.spectrum();
    // the row padding bytes are never written, they stay zero
    const size_t chunkRows = std::max<size_t>(chunkBytes / rowBytes, 1);
    std::vector<uint8_t> chunk(chunkRows * rowBytes, 0);
    size_t filledRows = 0;
    // bottom-up rows of BGR pixels
    for (int y = static_cast<int>(height) - 1; y >= 0; y--) {
        uint8_t* row = chunk.data() + (filledRows * rowBytes);
        const uint8_t* red = image.data(0, y, 0, 0);
        const uint8_t* green = channels >= 2 ? image.data(0, y, 0, 1) : red;
        const uint8_t* blue = channels >= 3 ? image.data(0, y, 0, 2) : red;
        for (unsigned int x = 0; x < width; x++) {
            row[(3 * x) + 0] = channels == 2 ? 0 : blue[x];
            row[(3 * x) + 1] = green[x];
            row[(3 * x) + 2] = red[x];
        }
        if (++filledRows == chunkRows || y == 0) {
            checkError(!file.write(reinterpret_cast<const char*>(chunk.data()), static_cast<std::streamsize>(filledRows * rowBytes)), "Unable to write the image file: " + path);
            filledRows = 0;
        }
    }
    file.close();
    checkError(!file, "Unable to write the image file: " + path);
}

// libpng output to the file stream
void writePngBytes(png_structp png, png_bytep data, const png_size_t length) {
    auto* file = static_cast<std::ofstream*>(png_get_io_ptr(png));
    if (!file->write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(length)))
        png_error(png, "write failed");
}
void flushPng(png_structp) {}

// writes the 8-bit PNG of the planes, row by row, libpng reports errors with longjmp: only trivially destructible objects live here
bool writePngRows(png_structp png, png_infop info, const uint8_t* const* planes, const int channels, const png_uint_32 width, const png_uint_32 height, uint8_t* row) {
    if (setjmp(png_jmpbuf(png)))
        return false;
    const int colorType = channels == 1 ? PNG_COLOR_TYPE_GRAY : channels == 2 ? PNG_COLOR_TYPE_GRAY_ALPHA : channels == 3 ? PNG_COLOR_TYPE_RGB : PNG_COLOR_TYPE_RGB_ALPHA;
    png_set_IHDR(png, info, width, height, 8, colorType, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    // the "up" filter with run-length deflate: about 5x faster than libpng's defaults (all filters, full deflate) for files ~15% larger
    png_set_filter(png, PNG_FILTER_TYPE_BASE, PNG_FILTER_UP);
    png_set_compression_strategy(png, Z_RLE);
    png_write_info(png, info);
    for (png_uint_32 y = 0; y < height; y++) {
        const size_t rowOffset = static_cast<size_t>(y) * width;
        for (int channel = 0; channel < channels; channel++) {
            const uint8_t* plane = planes[channel] + rowOffset;
            for (png_uint_32 x = 0; x < width; x++)
                row[(static_cast<size_t>(x) * channels) + channel] = plane[x];
        }
        png_write_row(png, row);
    }
    png_write_end(png, info);
    return true;
}

// the same PNG as CImg's save_png (8-bit gray, gray + alpha, RGB or RGBA, not interlaced), with faster compression settings (CImg has no option for them)
void savePng(const Gray8BufferIO& image, const string& path) {
    const int channels = std::min(image.spectrum(), 4);
    std::ofstream file(path, std::ofstream::binary);
    checkError(!file, "Unable to write the image file: " + path);
    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    png_infop info = png ? png_create_info_struct(png) : nullptr;
    if (!info) {
        png_destroy_write_struct(&png, nullptr);
        throw std::runtime_error("Unable to initialize the PNG encoder for: " + path);
    }
    const uint8_t* planes[4] = {};
    for (int channel = 0; channel < channels; channel++)
        planes[channel] = image.data(0, 0, 0, channel);
    std::vector<uint8_t> row(static_cast<size_t>(image.width()) * channels);
    png_set_write_fn(png, &file, writePngBytes, flushPng);
    const bool written = writePngRows(png, info, planes, channels, image.width(), image.height(), row.data());
    png_destroy_write_struct(&png, &info);
    checkError(!written, "Unable to write the image file: " + path);
    file.close();
    checkError(!file, "Unable to write the image file: " + path);
}

// save a CImg image selecting the correct encoder by file extension
void saveCimgByExtension(const Gray8BufferIO& cimgToSave, const string& path) {
    const string extension = lowercaseExtension(path);
    if (extension == "png")
        savePng(cimgToSave, path);
    else if (extension == "bmp")
        saveBmp(cimgToSave, path);
    else if (hasJpegExtension(path))
        cimgToSave.save_jpeg(path.c_str());
    else if (extension == "webp")
        cimgToSave.save_webp(path.c_str());
    else if (extension == "tif" || extension == "tiff")
        cimgToSave.save_tiff(path.c_str(), 20);
    else
        throw std::runtime_error("Unsupported image format: " + extension);
}

// anything outside the EXIF orientations 1-8 is not rotated
int exifOrientation(const TinyEXIF::EXIFInfo& exif) { return exif.Orientation >= 1 && exif.Orientation <= 8 ? exif.Orientation : 1; }

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
    // JPEG has no alpha: 4 components are CMYK, which CImg returns unconverted (K would be taken as the alpha channel)
    checkError(fileType == "jpg" && image.spectrum() == 4, "Unsupported image: CMYK JPEG, only grayscale and RGB JPEG images are supported");
    InternalUtils::checkImageDimensions(image.height(), image.width());
    InternalUtils::checkedElements(image.height(), image.width(), image.spectrum());
    return image;
}

// zero out RGB channels where alpha is 0, branchless, exploiting CImg's planar layout (RRRR...GGGG...BBBB...)
void cimgAlphaZero(Gray8BufferIO& rgb, const uint8_t* A) {
    const int planeSize = rgb.width() * rgb.height();
    uint8_t* R = rgb.data();
    uint8_t* G = R + planeSize;
    uint8_t* B = G + planeSize;
    // 32 pixels per step (clang does not vectorize the scalar loop!)
    constexpr int lanes = 32;
    const int blocks = planeSize / lanes;
    // GPU builds load on the batch prefetch thread: no OpenMP region there (MSVC's OpenMP threads spin after each region, slowing the decoding and the host threads next to it
#if !defined(_USE_GPU_)
#pragma omp parallel for schedule(static)
#endif
    for (int block = 0; block < blocks; block++) {
        const size_t offset = static_cast<size_t>(block) * lanes;
        const __m256i transparent = _mm256_cmpeq_epi8(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(A + offset)), _mm256_setzero_si256());
        for (uint8_t* plane : {R, G, B}) {
            __m256i* pixels = reinterpret_cast<__m256i*>(plane + offset);
            _mm256_storeu_si256(pixels, _mm256_andnot_si256(transparent, _mm256_loadu_si256(pixels)));
        }
    }
    for (int i = blocks * lanes; i < planeSize; i++) {
        const uint8_t mask = A[i] ? 0xFF : 0;
        R[i] &= mask;
        G[i] &= mask;
        B[i] &= mask;
    }
}

// the EXIF orientation on the host (CImg), the same transforms as the upload kernels
void rotate(Gray8BufferIO& img, const int orientation) {
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
// orients (EXIF orientation) a row-major planar 8-bit image on the device (1 or 3 channels, the CImg layout): column-major 8-bit RGB planes (empty for
// grayscale) and float luma. With "preview", the displayed row-major interleaved pixels (the original image preview) are downloaded into it
std::pair<ImageOutputBuffer, ImageBuffer> orientOnDevice(const ImageOutputBuffer& source, const int orientation, std::vector<uint8_t>* preview) {
    const int channels = source.getChannels();
    const int srcRows = source.getRows();
    const int srcCols = source.getCols();
    const bool swapAxes = orientation >= 5;
    const int rows = swapAxes ? srcCols : srcRows;
    const int cols = swapAxes ? srcRows : srcCols;
#if defined(_USE_CUDA_)
    const cudaStream_t stream = CudaStreamManager::getInstance().getComputeStream();
    CudaArray<uint8_t> rgb = channels == 3 ? CudaArray<uint8_t>(rows, cols, 3, stream) : CudaArray<uint8_t>();
    CudaArray<float> gray(rows, cols, stream);
    CudaArray<uint8_t> display = preview ? CudaArray<uint8_t>(rows, cols, channels, stream) : CudaArray<uint8_t>();
    cuda_utils::launchOrientRowMajorToColMajorKernel(source.data(), rgb.data(), gray.data(), display.data(), srcCols, srcRows, channels, orientation, stream);
#elif defined(_USE_OPENCL_)
    auto& mgr = OclQueueManager::getInstance();
    const cl_command_queue queue = mgr.getQueueRaw();
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

// uploads the CImg image (1 or 3 channels) and orients it on the device
std::pair<ImageOutputBuffer, ImageBuffer> uploadOriented(const Gray8BufferIO& img, const int orientation, std::vector<uint8_t>* preview) {
#if defined(_USE_CUDA_)
    const ImageOutputBuffer source(img.height(), img.width(), img.spectrum(), img.data(), CudaStreamManager::getInstance().getComputeStream());
#elif defined(_USE_OPENCL_)
    const ImageOutputBuffer source(img.height(), img.width(), img.spectrum(), img.data(), OclQueueManager::getInstance().getQueueRaw());
#endif
    return orientOnDevice(source, orientation, preview);
}
} // namespace
#endif

#if defined(_USE_CUDA_)
namespace {
// the whole file when it starts with the JPEG SOI marker (FF D8), empty otherwise
std::vector<uint8_t> readJpegFile(const string& imageFile) {
    std::ifstream fileStream(imageFile, std::ifstream::binary | std::ifstream::ate);
    const std::streamoff size = fileStream ? static_cast<std::streamoff>(fileStream.tellg()) : 0;
    std::vector<uint8_t> file(size >= 2 ? 2 : 0);
    if (file.empty() || !fileStream.seekg(0).read(reinterpret_cast<char*>(file.data()), 2) || file[0] != 0xFF || file[1] != 0xD8)
        return {};
    file.resize(static_cast<size_t>(size));
    fileStream.read(reinterpret_cast<char*>(file.data() + 2), size - 2);
    return fileStream ? file : std::vector<uint8_t>();
}
} // namespace
#endif

void InternalUtils::saveImage(const string& imagePath, const string& suffix, const ImageOutputBuffer& watermark, const std::vector<uint8_t>& alphaChannel) {
    const string watermarkedFile = addSuffixBeforeExtension(imagePath, suffix);
#if defined(_USE_GPU_)
    const int rows = watermark.getRows();
    const int cols = watermark.getCols();
    const int channels = watermark.getChannels();
    const bool hasAlpha = !alphaChannel.empty();
#if defined(_USE_CUDA_)
    // async saves can run on another host thread, select the buffer device first
    CUDA_CHECK(cudaSetDevice(watermark.getDeviceIndex()));
    auto stream = CudaStreamManager::getInstance().getComputeStream();
    CudaArray<uint8_t> rowMajor(rows, cols, channels, stream);
    cuda_utils::launchColMajorToRowMajorU8Kernel(watermark.data(), rowMajor.data(), cols, rows, channels, stream);
    // JPEG (no alpha) is encoded on the device, only the bitstream is downloaded, CImg saves it when nvJPEG fails
    if (!hasAlpha && hasJpegExtension(watermarkedFile)) {
        if (const std::vector<uint8_t> jpeg = nvjpeg_utils::encode(rowMajor, stream); !jpeg.empty()) {
            std::ofstream file(watermarkedFile, std::ofstream::binary);
            checkError(!file.write(reinterpret_cast<const char*>(jpeg.data()), static_cast<std::streamsize>(jpeg.size())), "Unable to write the image file: " + watermarkedFile);
            return;
        }
    }
#elif defined(_USE_OPENCL_)
    auto& mgr = OclQueueManager::getInstance();
    OclArray<uint8_t> rowMajor(rows, cols, channels, mgr.getQueueRaw());
    cl_utils::launchColMajorToRowMajorU8(watermark.clBuffer(), rowMajor.clBuffer(), cols, rows, channels, mgr.getQueue());
#endif
    Gray8BufferIO output(cols, rows, 1, hasAlpha ? 4 : channels);
    rowMajor.toHost(output.data());
    if (hasAlpha)
        std::memcpy(output.data() + (3 * cols * rows), alphaChannel.data(), cols * rows);
    saveCimgByExtension(output, watermarkedFile);
#elif defined(_USE_EIGEN_)
    const auto cimgToSave = watermark.isRGB() ? eigen_utils::eigenRgbToCimg(watermark.getRGB(), alphaChannel) : eigen_utils::eigenGrayToCimg(watermark.getGray());
    saveCimgByExtension(cimgToSave, watermarkedFile);
#endif
}

ImageFileBuffer InternalUtils::loadImage(const string& imageFile, const bool captureOriginal) {
    ImageFileBuffer buf;
    auto& rgbImage = buf.rgbImage;
    auto& image = buf.image;
    auto& alphaChannel = buf.alphaChannel;
    auto& rows = buf.rows;
    auto& cols = buf.cols;
    auto& isRGB = buf.isRGB;
#if defined(_USE_CUDA_)
    // a JPEG file is read once, nvJPEG decodes it on the device and TinyEXIF parses the orientation from it
    // if nvJPEG cannot decode, we take the CImg path below
    const std::vector<uint8_t> jpeg = readJpegFile(imageFile);
    if (!jpeg.empty()) {
        const cudaStream_t stream = CudaStreamManager::getInstance().getComputeStream();
        if (auto decoded = nvjpeg_utils::decode(jpeg.data(), jpeg.size(), stream)) {
            const int orientation = exifOrientation(TinyEXIF::EXIFInfo(jpeg.data(), static_cast<unsigned>(jpeg.size())));
            const bool swapAxes = orientation >= 5;
            rows = swapAxes ? decoded->getCols() : decoded->getRows();
            cols = swapAxes ? decoded->getRows() : decoded->getCols();
            if (captureOriginal)
                buf.previewChannels = decoded->getChannels();
            auto [rgb, gray] = orientOnDevice(*decoded, orientation, captureOriginal ? &buf.originalPreview : nullptr);
            isRGB = !rgb.empty();
            rgbImage = std::move(rgb);
            image = std::move(gray);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            return buf;
        }
    }
#endif
    std::ifstream fileStream(imageFile, std::ifstream::binary);
    TinyEXIF::EXIFInfo exif(fileStream); // parse EXIF for orientation
    auto cimgRgb = loadImage8(imageFile);
    const int channels = cimgRgb.spectrum();
    const int orientation = exifOrientation(exif);
#if defined(_USE_GPU_)
    // the upload kernel orients 1 and 3 channel images and writes their original preview
    const bool hostImage = channels == 4;
    const int deviceOrientation = hostImage ? 1 : orientation;
    if (hostImage)
        rotate(cimgRgb, orientation);
    const bool swapAxes = deviceOrientation >= 5;
    rows = swapAxes ? cimgRgb.width() : cimgRgb.height();
    cols = swapAxes ? cimgRgb.height() : cimgRgb.width();
#else
    constexpr bool hostImage = true;
    rotate(cimgRgb, orientation);
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
        const uint8_t* alphaPlane = cimgRgb.data(0, 0, 0, 3);
        alphaChannel.assign(alphaPlane, alphaPlane + static_cast<size_t>(cimgRgb.width()) * cimgRgb.height());
        auto rgbView = cimgRgb.get_shared_channels(0, 2);
        cimgAlphaZero(rgbView, alphaChannel.data());
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
        const uint8_t* alphaPlane = cimgRgb.data(0, 0, 0, 3);
        alphaChannel.assign(alphaPlane, alphaPlane + static_cast<size_t>(cimgRgb.width()) * cimgRgb.height());
        auto rgbView = cimgRgb.get_shared_channels(0, 2);
        cimgAlphaZero(rgbView, alphaChannel.data());
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

int InternalUtils::batchLoadThreads() { return std::clamp(static_cast<int>(std::thread::hardware_concurrency()) / 4, 2, 8); }

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
