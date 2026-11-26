#include "buffer.hpp"
#include "CheckedSize.hpp"
#include "cimg_init.h"
#include "common_utils.hpp"
#include "ImageFileBuffer.hpp"
#include "ImagePixels.hpp"
#include "include/Utf8Path.hpp"
#include "TinyEXIF.h"
#include "utils.hpp"
#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <zlib.h>

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
    const std::u8string ext = WatermarkCore::pathFromUtf8(file).extension().u8string();
    checkError(ext.size() <= 1 || ext.size() > file.size() || file[file.size() - ext.size()] != '.', "Filename has no valid extension: " + file);
    const size_t dot = file.size() - ext.size();
    return file.substr(0, dot) + suffix + file.substr(dot);
}

string lowercaseExtension(const string& filePath) {
    const std::u8string ext = WatermarkCore::pathFromUtf8(filePath).extension().u8string();
    if (ext.size() <= 1)
        return {};
    string extension(reinterpret_cast<const char*>(ext.data() + 1), ext.size() - 1);
    std::transform(extension.begin(), extension.end(), extension.begin(), [](const unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return extension;
}

bool hasJpegExtension(const string& path) {
    const string extension = lowercaseExtension(path);
    return extension == "jpg" || extension == "jpeg";
}

// 24-bit BMP export without CImg overhead
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
    // row padding bytes remain zero
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

// write 8-bit PNG, libpng errors longjmp, keep local objects trivially destructible
bool writePngRows(png_structp png, png_infop info, const uint8_t* const* planes, const int channels, const png_uint_32 width, const png_uint_32 height, uint8_t* row) {
    if (setjmp(png_jmpbuf(png)))
        return false;
    const int colorType = channels == 1 ? PNG_COLOR_TYPE_GRAY : channels == 2 ? PNG_COLOR_TYPE_GRAY_ALPHA : channels == 3 ? PNG_COLOR_TYPE_RGB : PNG_COLOR_TYPE_RGB_ALPHA;
    png_set_IHDR(png, info, width, height, 8, colorType, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    // up filter with run-length deflate for faster encoding
    png_set_filter(png, PNG_FILTER_TYPE_BASE, PNG_FILTER_UP);
    png_set_compression_strategy(png, Z_RLE);
    png_write_info(png, info);
    for (png_uint_32 y = 0; y < height; y++) {
        const size_t rowOffset = static_cast<size_t>(y) * width;
        if (channels == 1) {
            png_write_row(png, const_cast<png_bytep>(planes[0] + rowOffset));
            continue;
        }
        if (channels == 3 || channels == 4) {
            ImagePixels::interleaveRow(planes[0] + rowOffset, planes[1] + rowOffset, planes[2] + rowOffset, channels == 4 ? planes[3] + rowOffset : nullptr, row, width, channels);
            png_write_row(png, row);
            continue;
        }
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

// PNG export with faster compression settings
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

// save image using encoder for file extension
void saveCimgByExtension(const Gray8BufferIO& cimgToSave, const string& path) {
    const string extension = lowercaseExtension(path);
    if (extension == "png") {
        savePng(cimgToSave, path);
    } else if (extension == "bmp") {
        checkError(cimgToSave.spectrum() == 4, "BMP format does not support an alpha channel, save as PNG, TIFF or WebP instead: " + path);
        saveBmp(cimgToSave, path);
    } else if (hasJpegExtension(path)) {
        checkError(cimgToSave.spectrum() == 4, "JPEG format does not support an alpha channel, save as PNG, TIFF or WebP instead: " + path);
        cimgToSave.save_jpeg(path.c_str());
    } else if (extension == "webp") {
        cimgToSave.save_webp(path.c_str());
    } else if (extension == "tif" || extension == "tiff") {
        cimgToSave.save_tiff(path.c_str(), 20);
    } else {
        throw std::runtime_error("Unsupported image format: " + extension);
    }
}

// unhandled orientations skip rotation
int exifOrientation(const TinyEXIF::EXIFInfo& exif) { return exif.Orientation >= 1 && exif.Orientation <= 8 ? exif.Orientation : 1; }

// decode image file to 8 bits per sample
Gray8BufferIO loadImage8(const string& imageFile) {
    if (imageFile == "-")
        throw std::invalid_argument("Image input requires a file path");
    char header[12]{};
    std::ifstream input(imageFile, std::ios::binary);
    if (!input)
        throw std::runtime_error("Cannot open image: " + imageFile);
    input.read(header, sizeof(header));
    const auto matches = [&](const char* signature, size_t length) { return input.gcount() >= static_cast<std::streamsize>(length) && !std::memcmp(header, signature, length); };
    const bool jpeg = matches("\xff\xd8\xff", 3);
    unsigned int bitsPerValue = 8;
    Gray8BufferIO image;
    if (matches("\x89PNG\r\n\x1a\n", 8))
        image.load_png(imageFile.c_str(), &bitsPerValue);
    else if (matches("II*\0", 4) || matches("MM\0*", 4) || matches("II+\0", 4) || matches("MM\0+", 4))
        image.load_tiff(imageFile.c_str(), 0, 0, 1, &bitsPerValue);
    else if (header[0] == 'P' && header[1] >= '1' && header[1] <= '6')
        image.load_pnm(imageFile.c_str());
    else if (jpeg)
        image.load_jpeg(imageFile.c_str());
    else if (matches("BM", 2))
        image.load_bmp(imageFile.c_str());
    else if (input.gcount() == 12 && matches("RIFF", 4) && !std::memcmp(header + 8, "WEBP", 4))
        image.load_webp(imageFile.c_str());
    else
        throw std::runtime_error("Unsupported image format: " + imageFile);
    checkError(image.depth() != 1, "Unsupported image depth");
    checkError(bitsPerValue > 8, "Unsupported image: " + std::to_string(bitsPerValue) + " bits per sample, only 8-bit images are supported");
    // convert CMYK JPEG to RGB, JPEG has no alpha channel
    checkError(jpeg && image.spectrum() == 4, "Unsupported image: CMYK JPEG, only grayscale and RGB JPEG images are supported");
    InternalUtils::checkImageDimensions(image.height(), image.width());
    InternalUtils::checkedElements(image.height(), image.width(), image.spectrum());
    return image;
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

} // namespace

// GPU helpers for CImg (row-major) -> GPU array (column-major) conversion
#if defined(_USE_GPU_)
namespace {
// orients planar 8-bit image on device to column-major RGB and float luma
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

// upload image and orient on device
std::pair<ImageOutputBuffer, ImageBuffer> uploadOriented(const Gray8BufferIO& img, const int orientation, std::vector<uint8_t>* preview) {
#if defined(_USE_CUDA_)
    const ImageOutputBuffer source(img.height(), img.width(), img.spectrum(), img.data(), CudaStreamManager::getInstance().getComputeStream());
    return orientOnDevice(source, orientation, preview);
#elif defined(_USE_OPENCL_)
    const cl_command_queue queue = OclQueueManager::getInstance().getQueueRaw();
    struct UnwindSync {
        cl_command_queue queue;
        bool armed = true;
        ~UnwindSync() {
            if (armed)
                clFinish(queue);
        }
    } sync{queue};
    const ImageOutputBuffer source = ImageOutputBuffer::fromHostAsync(img.height(), img.width(), img.spectrum(), img.data(), queue);
    auto result = orientOnDevice(source, orientation, preview);
    if (!preview)
        OclQueueManager::getInstance().getQueue().finish();
    sync.armed = false;
    return result;
#endif
}
} // namespace
#endif

#if defined(_USE_CUDA_)
namespace {
// file buffer when header matches JPEG SOI marker
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

void InternalUtils::saveImage(const string& imagePath, const string& suffix, const ImageOutputBuffer& watermark, const std::vector<uint8_t>& alphaChannel, [[maybe_unused]] const bool parallel) {
    const string watermarkedFile = addSuffixBeforeExtension(imagePath, suffix);
    const string extension = lowercaseExtension(watermarkedFile);
    if (!alphaChannel.empty() && (extension == "bmp" || extension == "jpg" || extension == "jpeg"))
        throw std::runtime_error((extension == "bmp" ? "BMP" : "JPEG") + string(" format does not support an alpha channel, save as PNG, TIFF or WebP instead: ") + watermarkedFile);
#if defined(_USE_GPU_)
    const int rows = watermark.getRows();
    const int cols = watermark.getCols();
    const int channels = watermark.getChannels();
    const bool hasAlpha = !alphaChannel.empty();
#if defined(_USE_CUDA_)
    // async save may run on another thread, activate device first
    const cuda_utils::ScopedDevice binding(watermark.getDeviceIndex());
    CUDA_CHECK(binding.status);
    auto stream = CudaStreamManager::getInstance().getComputeStream();
    CudaArray<uint8_t> rowMajor(rows, cols, channels, stream);
    cuda_utils::launchColMajorToRowMajorU8Kernel(watermark.data(), rowMajor.data(), cols, rows, channels, stream);
    // encode JPEG on device, download bitstream and fall back to CImg on failure
    if (!hasAlpha && hasJpegExtension(watermarkedFile)) {
        if (const std::vector<uint8_t> jpeg = nvjpeg_utils::encode(rowMajor, stream); !jpeg.empty()) {
            std::ofstream file(watermarkedFile, std::ofstream::binary);
            checkError(!file.write(reinterpret_cast<const char*>(jpeg.data()), static_cast<std::streamsize>(jpeg.size())), "Unable to write the image file: " + watermarkedFile);
            return;
        }
    }
#elif defined(_USE_OPENCL_)
    const OclQueueManager::Binding binding(watermark.getOwner());
    auto& mgr = OclQueueManager::getInstance();
    OclArray<uint8_t> rowMajor(rows, cols, channels, mgr.getQueueRaw());
    cl_utils::launchColMajorToRowMajorU8(watermark.clBuffer(), rowMajor.clBuffer(), cols, rows, channels, mgr.getQueue());
#endif
    Gray8BufferIO output(cols, rows, 1, hasAlpha ? 4 : channels);
    rowMajor.toHost(output.data());
    if (hasAlpha) {
        const size_t planeSize = static_cast<size_t>(cols) * rows;
        checkError(alphaChannel.size() != planeSize, "Invalid alpha plane size");
        std::memcpy(output.data() + (3 * planeSize), alphaChannel.data(), planeSize);
    }
    saveCimgByExtension(output, watermarkedFile);
#elif defined(_USE_EIGEN_)
    const auto cimgToSave = watermark.isRGB() ? eigen_utils::eigenRgbToCimg(watermark.getRGB(), alphaChannel, parallel) : eigen_utils::eigenGrayToCimg(watermark.getGray(), parallel);
    saveCimgByExtension(cimgToSave, watermarkedFile);
#endif
}

ImageFileBuffer InternalUtils::loadImage(const string& imageFile, const bool captureOriginal, [[maybe_unused]] const bool parallel) {
    ImageFileBuffer buf;
    auto& rgbImage = buf.rgbImage;
    auto& image = buf.image;
    auto& alphaChannel = buf.alphaChannel;
    auto& rows = buf.rows;
    auto& cols = buf.cols;
    auto& isRGB = buf.isRGB;
#if defined(_USE_CUDA_)
    // read JPEG once, nvJPEG decodes on device, fall back to CImg on error
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
            if (!captureOriginal)
                CUDA_CHECK(cudaStreamSynchronize(stream));
            return buf;
        }
    }
#endif
    std::ifstream fileStream(imageFile, std::ifstream::binary);
    TinyEXIF::EXIFInfo exif(fileStream);
    auto cimgRgb = loadImage8(imageFile);
    const int channels = cimgRgb.spectrum();
    const int orientation = exifOrientation(exif);
#if defined(_USE_GPU_)
    // upload kernel orients image and generates preview
    const bool hostImage = channels == 4;
    const int deviceOrientation = hostImage ? 1 : orientation;
    if (hostImage)
        rotate(cimgRgb, orientation);
    const bool swapAxes = deviceOrientation >= 5;
    rows = swapAxes ? cimgRgb.width() : cimgRgb.height();
    cols = swapAxes ? cimgRgb.height() : cimgRgb.width();
#else
    const bool hostImage = captureOriginal || channels == 4;
    const int eigenOrientation = hostImage ? 1 : orientation;
    if (hostImage)
        rotate(cimgRgb, orientation);
    const bool swapAxes = eigenOrientation >= 5 && eigenOrientation <= 8;
    rows = swapAxes ? cimgRgb.width() : cimgRgb.height();
    cols = swapAxes ? cimgRgb.height() : cimgRgb.width();
#endif

    if (captureOriginal && (channels == 1 || channels == 3 || channels == 4)) {
        buf.previewChannels = channels;
        // convert CPU pixels for single image preview
        if (hostImage) {
            const size_t planeSize = static_cast<size_t>(rows) * cols;
            buf.originalPreview.resize(planeSize * channels);
            ImagePixels::makePreview(cimgRgb.data(), buf.originalPreview.data(), planeSize, channels);
        }
    }

#if defined(_USE_GPU_)
#if defined(_USE_CUDA_)
    auto stream = CudaStreamManager::getInstance().getComputeStream();
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
        ImagePixels::clearTransparent(rgbView.data(), alphaChannel.data(), alphaChannel.size());
        auto [rgb, gray] = uploadOriented(rgbView, deviceOrientation, nullptr);
        rgbImage = std::move(rgb);
        image = std::move(gray);
        isRGB = true;
        break;
    }
    case 2: throw std::runtime_error("Unsupported 2-channel grayscale+alpha image format: " + imageFile);
    default: throw std::runtime_error("Invalid image dimensions");
    }
#if defined(_USE_CUDA_)
    if (!devicePreview)
        CUDA_CHECK(cudaStreamSynchronize(stream));
#endif
#elif defined(_USE_EIGEN_)
    switch (channels) {
    case 1: image = eigen_utils::cimgToEigenGray(cimgRgb, parallel, eigenOrientation); break;
    case 3: {
        auto [rgb, gray] = eigen_utils::cimgToEigenRgbAndGray(cimgRgb, parallel, eigenOrientation);
        rgbImage = std::move(rgb);
        image = std::move(gray);
        break;
    }
    case 4: {
        const uint8_t* alphaPlane = cimgRgb.data(0, 0, 0, 3);
        alphaChannel.assign(alphaPlane, alphaPlane + static_cast<size_t>(cimgRgb.width()) * cimgRgb.height());
        auto rgbView = cimgRgb.get_shared_channels(0, 2);
        if (parallel)
            ImagePixels::clearTransparent(rgbView.data(), alphaChannel.data(), alphaChannel.size());
        else
            ImagePixels::clearTransparentSerial(rgbView.data(), alphaChannel.data(), alphaChannel.size());
        auto [rgb, gray] = eigen_utils::cimgToEigenRgbAndGray(rgbView, parallel);
        rgbImage = std::move(rgb);
        image = std::move(gray);
        break;
    }
    case 2: throw std::runtime_error("Unsupported 2-channel grayscale+alpha image format: " + imageFile);
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
    return isRGB ? eigen_utils::u8RgbToFloatGray(buffer.getRGB()) : eigen_utils::u8GrayToFloat(buffer.getGray());
#endif
}
