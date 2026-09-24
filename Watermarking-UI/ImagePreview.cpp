#include "ImagePreview.hpp"
#include <cstring>
#include <WatermarkCore.hpp>

/*!
 *  \brief  Image conversion implementations between session buffers and QImage
 *  \author Dimitris Karatzas
 */

// Copy original pixel buffer into a QImage for side by side preview
QImage originalPreviewFromSession(WatermarkCore::ImageSession* session) {
    WatermarkCore::OriginalPixelData pixels = WatermarkCore::takeOriginalPixelData(session);
    if (pixels.pixels.empty())
        return {};
    const QImage::Format format = pixels.channels == 4 ? QImage::Format_RGBA8888 : pixels.channels == 3 ? QImage::Format_RGB888 : QImage::Format_Grayscale8;
    QImage preview(pixels.width, pixels.height, format);
    if (preview.isNull())
        return preview;
    const size_t rowBytes = static_cast<size_t>(pixels.width) * pixels.channels;
    for (int row = 0; row < pixels.height; ++row)
        std::memcpy(preview.scanLine(row), pixels.pixels.data() + static_cast<size_t>(row) * rowBytes, rowBytes);
    return preview;
}

// Let the core convert its planar session pixels into Qt's display buffer
QImage imagePreviewFromSession(const WatermarkCore::ImageSession* session) {
    const WatermarkCore::SessionPixelData pixelData = WatermarkCore::getSessionPixelData(session);
    QImage preview(pixelData.width, pixelData.height, pixelData.channels == 3 ? QImage::Format_RGB888 : QImage::Format_Grayscale8);
    if (preview.isNull())
        return preview;
    WatermarkCore::copySessionPixelsForPreview(pixelData, preview.bits(), static_cast<size_t>(preview.bytesPerLine()));
    return preview;
}
