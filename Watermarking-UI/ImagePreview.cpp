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

// Let the core write its session pixels straight into Qt's display buffer (converted on the GPU for the GPU backends)
QImage imagePreviewFromSession(const WatermarkCore::ImageSession* session) {
    QImage preview;
    updatePreviewFromSession(session, preview);
    return preview;
}

// reused buffer skips the allocation and the first touch page faults of a new one
void updatePreviewFromSession(const WatermarkCore::ImageSession* session, QImage& preview) {
    const auto [width, height, channels] = WatermarkCore::getSessionPreviewFormat(session);
    const QImage::Format format = channels == 4 ? QImage::Format_RGBA8888 : channels == 3 ? QImage::Format_RGB888 : QImage::Format_Grayscale8;
    // a shared buffer is still displayed elsewhere, writing to it would detach it with a copy of the old pixels first
    if (!preview.isDetached() || preview.width() != width || preview.height() != height || preview.format() != format)
        preview = QImage(width, height, format);
    if (preview.isNull())
        return;
    WatermarkCore::copySessionPreview(session, preview.bits(), static_cast<size_t>(preview.bytesPerLine()));
}
