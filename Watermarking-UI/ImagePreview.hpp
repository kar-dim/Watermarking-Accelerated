#pragma once

#include <QImage>

namespace WatermarkCore {
struct ImageSession;
}

/*!
 *  \brief  Image conversion routines for displaying core session buffers in Qt
 *  \author Dimitris Karatzas
 */

// Convert core watermarking session pixels into a displayable QImage
QImage imagePreviewFromSession(const WatermarkCore::ImageSession* session);
// Extract original input pixels from the session into a displayable QImage
QImage originalPreviewFromSession(WatermarkCore::ImageSession* session);
