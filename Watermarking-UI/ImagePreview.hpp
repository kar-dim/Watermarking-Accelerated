#pragma once

#include <QImage>
#include <QLabel>

namespace WatermarkCore {
struct ImageSession;
}

/*!
 *  \brief  Image previews and conversion of core session buffers for Qt
 *  \author Dimitris Karatzas
 */

class ImagePreviewView : public QLabel {
  public:
    explicit ImagePreviewView(QWidget* parent = nullptr);
    void setImage(const QImage& image);
    [[nodiscard]] bool hasImage() const { return !image_.isNull(); }

  protected:
    void paintEvent(QPaintEvent* event) override;

  private:
    QImage image_;
};

// Convert core watermarking session pixels into a displayable QImage
QImage imagePreviewFromSession(const WatermarkCore::ImageSession* session);
// Same, into "preview": its buffer is rewritten in place when the size matches and no other QImage shares it, otherwise it gets a new one
void updatePreviewFromSession(const WatermarkCore::ImageSession* session, QImage& preview);
// Extract original input pixels from the session into a displayable QImage
QImage originalPreviewFromSession(WatermarkCore::ImageSession* session);
