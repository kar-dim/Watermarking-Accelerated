#pragma once

#include <QImage>
#include <QPointF>
#include <QWidget>

/*!
 *  \brief  Split slider view for comparing original and watermarked images
 *  \author Dimitris Karatzas
 */
// Widget for visual comparison between original and watermarked images
class ComparisonView : public QWidget {
  public:
    explicit ComparisonView(QWidget* parent = nullptr);
    // Set both images and reset zoom and pan positions
    void setImages(const QImage& original, const QImage& watermarked);
    // Check if both original and watermarked images are loaded
    [[nodiscard]] bool hasImages() const { return !original_.isNull() && !watermarked_.isNull(); }

  protected:
    // Event handlers for rendering, zoom, pan, and divider drag
    void paintEvent(QPaintEvent* event) override;
    void wheelEvent(QWheelEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void mouseDoubleClickEvent(QMouseEvent* event) override;

  private:
    // Reset zoom and pan back to default center view
    void resetView();
    // Loaded images, viewport coordinates, and split interaction state
    QImage original_;
    QImage watermarked_;
    QPointF pan_;
    QPointF lastMouse_;
    double zoom_ = 1.0;
    double divider_ = 0.5;
    bool draggingDivider_ = false;
    bool panning_ = false;
};
