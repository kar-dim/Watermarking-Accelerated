#pragma once

#include <QImage>
#include <QPointF>
#include <QWidget>

/*!
 *  \brief  Split slider view for comparing original and watermarked images
 *  \author Dimitris Karatzas
 */
class ComparisonView : public QWidget {
  public:
    explicit ComparisonView(QWidget* parent = nullptr);
    void setImages(const QImage& original, const QImage& watermarked);
    [[nodiscard]] bool hasImages() const { return !original_.isNull() && !watermarked_.isNull(); }

  protected:
    void paintEvent(QPaintEvent* event) override;
    void wheelEvent(QWheelEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void mouseDoubleClickEvent(QMouseEvent* event) override;

  private:
    void resetView();
    QImage original_;
    QImage watermarked_;
    QPointF pan_;
    QPointF lastMouse_;
    double zoom_ = 1.0;
    double divider_ = 0.5;
    bool draggingDivider_ = false;
    bool panning_ = false;
};
