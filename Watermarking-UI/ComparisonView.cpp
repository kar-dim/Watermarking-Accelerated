#include "ComparisonView.hpp"
#include "ThemePalette.hpp"
#include <algorithm>
#include <cmath>
#include <QMouseEvent>
#include <QPainter>
#include <QPaintEvent>
#include <QPen>
#include <QPointF>
#include <QRectF>
#include <QSizeF>
#include <QWheelEvent>

/*!
 *  \brief  Implementation of the side by side image comparison widget
 *  \author Dimitris Karatzas
 */

// Configure comparison canvas defaults and enable mouse tracking
ComparisonView::ComparisonView(QWidget* parent) : QWidget(parent) {
    setObjectName("comparisonCanvas");
    setMinimumSize(400, 330);
    setMouseTracking(true);
    setCursor(Qt::OpenHandCursor);
}

// Assign images and reset view to initial center position
void ComparisonView::setImages(const QImage& original, const QImage& watermarked) {
    original_ = original;
    watermarked_ = watermarked;
    resetView();
}

// Reset zoom and pan coordinates back to default centered position
void ComparisonView::resetView() {
    zoom_ = 1.0;
    divider_ = 0.5;
    pan_ = QPointF();
    update();
}

// Render split view of both images with divider line and labels
void ComparisonView::paintEvent(QPaintEvent*) {
    QPainter painter(this);
    painter.setRenderHint(QPainter::SmoothPixmapTransform);
    painter.fillRect(rect(), themePalette().canvas);
    if (!hasImages()) {
        painter.setPen(themePalette().muted);
        painter.drawText(rect(), Qt::AlignCenter, "Preview will appear here");
        return;
    }

    const double fit = std::min(double(width()) / watermarked_.width(), double(height()) / watermarked_.height());
    const QSizeF size = QSizeF(watermarked_.size()) * fit * zoom_;
    const QPointF center = rect().center() + pan_;
    const QRectF target(center.x() - size.width() / 2.0, center.y() - size.height() / 2.0, size.width(), size.height());
    const int splitX = static_cast<int>(std::round(divider_ * width()));

    painter.save();
    painter.setClipRect(QRect(0, 0, splitX, height()));
    painter.drawImage(target, original_);
    painter.restore();
    painter.save();
    painter.setClipRect(QRect(splitX, 0, width() - splitX, height()));
    painter.drawImage(target, watermarked_);
    painter.restore();

    // Both images share one target rectangle, the divider changes only clipping
    painter.setPen(QPen(themePalette().divider, 2));
    painter.drawLine(splitX, 0, splitX, height());
    painter.setBrush(themePalette().divider);
    painter.drawEllipse(QPointF(splitX, height() / 2.0), 16, 16);
    painter.setPen(themePalette().dividerInk);
    painter.drawText(QRectF(splitX - 13, height() / 2.0 - 12, 26, 24), Qt::AlignCenter, "↔");
    painter.setPen(themePalette().text);
    painter.fillRect(QRect(12, 12, 68, 24), QColor(0, 0, 0, 170));
    painter.fillRect(QRect(width() - 113, 12, 101, 24), QColor(0, 0, 0, 170));
    painter.drawText(QRect(12, 12, 68, 24), Qt::AlignCenter, "Original");
    painter.drawText(QRect(width() - 113, 12, 101, 24), Qt::AlignCenter, "Watermarked");
}

// Handle mouse wheel zoom centered around cursor position
void ComparisonView::wheelEvent(QWheelEvent* event) {
    if (!hasImages()) {
        event->ignore();
        return;
    }
    const double oldZoom = zoom_;
    zoom_ = std::clamp(zoom_ * std::pow(1.2, event->angleDelta().y() / 120.0), 1.0, 16.0);
    const QPointF relative = event->position() - rect().center();
    pan_ = relative - (relative - pan_) * (zoom_ / oldZoom);
    update();
    event->accept();
}

// Handle mouse press to start panning or dragging divider
void ComparisonView::mousePressEvent(QMouseEvent* event) {
    if (!hasImages() || event->button() != Qt::LeftButton)
        return;
    lastMouse_ = event->position();
    draggingDivider_ = std::abs(lastMouse_.x() - divider_ * width()) <= 18.0;
    panning_ = !draggingDivider_;
    setCursor(draggingDivider_ ? Qt::SplitHCursor : Qt::ClosedHandCursor);
    event->accept();
}

// Update divider split position or pan image offset on mouse move
void ComparisonView::mouseMoveEvent(QMouseEvent* event) {
    if (!hasImages())
        return;
    if (draggingDivider_) {
        divider_ = std::clamp(event->position().x() / width(), 0.0, 1.0);
        update();
    } else if (panning_) {
        pan_ += event->position() - lastMouse_;
        lastMouse_ = event->position();
        update();
    } else {
        setCursor(std::abs(event->position().x() - divider_ * width()) <= 18.0 ? Qt::SplitHCursor : Qt::OpenHandCursor);
    }
}

// Reset dragging and panning flags on mouse release
void ComparisonView::mouseReleaseEvent(QMouseEvent*) {
    draggingDivider_ = false;
    panning_ = false;
    setCursor(Qt::OpenHandCursor);
}

// Double click resets zoom and pan back to initial state
void ComparisonView::mouseDoubleClickEvent(QMouseEvent* event) {
    if (event->button() == Qt::LeftButton && hasImages()) {
        resetView();
        event->accept();
    }
}
