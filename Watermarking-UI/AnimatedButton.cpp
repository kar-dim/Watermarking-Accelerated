#include "AnimatedButton.hpp"
#include "ThemePalette.hpp"
#include <algorithm>
#include <QEnterEvent>
#include <QEvent>
#include <QMouseEvent>
#include <QPainter>
#include <QPaintEvent>
#include <QPen>
#include <QRectF>
#include <QSize>
#include <QVariant>

/*!
 *  \brief  Implementation of the animated push button
 *  \author Dimitris Karatzas
 */

AnimatedButton::AnimatedButton(const QString& text, QWidget* parent) : QPushButton(text, parent) {
    setMinimumHeight(40);
    setCursor(Qt::PointingHandCursor);
    animation_.setDuration(120);
    animation_.setEasingCurve(QEasingCurve::OutQuad);
    connect(&animation_, &QVariantAnimation::valueChanged, this, [this](const QVariant& value) {
        fill_ = value.value<QColor>();
        update();
    });
}

void AnimatedButton::refreshAppearance() {
    animation_.stop();
    fill_ = baseColor();
    update();
}

bool AnimatedButton::primary() const { return objectName() == "primaryButton" && !property("stop").toBool(); }

QColor AnimatedButton::baseColor() const { return primary() ? themePalette().accent : themePalette().surface; }
QColor AnimatedButton::hoverColor() const { return primary() ? themePalette().accent.lighter(110) : themePalette().surfaceHover; }
QColor AnimatedButton::pressedColor() const { return primary() ? themePalette().accent.darker(110) : themePalette().surfacePressed; }

void AnimatedButton::animateTo(const QColor& color) {
    animation_.stop();
    animation_.setStartValue(fill_.isValid() ? fill_ : baseColor());
    animation_.setEndValue(color);
    animation_.start();
}

QSize AnimatedButton::sizeHint() const {
    const QSize base = QPushButton::sizeHint();
    return QSize(std::max(base.width(), fontMetrics().horizontalAdvance(text()) + 36), std::max(base.height(), 40));
}

void AnimatedButton::paintEvent(QPaintEvent*) {
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);
    painter.setOpacity(isEnabled() ? 1.0 : 0.4);
    const QRectF bounds = rect().adjusted(1, 1, -1, -1);
    const QColor background = fill_.isValid() ? fill_ : baseColor();
    painter.setBrush(background);
    if (primary())
        painter.setPen(Qt::NoPen);
    else
        painter.setPen(QPen(themePalette().accent, hasFocus() ? 2 : 1));
    painter.drawRoundedRect(bounds, 8, 8);
    painter.setPen(primary() ? themePalette().ink : themePalette().text);
    painter.setFont(font());
    painter.drawText(rect(), Qt::AlignCenter, text());
}

void AnimatedButton::enterEvent(QEnterEvent* event) {
    animateTo(hoverColor());
    QPushButton::enterEvent(event);
}

void AnimatedButton::leaveEvent(QEvent* event) {
    animateTo(baseColor());
    QPushButton::leaveEvent(event);
}

void AnimatedButton::mousePressEvent(QMouseEvent* event) {
    animateTo(pressedColor());
    QPushButton::mousePressEvent(event);
}

void AnimatedButton::mouseReleaseEvent(QMouseEvent* event) {
    animateTo(underMouse() ? hoverColor() : baseColor());
    QPushButton::mouseReleaseEvent(event);
}

void AnimatedButton::changeEvent(QEvent* event) {
    QPushButton::changeEvent(event);
    if (event->type() == QEvent::EnabledChange) {
        animation_.stop();
        fill_ = baseColor();
        update();
    }
}
