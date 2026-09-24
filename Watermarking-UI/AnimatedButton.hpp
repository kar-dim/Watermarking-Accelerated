#pragma once

#include <QColor>
#include <QPushButton>
#include <QVariantAnimation>

/*!
 *  \brief  Custom push button with smooth hover and click animations
 *  \author Dimitris Karatzas
 */
// Custom button with smooth color animations for hover and click
class AnimatedButton : public QPushButton {
  public:
    explicit AnimatedButton(const QString& text, QWidget* parent = nullptr);
    // Refresh button colors when the theme or properties change
    void refreshAppearance();

  protected:
    void paintEvent(QPaintEvent* event) override;
    void enterEvent(QEnterEvent* event) override;
    void leaveEvent(QEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void changeEvent(QEvent* event) override;
    // Calculate preferred size based on button text width and padding
    QSize sizeHint() const override;

  private:
    // Smoothly transition the button fill to a new target color
    void animateTo(const QColor& color);
    // Background colors for standard, hover, and pressed states
    QColor baseColor() const;
    QColor hoverColor() const;
    QColor pressedColor() const;
    // Check if this button is styled as a primary action button
    bool primary() const;
    // Color animation controller and active fill color
    QVariantAnimation animation_;
    QColor fill_;
};
