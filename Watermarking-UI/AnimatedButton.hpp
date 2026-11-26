#pragma once

#include <QColor>
#include <QPushButton>
#include <QVariantAnimation>

/*!
 *  \brief  Custom push button with hover and click animations
 *  \author Dimitris Karatzas
 */
class AnimatedButton : public QPushButton {
  public:
    explicit AnimatedButton(const QString& text, QWidget* parent = nullptr);
    void refreshAppearance();

  protected:
    void paintEvent(QPaintEvent* event) override;
    void enterEvent(QEnterEvent* event) override;
    void leaveEvent(QEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void changeEvent(QEvent* event) override;
    QSize sizeHint() const override;

  private:
    void animateTo(const QColor& color);
    QColor baseColor() const;
    QColor hoverColor() const;
    QColor pressedColor() const;
    bool primary() const;
    QVariantAnimation animation_;
    QColor fill_;
};
