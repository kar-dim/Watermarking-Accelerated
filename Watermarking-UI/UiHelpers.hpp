#pragma once

#include "AnimatedButton.hpp"
#include <exception>
#include <functional>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QString>
#include <QStyle>
#include <QWidget>
#include <WatermarkCore.hpp>

/*!
 *  \brief  Shared widget helpers for file pickers, status labels, and default encoder options
 *  \author Dimitris Karatzas
 */

inline QWidget* makeFileRow(QLineEdit*& field, const QString& buttonText, QWidget* parent, const std::function<void()>& browse) {
    auto* row = new QWidget(parent);
    row->setProperty("workflowContainer", true);
    auto* layout = new QHBoxLayout(row);
    layout->setContentsMargins(0, 0, 0, 0);
    field = new QLineEdit(row);
    auto* button = new AnimatedButton(buttonText, row);
    layout->addWidget(field, 1);
    layout->addWidget(button);
    QObject::connect(button, &QPushButton::clicked, row, browse);
    return row;
}

inline void setStatus(QLabel* label, const QString& message, const QString& state) {
    label->setText(message);
    if (label->property("state").toString() != state) {
        label->setProperty("state", state);
        label->style()->unpolish(label);
        label->style()->polish(label);
    }
}

inline QString hardwareEncoderDefaultsText(const int device) {
    try {
        return QString::fromStdString(WatermarkCore::defaultVideoEncoderOptions(true, device));
    } catch (const std::exception&) { return {}; }
}
