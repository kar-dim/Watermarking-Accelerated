#pragma once

#include <algorithm>
#include <functional>
#include <QAbstractItemView>
#include <QComboBox>
#include <QKeyEvent>
#include <QMouseEvent>
#include <QStandardItemModel>
#include <QStyleOptionComboBox>
#include <QStylePainter>
#include <vector>

/*!
 * \brief A device picker with checkable choices and the standard combo box appearance
 * \author Dimitris Karatzas
 */
class MultiDeviceComboBox : public QComboBox {
  public:
    explicit MultiDeviceComboBox(QWidget* parent = nullptr) : QComboBox(parent) {
        auto* choices = new QStandardItemModel(this);
        setModel(choices);
        view()->viewport()->installEventFilter(this);
        view()->installEventFilter(this);
        setAccessibleName("Batch devices");
        setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
        connect(choices, &QStandardItemModel::itemChanged, this, [this] {
            updateSummary();
            if (selectionChanged)
                selectionChanged();
        });
        updateSummary();
    }

    void addDevice(const QString& name, const int device, const bool checked) {
        auto* item = new QStandardItem(name);
        item->setData(device, Qt::UserRole);
        item->setFlags(Qt::ItemIsEnabled | Qt::ItemIsSelectable | Qt::ItemIsUserCheckable);
        item->setCheckState(checked ? Qt::Checked : Qt::Unchecked);
        static_cast<QStandardItemModel*>(model())->appendRow(item);
        updateSummary();
    }

    std::vector<int> selectedDevices() const {
        std::vector<int> devices;
        for (int row = 0; row < count(); ++row)
            if (itemData(row, Qt::CheckStateRole).toInt() == Qt::Checked)
                devices.push_back(itemData(row).toInt());
        return devices;
    }

    // look up row by stored device ID instead of assuming list index equals ordinal
    QString deviceName(const int device) const {
        const int row = findData(device);
        return row >= 0 ? itemText(row) : QString();
    }

    std::function<void()> selectionChanged;

  protected:
    void paintEvent(QPaintEvent*) override {
        QStylePainter painter(this);
        QStyleOptionComboBox option;
        initStyleOption(&option);
        const QRect editRect = style()->subControlRect(QStyle::CC_ComboBox, &option, QStyle::SC_ComboBoxEditField, this);
        option.currentText = fontMetrics().elidedText(summary_, Qt::ElideRight, std::max(0, editRect.width()));
        painter.drawComplexControl(QStyle::CC_ComboBox, option);
        painter.drawControl(QStyle::CE_ComboBoxLabel, option);
    }

    bool eventFilter(QObject* watched, QEvent* event) override {
        // Consume row clicks before QComboBox closes the popup after the first choice
        if (watched == view()->viewport() && (event->type() == QEvent::MouseButtonPress || event->type() == QEvent::MouseButtonRelease)) {
            auto* mouse = static_cast<QMouseEvent*>(event);
            if (mouse->button() == Qt::LeftButton) {
                if (event->type() == QEvent::MouseButtonRelease) {
                    const auto index = view()->indexAt(mouse->position().toPoint());
                    view()->setCurrentIndex(index);
                    toggle(index.row());
                }
                return true;
            }
        }
        if (watched == view() && event->type() == QEvent::KeyPress) {
            const int key = static_cast<QKeyEvent*>(event)->key();
            if (key == Qt::Key_Space || key == Qt::Key_Return || key == Qt::Key_Enter) {
                toggle(view()->currentIndex().row());
                return true;
            }
        }
        return QComboBox::eventFilter(watched, event);
    }

    void wheelEvent(QWheelEvent* event) override { event->ignore(); }

  private:
    void toggle(const int row) {
        if (row < 0 || row >= count())
            return;
        setItemData(row, itemData(row, Qt::CheckStateRole).toInt() == Qt::Checked ? Qt::Unchecked : Qt::Checked, Qt::CheckStateRole);
    }

    void updateSummary() {
        QStringList names;
        for (int row = 0; row < count(); ++row)
            if (itemData(row, Qt::CheckStateRole).toInt() == Qt::Checked)
                names.push_back(itemText(row));
        summary_ = count() == 0 ? "No devices found" : names.isEmpty() ? "Select devices" : names.join(", ");
        setToolTip(count() == 0 ? "No compatible compute devices were found." : names.isEmpty() ? "Select one or more devices." : names.join('\n'));
        setAccessibleDescription(summary_);
        update();
    }

    QString summary_;
};
