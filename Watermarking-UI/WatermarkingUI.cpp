#include "WatermarkingUI.h"
#include "AnimatedButton.hpp"
#include "ThemePalette.hpp"
#include "common_utils.hpp"
#include <algorithm>
#include <exception>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>
#include <QAbstractItemView>
#include <QApplication>
#include <QChar>
#include <QColor>
#include <QCursor>
#include <QDir>
#include <QDropEvent>
#include <QFileDialog>
#include <QFileInfo>
#include <QFrame>
#include <QFormLayout>
#include <QGuiApplication>
#include <QHeaderView>
#include <QHBoxLayout>
#include <QMessageBox>
#include <QMimeData>
#include <QPixmap>
#include <QPoint>
#include <QRect>
#include <QScrollArea>
#include <QScreen>
#include <QSizePolicy>
#include <QStyle>
#include <QTableWidgetItem>
#include <QThread>
#include <QUrl>
#include <QVBoxLayout>
#include <WatermarkCore.hpp>

/*!
 *  \brief  Implementation of the main graphical interface for Watermarking
 *  \author Dimitris Karatzas
 */

using namespace WatermarkCore;
namespace fs = std::filesystem;

namespace {
constexpr auto imageFilter = "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)";

QColor successColor() { return themePalette().accent; }

// Create an input row combining a text path field with a browse button
QWidget* makeFileRow(QLineEdit*& field, const QString& buttonText, QWidget* parent, const std::function<void()>& browse) {
    auto* row = new QWidget(parent);
    auto* layout = new QHBoxLayout(row);
    layout->setContentsMargins(0, 0, 0, 0);
    field = new QLineEdit(row);
    auto* button = new AnimatedButton(buttonText, row);
    button->setObjectName("secondaryButton");
    layout->addWidget(field, 1);
    layout->addWidget(button);
    QObject::connect(button, &QPushButton::clicked, row, browse);
    return row;
}

// Set text message and update dynamic CSS styling state on status label
void setStatus(QLabel* label, const QString& message, const QString& state) {
    label->setText(message);
    label->setProperty("state", state);
    label->style()->unpolish(label);
    label->style()->polish(label);
}

// Extract and validate first acceptable file or folder path from drop mime data
QString supportedDropPath(const QMimeData* data, const bool folder) {
    if (data == nullptr || !data->hasUrls())
        return {};
    for (const QUrl& url : data->urls()) {
        if (!url.isLocalFile())
            continue;
        const QString path = url.toLocalFile();
        const QFileInfo info(path);
        if (folder ? info.isDir() : (info.isFile() && CommonUtils::hasSupportedImageExtension(fs::path(info.fileName().toStdWString()))))
            return path;
    }
    return {};
}
} // namespace

// Construct settings inputs for password, window size p, and PSNR strength
QWidget* WatermarkingUI::createSettingsRow(SettingsControls& controls, QWidget* parent) {
    auto* row = new QWidget(parent);
    row->setObjectName("settingsRow");
    auto* layout = new QHBoxLayout(row);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(10);
    const auto addColumn = [row, layout](const QString& label, QWidget* input, const int stretch) {
        auto* column = new QWidget(row);
        column->setObjectName("settingsColumn");
        auto* columnLayout = new QVBoxLayout(column);
        columnLayout->setContentsMargins(0, 0, 0, 0);
        columnLayout->setSpacing(4);
        columnLayout->addWidget(new QLabel(label, column));
        columnLayout->addWidget(input);
        layout->addWidget(column, stretch);
        return column;
    };
    controls.password = new QLineEdit(row);
    controls.password->setPlaceholderText("Use the same password when detecting");
    controls.password->setEchoMode(QLineEdit::PasswordEchoOnEdit);
    addColumn("Watermark password", controls.password, 3);
    controls.p = new QComboBox(row);
    for (const int p : {3, 5, 7, 9})
        controls.p->addItem(QString::number(p), p);
    addColumn("Window size (p)", controls.p, 1);
    controls.psnr = new QDoubleSpinBox(row);
    controls.psnr->setRange(1.0, 100.0);
    controls.psnr->setDecimals(1);
    controls.psnr->setValue(40.0);
    controls.psnrColumn = addColumn("PSNR (dB)", controls.psnr, 1);
    return row;
}

// Construct main application window, configure layouts, and register UI tabs
WatermarkingUI::WatermarkingUI(QWidget* parent) : QMainWindow(parent) {
    setWindowTitle("Watermarking");
    setAcceptDrops(true);
    setMinimumSize(760, 520);
    // Use 90% of the screen's work area height so larger screens get more preview space.
    QScreen* startupScreen = QGuiApplication::screenAt(QCursor::pos());
    if (startupScreen == nullptr)
        startupScreen = QGuiApplication::primaryScreen();
    const QRect workArea = startupScreen->availableGeometry();
    resize(std::min(1120, workArea.width()), static_cast<int>(workArea.height() * 0.9));

    auto* scroll = new QScrollArea(this);
    scroll->setObjectName("mainScroll");
    scroll->setWidgetResizable(true);
    scroll->setFrameShape(QFrame::NoFrame);
    auto* central = new QWidget(scroll);
    auto* root = new QVBoxLayout(central);
    root->setContentsMargins(20, 12, 20, 16);
    root->setSpacing(12);
    scroll->setWidget(central);
    setCentralWidget(scroll);

    auto* deviceRow = new QWidget(central);
    auto* deviceLayout = new QHBoxLayout(deviceRow);
    deviceLayout->setContentsMargins(0, 0, 0, 0);
    const std::string backend = getBackendName();
    const QString backendName = backend == "eigen" ? "CPU" : QString::fromStdString(backend).toUpper();
    auto* backendLabel = new QLabel(QString("Backend: %1").arg(backendName), deviceRow);
    backendLabel->setObjectName("backendLabel");
    deviceLayout->addWidget(backendLabel);
    deviceLayout->addStretch();
    std::vector<std::string> availableDevices;
    try {
        availableDevices = getAvailableDevices();
    } catch (const std::exception&) { availableDevices.clear(); }
    // A picker is useful only when the GPU backend exposes multiple devices
    if (backend != "eigen" && availableDevices.size() > 1) {
        deviceLayout->addWidget(new QLabel("Device", deviceRow));
        deviceComboBox = new QComboBox(deviceRow);
        deviceComboBox->setObjectName("deviceComboBox");
        for (const auto& device : availableDevices)
            deviceComboBox->addItem(QString::fromStdString(device));
        deviceLayout->addWidget(deviceComboBox);
    } else {
        const QString name = availableDevices.empty() ? "No device found" : QString::fromStdString(availableDevices.front());
        auto* deviceName = new QLabel(name, deviceRow);
        deviceName->setObjectName("deviceName");
        deviceLayout->addWidget(deviceName);
    }
    root->addWidget(deviceRow);

    tabs = new QTabWidget(central);
    tabs->setObjectName("workflowTabs");
    root->addWidget(tabs, 1);

    auto* singlePage = new QWidget(tabs);
    singlePage->setObjectName("workflowPage");
    auto* singleLayout = new QVBoxLayout(singlePage);
    singleLayout->setContentsMargins(18, 20, 18, 18);
    singleLayout->setSpacing(14);
    auto* singleFiles = new QFormLayout;
    singleFiles->setLabelAlignment(Qt::AlignRight);
    singleMode = new QComboBox(singlePage);
    singleMode->setObjectName("singleMode");
    singleMode->addItem("Embed ME watermark");
    singleMode->addItem("Detect ME watermark");
    singleFiles->addRow("Operation", singleMode);
    singleFiles->addRow("Input image", makeFileRow(singleInput, "Browse", singlePage, [this] {
        if (busy)
            return;
        const QString path = QFileDialog::getOpenFileName(this, "Choose an image", QString(), imageFilter);
        if (!path.isEmpty())
            selectSingleImage(path);
    }));
    singleInput->setObjectName("singleInput");
    auto* singleOutputRow = makeFileRow(singleOutput, "Browse", singlePage, [this] {
        if (busy)
            return;
        const QString path = QFileDialog::getSaveFileName(this, "Choose output image", singleOutput->text(), imageFilter);
        if (!path.isEmpty())
            singleOutput->setText(path);
    });
    singleFiles->addRow("Output image", singleOutputRow);
    singleOutput->setObjectName("singleOutput");
    singleLayout->addLayout(singleFiles);
    singleLayout->addWidget(createSettingsRow(singleSettings, singlePage));
    singleSettings.password->setObjectName("singlePassword");
    singleSettings.psnrColumn->setObjectName("singlePsnrColumn");
    auto* singleActions = new QHBoxLayout;
    previewButton = new AnimatedButton("Preview watermark", singlePage);
    previewButton->setObjectName("primaryButton");
    saveButton = new AnimatedButton("Save image", singlePage);
    saveButton->setEnabled(false);
    singleActions->addWidget(previewButton);
    singleActions->addWidget(saveButton);
    singleActions->addStretch();
    singleLayout->addLayout(singleActions);
    singleStatus = new QLabel("Choose an image, then preview the ME watermark.", singlePage);
    singleStatus->setObjectName("singleStatus");
    singleLayout->addWidget(singleStatus);
    auto* comparisonHint = new QLabel("Drag the divider to compare. Drag the image to pan, scroll to zoom, or double-click to reset.", singlePage);
    comparisonHint->setObjectName("mutedLabel");
    singleLayout->addWidget(comparisonHint);
    singleImageView = new ComparisonView(singlePage);
    singleImageView->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    singleLayout->addWidget(singleImageView, 1);
    singleDetectArea = new QWidget(singlePage);
    singleDetectArea->setObjectName("singleDetectArea");
    auto* detectLayout = new QVBoxLayout(singleDetectArea);
    detectLayout->setContentsMargins(0, 0, 0, 0);
    singleDetectionResult = new QLabel("ME correlation will appear here", singleDetectArea);
    singleDetectionResult->setObjectName("detectionResult");
    singleDetectionResult->setAlignment(Qt::AlignCenter);
    singleDetectionResult->setTextInteractionFlags(Qt::TextSelectableByMouse);
    singleDetectionResult->setFixedHeight(92);
    singleDetectionResult->setMinimumWidth(360);
    singleDetectionResult->setMaximumWidth(520);
    detectLayout->addWidget(singleDetectionResult, 0, Qt::AlignLeft);
    detectLayout->addStretch(1);
    singleDetectArea->hide();
    singleLayout->addWidget(singleDetectArea, 1);
    tabs->addTab(singlePage, "Single image");

    auto* batchPage = new QWidget(tabs);
    batchPage->setObjectName("workflowPage");
    auto* batchLayout = new QVBoxLayout(batchPage);
    batchLayout->setContentsMargins(18, 20, 18, 18);
    batchLayout->setSpacing(12);
    auto* batchFiles = new QFormLayout;
    batchFiles->addRow("Image folder", makeFileRow(batchFolder, "Browse", batchPage, [this] {
        if (busy)
            return;
        const QString folder = QFileDialog::getExistingDirectory(this, "Choose an image folder", batchFolder->text());
        if (!folder.isEmpty())
            selectBatchFolder(folder);
    }));
    batchMode = new QComboBox(batchPage);
    batchMode->setObjectName("batchMode");
    batchMode->addItem("Embed ME watermark");
    batchMode->addItem("Detect ME watermark");
    batchFiles->addRow("Operation", batchMode);
    batchLayout->addLayout(batchFiles);
    batchLayout->addWidget(createSettingsRow(batchSettings, batchPage));
    batchSettings.password->setObjectName("batchPassword");
    batchSettings.psnrColumn->setObjectName("batchPsnrColumn");
    batchFolder->setObjectName("batchFolder");
    batchOutputHint = new QLabel("Embedded files go to a watermark_output folder inside the selected folder.", batchPage);
    batchOutputHint->setObjectName("mutedLabel");
    batchLayout->addWidget(batchOutputHint);
    auto* batchActions = new QHBoxLayout;
    batchStartButton = new AnimatedButton("Start batch", batchPage);
    batchStartButton->setObjectName("primaryButton");
    batchCancelButton = new AnimatedButton("Stop after current image", batchPage);
    batchCancelButton->setEnabled(false);
    batchActions->addWidget(batchStartButton);
    batchActions->addWidget(batchCancelButton);
    batchActions->addStretch();
    batchLayout->addLayout(batchActions);
    batchStatus = new QLabel("Select a folder to build the queue.", batchPage);
    batchStatus->setObjectName("batchStatus");
    batchLayout->addWidget(batchStatus);
    batchProgress = new QProgressBar(batchPage);
    batchProgress->setValue(0);
    batchLayout->addWidget(batchProgress);
    batchTable = new QTableWidget(batchPage);
    batchTable->setObjectName("batchQueue");
    batchTable->setColumnCount(3);
    batchTable->setHorizontalHeaderLabels({"File", "Status", "Result"});
    batchTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
    batchTable->setSelectionBehavior(QAbstractItemView::SelectRows);
    batchTable->setAlternatingRowColors(true);
    batchTable->verticalHeader()->setVisible(false);
    batchTable->horizontalHeader()->setSectionResizeMode(0, QHeaderView::Stretch);
    batchTable->horizontalHeader()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
    batchTable->horizontalHeader()->setSectionResizeMode(2, QHeaderView::Stretch);
    batchLayout->addWidget(batchTable, 1);
    tabs->addTab(batchPage, "Batch images");

    auto* benchmarkPage = new QWidget(tabs);
    benchmarkPage->setObjectName("workflowPage");
    auto* benchmarkLayout = new QVBoxLayout(benchmarkPage);
    benchmarkLayout->setContentsMargins(18, 20, 18, 18);
    benchmarkLayout->setSpacing(14);
    auto* benchmarkIntro = new QLabel("Sweep p, PSNR, and the bundled sample images. The live preview and timings update as each case finishes.", benchmarkPage);
    benchmarkIntro->setObjectName("mutedLabel");
    benchmarkIntro->setWordWrap(true);
    benchmarkLayout->addWidget(benchmarkIntro);
    benchmarkStartButton = new AnimatedButton("Start benchmark", benchmarkPage);
    benchmarkStartButton->setObjectName("primaryButton");
    benchmarkLayout->addWidget(benchmarkStartButton, 0, Qt::AlignLeft);
    benchmarkStatus = new QLabel("Ready to benchmark", benchmarkPage);
    benchmarkStatus->setObjectName("benchmarkStatus");
    benchmarkLayout->addWidget(benchmarkStatus);
    benchmarkProgress = new QProgressBar(benchmarkPage);
    benchmarkProgress->setObjectName("benchmarkProgress");
    benchmarkProgress->setValue(0);
    benchmarkLayout->addWidget(benchmarkProgress);
    benchmarkImageView = new QLabel("Live benchmark image will appear here", benchmarkPage);
    benchmarkImageView->setObjectName("imageCanvas");
    benchmarkImageView->setAlignment(Qt::AlignCenter);
    benchmarkImageView->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
    benchmarkImageView->setMinimumSize(400, 280);
    benchmarkLayout->addWidget(benchmarkImageView, 1);
    tabs->addTab(benchmarkPage, "Benchmark");

    dropOverlay = new QLabel(tabs);
    dropOverlay->setObjectName("dropOverlay");
    dropOverlay->setAlignment(Qt::AlignCenter);
    dropOverlay->setAttribute(Qt::WA_TransparentForMouseEvents);
    dropOverlay->hide();

    connect(singleInput, &QLineEdit::textChanged, this, [this] { invalidateSingleResult(); });
    connect(singleSettings.password, &QLineEdit::textChanged, this, [this] { invalidateSingleResult(); });
    connect(singleSettings.p, &QComboBox::currentIndexChanged, this, [this] { invalidateSingleResult(); });
    connect(singleSettings.psnr, &QDoubleSpinBox::valueChanged, this, [this] { invalidateSingleResult(); });
    connect(singleMode, &QComboBox::currentIndexChanged, this, [this, singleFiles, singleOutputRow, comparisonHint](const int index) {
        const bool detect = index == 1;
        singleFiles->setRowVisible(singleOutputRow, !detect);
        previewButton->setText(detect ? "Detect watermark" : "Preview watermark");
        saveButton->setVisible(!detect);
        comparisonHint->setVisible(!detect);
        singleImageView->setVisible(!detect);
        singleDetectArea->setVisible(detect);
        singleSettings.psnrColumn->setVisible(!detect);
        invalidateSingleResult();
    });
    connect(singleSettings.password, &QLineEdit::textChanged, batchSettings.password, [this](const QString& value) {
        if (batchSettings.password->text() != value)
            batchSettings.password->setText(value);
    });
    connect(batchSettings.password, &QLineEdit::textChanged, singleSettings.password, [this](const QString& value) {
        if (singleSettings.password->text() != value)
            singleSettings.password->setText(value);
    });
    connect(singleSettings.p, &QComboBox::currentIndexChanged, batchSettings.p, [this](const int index) {
        if (batchSettings.p->currentIndex() != index)
            batchSettings.p->setCurrentIndex(index);
    });
    connect(batchSettings.p, &QComboBox::currentIndexChanged, singleSettings.p, [this](const int index) {
        if (singleSettings.p->currentIndex() != index)
            singleSettings.p->setCurrentIndex(index);
    });
    connect(singleSettings.psnr, &QDoubleSpinBox::valueChanged, batchSettings.psnr, [this](const double value) {
        if (batchSettings.psnr->value() != value)
            batchSettings.psnr->setValue(value);
    });
    connect(batchSettings.psnr, &QDoubleSpinBox::valueChanged, singleSettings.psnr, [this](const double value) {
        if (singleSettings.psnr->value() != value)
            singleSettings.psnr->setValue(value);
    });
    if (deviceComboBox)
        connect(deviceComboBox, &QComboBox::currentIndexChanged, this, [this] { invalidateSingleResult(); });
    connect(previewButton, &QPushButton::clicked, this, [this] { startSingleOperation(); });
    connect(saveButton, &QPushButton::clicked, this, [this] { saveSingleImage(); });
    connect(batchMode, &QComboBox::currentIndexChanged, this, [this](const int index) {
        batchOutputHint->setText(index == 0 ? "Embedded files go to a watermark_output folder inside the selected folder." : "Detection reports one correlation per image, no files are written.");
        batchSettings.psnrColumn->setVisible(index == 0);
    });
    connect(batchFolder, &QLineEdit::editingFinished, this, [this] { refreshBatchQueue(); });
    connect(batchStartButton, &QPushButton::clicked, this, [this] { startBatch(); });
    connect(batchCancelButton, &QPushButton::clicked, this, [this] { cancelBatch(); });
    connect(benchmarkStartButton, &QPushButton::clicked, this, [this] { startBenchmark(); });
    connect(tabs, &QTabWidget::currentChanged, this, [this] { dropOverlay->hide(); });
    qApp->installEventFilter(this);
}

void WatermarkingUI::selectSingleImage(const QString& path) {
    singleInput->setText(path);
    const QFileInfo info(path);
    singleOutput->setText(info.dir().filePath(info.completeBaseName() + "_watermarked." + info.suffix()));
}

void WatermarkingUI::selectBatchFolder(const QString& path) {
    batchFolder->setText(path);
    refreshBatchQueue();
}

ImageOptions WatermarkingUI::imageOptions(const bool batch) const {
    const SettingsControls& controls = batch ? batchSettings : singleSettings;
    return ImageOptions{controls.password->text(), controls.p->currentData().toInt(), static_cast<float>(controls.psnr->value()), deviceComboBox ? deviceComboBox->currentIndex() : 0};
}

void WatermarkingUI::setBusy(const bool running) {
    busy = running;
    singleSettings.password->setEnabled(!running);
    singleSettings.p->setEnabled(!running);
    singleSettings.psnr->setEnabled(!running);
    batchSettings.password->setEnabled(!running);
    batchSettings.p->setEnabled(!running);
    batchSettings.psnr->setEnabled(!running);
    singleInput->setEnabled(!running);
    singleOutput->setEnabled(!running);
    singleMode->setEnabled(!running);
    batchFolder->setEnabled(!running);
    batchMode->setEnabled(!running);
    if (deviceComboBox)
        deviceComboBox->setEnabled(!running);
    previewButton->setEnabled(!running);
    saveButton->setEnabled(!running && singleMode->currentIndex() == 0 && singleWorker != nullptr && singleWorker->session() != nullptr);
    batchStartButton->setEnabled(!running);
    batchCancelButton->setEnabled(running && batchWorker != nullptr);
    benchmarkStartButton->setEnabled(!running || benchmarkWorker != nullptr);
    benchmarkStartButton->setProperty("stop", benchmarkWorker != nullptr);
    benchmarkStartButton->setText(benchmarkWorker != nullptr ? (benchmarkWorker->isInterruptionRequested() ? "Stopping..." : "Stop benchmark") : "Start benchmark");
    static_cast<AnimatedButton*>(benchmarkStartButton)->refreshAppearance();
}

void WatermarkingUI::invalidateSingleResult() {
    if (busy)
        return;
    // Drop the old session while leaving its comparison image visible
    if (singleWorker != nullptr) {
        delete singleWorker;
        singleWorker = nullptr;
    }
    saveButton->setEnabled(false);
    if (singleMode->currentIndex() == 1) {
        singleDetectionResult->setText("ME correlation will appear here");
        setStatus(singleStatus, "Choose an image, then detect the ME watermark.", "idle");
    } else {
        setStatus(singleStatus,
            singleImageView->hasImages() ? "Showing the previous preview. Preview again to apply the current settings or device." : "Choose an image, then preview the ME watermark.", "idle");
    }
}

void WatermarkingUI::startSingleOperation() {
    if (busy)
        return;
    if (singleInput->text().trimmed().isEmpty() || singleSettings.password->text().isEmpty()) {
        QMessageBox::warning(this, "Missing input", "Choose an input image and enter a watermark password.");
        return;
    }
    invalidateSingleResult();
    const bool detect = singleMode->currentIndex() == 1;
    singleWorker = new SingleImageWorker(imageOptions(false), singleInput->text().trimmed(), detect, this);
    setStatus(singleStatus, detect ? "Loading image and detecting ME watermark..." : "Loading image and embedding ME watermark...", "running");
    setBusy(true);
    connect(singleWorker, &QThread::finished, this, [this, detect] {
        setBusy(false);
        if (!singleWorker->error().isEmpty()) {
            setStatus(singleStatus, singleWorker->error(), "error");
            QMessageBox::critical(this, detect ? "Detection failed" : "Preview failed", singleWorker->error());
            delete singleWorker;
            singleWorker = nullptr;
            return;
        }
        if (detect) {
            const QString value = QString::number(*singleWorker->correlation(), 'f', 4);
            singleDetectionResult->setText("ME correlation: " + value);
            setStatus(singleStatus, "ME correlation: " + value, "done");
            return;
        }
        singleImageView->setImages(singleWorker->original(), singleWorker->preview());
        saveButton->setEnabled(true);
        setStatus(singleStatus, "ME watermark ready. Save writes this exact result.", "done");
    });
    singleWorker->start();
}

void WatermarkingUI::saveSingleImage() {
    if (busy || singleWorker == nullptr || singleWorker->session() == nullptr)
        return;
    QString output = singleOutput->text().trimmed();
    if (output.isEmpty()) {
        output = QFileDialog::getSaveFileName(this, "Save watermarked image", QString(), imageFilter);
        if (output.isEmpty())
            return;
        singleOutput->setText(output);
    }
    std::error_code pathError;
    if (fs::equivalent(singleInput->text().toStdString(), output.toStdString(), pathError)) {
        QMessageBox::warning(this, "Invalid output", "Input and output must be different files.");
        return;
    }
    QApplication::setOverrideCursor(Qt::WaitCursor);
    try {
        saveImageExact(singleWorker->session(), output.toStdString());
        QApplication::restoreOverrideCursor();
        setStatus(singleStatus, "Saved watermarked image to " + output, "done");
    } catch (const std::exception& error) {
        QApplication::restoreOverrideCursor();
        setStatus(singleStatus, QString::fromUtf8(error.what()), "error");
        QMessageBox::critical(this, "Save failed", QString::fromUtf8(error.what()));
    }
}

void WatermarkingUI::refreshBatchQueue() {
    if (busy)
        return;
    batchFiles.clear();
    batchTable->setRowCount(0);
    batchProgress->setValue(0);
    if (batchFolder->text().trimmed().isEmpty()) {
        setStatus(batchStatus, "Select a folder to build the queue.", "idle");
        return;
    }
    const fs::path folder(batchFolder->text().trimmed().toStdString());
    try {
        batchFiles = CommonUtils::getValidImageFiles(folder);
    } catch (const std::exception& error) {
        setStatus(batchStatus, QString::fromUtf8(error.what()), "error");
        return;
    }
    batchTable->setRowCount(static_cast<int>(batchFiles.size()));
    for (size_t index = 0; index < batchFiles.size(); ++index) {
        batchTable->setItem(static_cast<int>(index), 0, new QTableWidgetItem(QString::fromStdString(batchFiles[index].filename().string())));
        batchTable->setItem(static_cast<int>(index), 1, new QTableWidgetItem("Queued"));
        batchTable->setItem(static_cast<int>(index), 2, new QTableWidgetItem());
    }
    batchCompleted = 0;
    batchProgress->setRange(0, std::max(1, static_cast<int>(batchFiles.size())));
    setStatus(batchStatus, batchFiles.empty() ? "The folder contains no supported image files." : QString("%1 images queued.").arg(static_cast<int>(batchFiles.size())),
        batchFiles.empty() ? "error" : "idle");
}

void WatermarkingUI::startBatch() {
    if (busy)
        return;
    if (batchFolder->text().trimmed().isEmpty() || batchSettings.password->text().isEmpty()) {
        QMessageBox::warning(this, "Missing input", "Choose an image folder and enter a watermark password.");
        return;
    }
    refreshBatchQueue();
    if (batchFiles.empty()) {
        QMessageBox::warning(this, "No images", "The folder contains no supported image files.");
        return;
    }
    invalidateSingleResult();
    const fs::path folder(batchFolder->text().trimmed().toStdString());
    const bool embed = batchMode->currentIndex() == 0;
    const fs::path outputDir = folder / "watermark_output";
    batchWorker = new BatchImageWorker(imageOptions(true), batchFiles, outputDir, embed, this);
    setStatus(batchStatus, QString("Processing %1 images...").arg(static_cast<int>(batchFiles.size())), "running");
    setBusy(true);
    connect(batchWorker, &BatchImageWorker::itemState, this, [this](const int index, const QString& state, const QString& detail) {
        if (index < 0 || index >= batchTable->rowCount())
            return;
        auto* status = batchTable->item(index, 1);
        if (state == "Done") {
            status->setText(QString(QChar(0x2713)) + " Done");
            status->setForeground(successColor());
            ++batchCompleted;
        } else if (state == "Error") {
            status->setText(QString(QChar(0x2715)) + " Error");
            status->setForeground(themePalette().error);
            ++batchCompleted;
        } else {
            status->setText("Working");
            status->setForeground(themePalette().warning);
        }
        batchTable->item(index, 2)->setText(detail);
        batchProgress->setValue(batchCompleted);
        if (state != "Working")
            batchStatus->setText(QString("%1 of %2 images finished").arg(batchCompleted).arg(batchTable->rowCount()));
    });
    connect(batchWorker, &BatchImageWorker::summaryReady, this, [this, outputDir, embed](const int succeeded, const int failed, const double seconds, const bool canceled, const QString& fatalError) {
        batchWorker->wait();
        for (int index = 0; index < batchTable->rowCount(); ++index) {
            const QString state = batchTable->item(index, 1)->text();
            if (state == "Queued" || state == "Working") {
                batchTable->item(index, 1)->setText(canceled ? "Stopped" : "Skipped");
                batchTable->item(index, 1)->setForeground(themePalette().muted);
            }
        }
        const int remaining = batchTable->rowCount() - succeeded - failed;
        const QString heading = fatalError.isEmpty() ? (canceled ? "Batch stopped" : "Batch complete") : "Batch failed";
        const QString details = QString("Succeeded: %1\nFailed: %2\nNot processed: %3\nTime: %4 seconds%5%6")
                                    .arg(succeeded)
                                    .arg(failed)
                                    .arg(remaining)
                                    .arg(seconds, 0, 'f', 2)
                                    .arg(embed ? "\nOutput: " + QString::fromStdString(outputDir.string()) : "")
                                    .arg(fatalError.isEmpty() ? "" : "\n\n" + fatalError);
        batchWorker->deleteLater();
        batchWorker = nullptr;
        setBusy(false);
        setStatus(batchStatus, heading + QString(" (%1 succeeded, %2 failed)").arg(succeeded).arg(failed), fatalError.isEmpty() && failed == 0 ? "done" : "error");
        QMessageBox::information(this, heading, details);
    });
    batchWorker->start();
}

void WatermarkingUI::cancelBatch() {
    if (batchWorker == nullptr)
        return;
    batchWorker->requestInterruption();
    batchCancelButton->setEnabled(false);
    setStatus(batchStatus, "Stopping after in-flight files finish...", "running");
}

void WatermarkingUI::startBenchmark() {
    if (benchmarkWorker != nullptr) {
        benchmarkWorker->requestInterruption();
        resetBenchmarkView();
        benchmarkStartButton->setEnabled(false);
        benchmarkStartButton->setText("Stopping...");
        setStatus(benchmarkStatus, "Stopping benchmark...", "running");
        return;
    }
    if (busy)
        return;
    invalidateSingleResult();
    const int device = deviceComboBox ? deviceComboBox->currentIndex() : 0;
    benchmarkWorker = new BenchmarkWorker(device, this);
    benchmarkProgress->setValue(0);
    setStatus(benchmarkStatus, "Running benchmark sweep...", "running");
    setBusy(true);
    connect(benchmarkWorker, &BenchmarkWorker::progressUpdated, this, [this](const int current, const int total) {
        if (benchmarkWorker == nullptr || benchmarkWorker->isInterruptionRequested())
            return;
        benchmarkProgress->setMaximum(total);
        benchmarkProgress->setValue(current);
    });
    connect(benchmarkWorker, &BenchmarkWorker::resultReady, this, &WatermarkingUI::onBenchmarkResult);
    connect(benchmarkWorker, &BenchmarkWorker::benchmarkCanceled, this, &WatermarkingUI::finishBenchmarkCanceled);
    connect(benchmarkWorker, &BenchmarkWorker::benchmarkFinished, this, [this](const double embedFps, const double detectFps, const int score) {
        if (benchmarkWorker->isInterruptionRequested()) {
            finishBenchmarkCanceled();
            return;
        }
        benchmarkWorker->wait();
        benchmarkWorker->deleteLater();
        benchmarkWorker = nullptr;
        setBusy(false);
        if (score <= 0) {
            setStatus(benchmarkStatus, "Benchmark failed", "error");
            QMessageBox::critical(this, "Benchmark failed", "The benchmark could not finish. Check the selected device, sample files, and available memory.");
            return;
        }
        setStatus(benchmarkStatus, "Benchmark complete", "done");
        QMessageBox::information(this, "Benchmark score",
            QString("Hardware: %1\nGeometric mean embed: %2 FPS\nGeometric mean detect: %3 FPS\n\nScore: %4")
                .arg(QString::fromStdString(getDeviceName(deviceComboBox ? deviceComboBox->currentIndex() : -1)))
                .arg(embedFps, 0, 'f', 1)
                .arg(detectFps, 0, 'f', 1)
                .arg(score));
    });
    benchmarkWorker->start();
}

void WatermarkingUI::resetBenchmarkView() {
    benchmarkPreviewImage = QImage();
    benchmarkImageView->clear();
    benchmarkImageView->setText("Live benchmark image will appear here");
    benchmarkProgress->setRange(0, 100);
    benchmarkProgress->setValue(0);
}

void WatermarkingUI::finishBenchmarkCanceled() {
    if (benchmarkWorker == nullptr)
        return;
    benchmarkWorker->wait();
    benchmarkWorker->deleteLater();
    benchmarkWorker = nullptr;
    resetBenchmarkView();
    setBusy(false);
    setStatus(benchmarkStatus, "Ready to benchmark", "idle");
}

void WatermarkingUI::onBenchmarkResult(
    const QImage& image, const int p, const float psnr, const double embedTime, const double detectTime, const double embedFps, const double detectFps, const QString& file, const float correlation) {
    if (benchmarkWorker == nullptr || benchmarkWorker->isInterruptionRequested())
        return;
    benchmarkPreviewImage = image;
    updateImageViews();
    setStatus(benchmarkStatus,
        QString("%1 | p=%2 | PSNR=%3 dB | Corr=%4\nEmbed: %5 ms (%6 FPS)  |  Detect: %7 ms (%8 FPS)")
            .arg(file)
            .arg(p)
            .arg(psnr, 0, 'f', 1)
            .arg(correlation, 0, 'f', 4)
            .arg(embedTime, 0, 'f', 2)
            .arg(embedFps, 0, 'f', 1)
            .arg(detectTime, 0, 'f', 2)
            .arg(detectFps, 0, 'f', 1),
        "running");
}

void WatermarkingUI::updateImageViews() {
    if (!benchmarkPreviewImage.isNull())
        benchmarkImageView->setPixmap(QPixmap::fromImage(benchmarkPreviewImage).scaled(benchmarkImageView->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
}

void WatermarkingUI::resizeEvent(QResizeEvent* event) {
    QMainWindow::resizeEvent(event);
    updateImageViews();
}

bool WatermarkingUI::eventFilter(QObject* watched, QEvent* event) {
    const QEvent::Type type = event->type();
    if (type != QEvent::DragEnter && type != QEvent::DragMove && type != QEvent::DragLeave && type != QEvent::Drop)
        return QMainWindow::eventFilter(watched, event);

    auto* target = qobject_cast<QWidget*>(watched);
    if (target == nullptr || target->window() != this)
        return QMainWindow::eventFilter(watched, event);

    if (type == QEvent::DragLeave) {
        if (!rect().contains(mapFromGlobal(QCursor::pos())))
            dropOverlay->hide();
        return QMainWindow::eventFilter(watched, event);
    }

    auto* drop = static_cast<QDropEvent*>(event);
    const int tab = tabs->currentIndex();
    const bool batch = tab == 1;
    const QString path = (!busy && (tab == 0 || batch)) ? supportedDropPath(drop->mimeData(), batch) : QString();
    if (path.isEmpty()) {
        dropOverlay->hide();
        drop->ignore();
        return true;
    }

    drop->acceptProposedAction();
    if (type == QEvent::Drop) {
        dropOverlay->hide();
        if (batch)
            selectBatchFolder(path);
        else
            selectSingleImage(path);
    } else {
        QWidget* page = tabs->currentWidget();
        dropOverlay->setGeometry(QRect(page->mapTo(tabs, QPoint(0, 0)), page->size()));
        dropOverlay->setText(batch ? QString("Drop folder to load image queue\n%1").arg(QFileInfo(path).fileName()) : QString("Drop image to select input\n%1").arg(QFileInfo(path).fileName()));
        dropOverlay->raise();
        dropOverlay->show();
    }
    return true;
}

void WatermarkingUI::closeEvent(QCloseEvent* event) {
    if (singleWorker != nullptr && singleWorker->isRunning()) {
        singleWorker->requestInterruption();
        singleWorker->wait();
    }
    if (batchWorker != nullptr && batchWorker->isRunning()) {
        batchWorker->requestInterruption();
        batchWorker->wait();
    }
    if (benchmarkWorker != nullptr && benchmarkWorker->isRunning()) {
        benchmarkWorker->requestInterruption();
        benchmarkWorker->wait();
    }
    QMainWindow::closeEvent(event);
}
