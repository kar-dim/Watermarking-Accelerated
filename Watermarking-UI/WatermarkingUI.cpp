#include "WatermarkingUI.h"
#include "AnimatedButton.hpp"
#include "ImagePreview.hpp"
#include "MultiDeviceComboBox.hpp"
#include <ImageBatch.hpp>
#include "ThemePalette.hpp"
#include <CorrelationFormat.hpp>
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
#include <QDesktopServices>
#include <QItemSelectionModel>
#include <set>
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
    row->setProperty("workflowContainer", true);
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
        if (folder ? info.isDir() : (info.isFile() && WatermarkCore::hasSupportedImageExtension(fs::path(info.fileName().toStdWString()))))
            return path;
    }
    return {};
}
} // namespace

// Construct settings inputs for password, window size p, and PSNR strength
QWidget* WatermarkingUI::createSettingsRow(SettingsControls& controls, QWidget* parent) {
    auto* row = new QWidget(parent);
    row->setProperty("workflowContainer", true);
    row->setObjectName("settingsRow");
    auto* layout = new QHBoxLayout(row);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(10);
    const auto addColumn = [row, layout](const QString& label, QWidget* input, const int stretch) {
        auto* column = new QWidget(row);
        column->setProperty("workflowContainer", true);
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
    // scale to 90% of work area height to give larger screens more preview space
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
    central->setProperty("workflowContainer", true);
    auto* root = new QVBoxLayout(central);
    root->setContentsMargins(20, 12, 20, 16);
    root->setSpacing(12);
    scroll->setWidget(central);
    setCentralWidget(scroll);

    auto* deviceRow = new QWidget(central);
    deviceRow->setProperty("workflowContainer", true);
    auto* deviceLayout = new QHBoxLayout(deviceRow);
    deviceLayout->setContentsMargins(0, 0, 0, 0);
    const std::string backend = getBackendName();
    const QString backendName = backend == "eigen" ? "CPU" : QString::fromStdString(backend).toUpper();
    auto* backendLabel = new QLabel(QString("Backend: %1").arg(backendName), deviceRow);
    backendLabel->setObjectName("backendLabel");
    deviceLayout->addWidget(backendLabel);
    deviceLayout->addStretch();
    devicePickerArea = new QWidget(deviceRow);
    devicePickerArea->setProperty("workflowContainer", true);
    devicePickerArea->setObjectName("devicePickerArea");
    auto pickerPolicy = devicePickerArea->sizePolicy();
    // Keep tabs aligned while batch tabs use their own device pickers
    pickerPolicy.setRetainSizeWhenHidden(true);
    devicePickerArea->setSizePolicy(pickerPolicy);
    auto* pickerLayout = new QHBoxLayout(devicePickerArea);
    pickerLayout->setContentsMargins(0, 0, 0, 0);
    deviceLayout->addWidget(devicePickerArea);
    std::vector<std::string> availableDevices;
    try {
        availableDevices = getAvailableDevices();
    } catch (const std::exception&) { availableDevices.clear(); }
    // A picker is useful only when the GPU backend exposes multiple devices
    if (backend != "eigen" && availableDevices.size() > 1) {
        pickerLayout->addWidget(new QLabel("Device", devicePickerArea));
        deviceComboBox = new QComboBox(devicePickerArea);
        deviceComboBox->setObjectName("deviceComboBox");
        for (const auto& device : availableDevices)
            deviceComboBox->addItem(QString::fromStdString(device));
        pickerLayout->addWidget(deviceComboBox);
    } else {
        const QString name = availableDevices.empty() ? "No device found" : QString::fromStdString(availableDevices.front());
        auto* deviceName = new QLabel(name, devicePickerArea);
        deviceName->setObjectName("deviceName");
        pickerLayout->addWidget(deviceName);
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
    singleMode->addItem("Embed watermark");
    singleMode->addItem("Detect watermark");
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
    singleStatus = new QLabel("Choose an image, then preview the watermark.", singlePage);
    singleStatus->setObjectName("singleStatus");
    singleLayout->addWidget(singleStatus);
    auto* comparisonHint = new QLabel("Drag the divider to compare. Drag the image to pan, scroll to zoom, or double-click to reset.", singlePage);
    comparisonHint->setObjectName("mutedLabel");
    singleLayout->addWidget(comparisonHint);
    singleImageView = new ComparisonView(singlePage);
    singleImageView->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    singleLayout->addWidget(singleImageView, 1);
    singleDetectArea = new QWidget(singlePage);
    singleDetectArea->setProperty("workflowContainer", true);
    singleDetectArea->setObjectName("singleDetectArea");
    auto* detectLayout = new QVBoxLayout(singleDetectArea);
    detectLayout->setContentsMargins(0, 0, 0, 0);
    auto* detectionCard = new QWidget(singleDetectArea);
    detectionCard->setProperty("workflowContainer", true);
    detectionCard->setObjectName("detectionCard");
    auto* cardLayout = new QHBoxLayout(detectionCard);
    cardLayout->setContentsMargins(24, 24, 24, 24);
    cardLayout->setSpacing(28);
    singleDetectionPreview = new ImagePreviewView(detectionCard);
    singleDetectionPreview->setObjectName("detectionThumbnail");
    singleDetectionPreview->setMinimumSize(240, 180);
    cardLayout->addWidget(singleDetectionPreview, 3);
    auto* resultLayout = new QVBoxLayout;
    resultLayout->setSpacing(8);
    resultLayout->addStretch();
    singleDetectionFile = new QLabel("No image selected", detectionCard);
    singleDetectionFile->setObjectName("detectionFile");
    singleDetectionFile->setTextFormat(Qt::PlainText);
    singleDetectionFile->setWordWrap(true);
    singleDetectionFile->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
    resultLayout->addWidget(singleDetectionFile);
    auto* detectionCaption = new QLabel("Correlation", detectionCard);
    detectionCaption->setObjectName("detectionCaption");
    resultLayout->addWidget(detectionCaption);
    singleDetectionResult = new QLabel("Not run", detectionCard);
    singleDetectionResult->setObjectName("detectionResult");
    singleDetectionResult->setProperty("state", "idle");
    singleDetectionResult->setTextInteractionFlags(Qt::TextSelectableByMouse);
    resultLayout->addWidget(singleDetectionResult);
    auto* detectionHint = new QLabel("Higher values mean a stronger match.", detectionCard);
    detectionHint->setObjectName("detectionHint");
    detectionHint->setWordWrap(true);
    resultLayout->addWidget(detectionHint);
    resultLayout->addStretch();
    cardLayout->addLayout(resultLayout, 2);
    detectLayout->addWidget(detectionCard, 1);
    singleDetectArea->hide();
    singleLayout->addWidget(singleDetectArea, 1);
    tabs->addTab(singlePage, "Single image");

    auto* batchPage = new QWidget(tabs);
    batchPage->setObjectName("workflowPage");
    auto* batchLayout = new QVBoxLayout(batchPage);
    batchLayout->setContentsMargins(18, 20, 18, 18);
    batchLayout->setSpacing(12);
    auto* batchForm = new QFormLayout;
    batchForm->setLabelAlignment(Qt::AlignRight);
    batchMode = new QComboBox(batchPage);
    batchMode->setObjectName("batchMode");
    batchMode->addItems({"Embed watermark", "Detect watermark"});
    batchForm->addRow("Operation", batchMode);
    imageBatchDevices = new MultiDeviceComboBox(batchPage);
    imageBatchDevices->setObjectName("imageBatchDevices");
    for (int device = 0; device < static_cast<int>(availableDevices.size()); ++device)
        imageBatchDevices->addDevice(QString::fromStdString(availableDevices[device]), device, device == (deviceComboBox ? deviceComboBox->currentIndex() : 0));
    imageBatchDevices->selectionChanged = [this] { batchStartButton->setEnabled(!busy && !batchFiles.empty() && !imageBatchDevices->selectedDevices().empty()); };
    batchOutputRow = makeFileRow(batchOutput, "Browse", batchPage, [this] {
        if (busy)
            return;
        const QString folder = QFileDialog::getExistingDirectory(this, "Choose batch output folder", batchOutput->text());
        if (!folder.isEmpty())
            batchOutput->setText(folder);
    });
    batchOutput->setObjectName("batchOutput");
    batchForm->addRow("Output folder", batchOutputRow);
    batchForm->addRow("Devices", imageBatchDevices);
    batchLayout->addLayout(batchForm);
    batchLayout->addWidget(createSettingsRow(batchSettings, batchPage));
    batchSettings.password->setObjectName("batchPassword");
    batchSettings.psnrColumn->setObjectName("batchPsnrColumn");
    batchOutputHint = new QLabel("Embedded images keep their format, existing files are kept.", batchPage);
    batchOutputHint->setObjectName("mutedLabel");
    batchLayout->addWidget(batchOutputHint);
    auto* batchFilesRow = new QHBoxLayout;
    batchAddFiles = new AnimatedButton("Add images", batchPage);
    batchAddFiles->setObjectName("batchAddFiles");
    batchAddFolder = new AnimatedButton("Add folder", batchPage);
    batchRemove = new AnimatedButton("Remove selected", batchPage);
    batchClear = new AnimatedButton("Clear", batchPage);
    for (auto* button : {batchAddFiles, batchAddFolder, batchRemove, batchClear})
        batchFilesRow->addWidget(button);
    batchFilesRow->addStretch();
    batchLayout->addLayout(batchFilesRow);
    auto* batchActions = new QHBoxLayout;
    batchStartButton = new AnimatedButton("Start batch", batchPage);
    batchStartButton->setObjectName("primaryButton");
    batchCancelButton = new AnimatedButton("Cancel batch", batchPage);
    batchCancelButton->setToolTip("Stop taking new images and finish processing and saving images already in flight.");
    batchCancelButton->setEnabled(false);
    auto* batchOpen = new AnimatedButton("Open output folder", batchPage);
    batchActions->addWidget(batchStartButton);
    batchActions->addWidget(batchCancelButton);
    batchActions->addStretch();
    batchActions->addWidget(batchOpen);
    batchLayout->addLayout(batchActions);
    connect(batchOpen, &QPushButton::clicked, this, [this] {
        const QString path = batchOutput->text().trimmed();
        if (QFileInfo(path).isDir())
            QDesktopServices::openUrl(QUrl::fromLocalFile(path));
    });
    connect(batchAddFiles, &QPushButton::clicked, this, [this] { addImageBatchPaths(QFileDialog::getOpenFileNames(this, "Add images", QString(), imageFilter)); });
    connect(batchAddFolder, &QPushButton::clicked, this, [this] {
        const QString folder = QFileDialog::getExistingDirectory(this, "Add images from a folder");
        if (!folder.isEmpty())
            addImageBatchPaths({folder});
    });
    connect(batchClear, &QPushButton::clicked, this, [this] {
        batchFiles.clear();
        refreshBatchQueue();
    });
    connect(batchRemove, &QPushButton::clicked, this, [this] {
        std::set<int, std::greater<int>> rows;
        for (const auto& index : batchTable->selectionModel()->selectedRows())
            rows.insert(index.row());
        for (const int row : rows)
            batchFiles.erase(batchFiles.begin() + row);
        refreshBatchQueue();
    });
    batchStatus = new QLabel("Add images or drop files and folders to build the queue.", batchPage);
    batchStatus->setObjectName("batchStatus");
    batchStatus->setWordWrap(true);
    auto* batchStatusRow = new QHBoxLayout;
    batchStatusRow->addWidget(batchStatus, 1);
    batchElapsed = new QLabel("Elapsed: 0.0 s", batchPage);
    batchElapsed->setObjectName("batchElapsed");
    batchStatusRow->addWidget(batchElapsed);
    batchLayout->addLayout(batchStatusRow);
    batchTimer = new QTimer(this);
    batchTimer->setInterval(100);
    connect(batchTimer, &QTimer::timeout, this, [this] {
        batchElapsed->setText(QString("Elapsed: %1 s").arg(batchClock.elapsed() / 1000.0, 0, 'f', 1));
        pollImageBatch();
    });
    batchProgress = new QProgressBar(batchPage);
    batchProgress->setValue(0);
    batchLayout->addWidget(batchProgress);
    batchTable = new QTableWidget(batchPage);
    batchTable->setObjectName("batchQueue");
    batchTable->setColumnCount(4);
    batchTable->setHorizontalHeaderLabels({"File", "Status", "Device", "Result"});
    batchTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
    batchTable->setSelectionBehavior(QAbstractItemView::SelectRows);
    batchTable->setSelectionMode(QAbstractItemView::ExtendedSelection);
    batchTable->setWordWrap(false);
    batchTable->setAlternatingRowColors(true);
    batchTable->verticalHeader()->setVisible(false);
    batchTable->horizontalHeader()->setSectionResizeMode(0, QHeaderView::Stretch);
    batchTable->horizontalHeader()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
    batchTable->horizontalHeader()->resizeSection(2, 180);
    batchTable->horizontalHeader()->setSectionResizeMode(3, QHeaderView::Stretch);
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
    createVideoTab();
    createVideoBatchTab();

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
    connect(batchMode, &QComboBox::currentIndexChanged, this, [this, batchForm](const int index) {
        batchOutputHint->setText(index == 0 ? "Embedded images keep their format, existing files are kept." : "Detection reports one correlation per image, no files are written.");
        batchForm->setRowVisible(batchOutputRow, index == 0);
        batchSettings.psnrColumn->setVisible(index == 0);
    });
    connect(batchStartButton, &QPushButton::clicked, this, [this] { startBatch(); });
    connect(batchCancelButton, &QPushButton::clicked, this, [this] { cancelBatch(); });
    connect(benchmarkStartButton, &QPushButton::clicked, this, [this] { startBenchmark(); });
    connect(tabs, &QTabWidget::currentChanged, this, [this] {
        dropOverlay->hide();
        devicePickerArea->setVisible(tabs->currentWidget() != videoBatchPage && tabs->currentIndex() != 2);
    });
    refreshBatchQueue();
    qApp->installEventFilter(this);
}

void WatermarkingUI::selectSingleImage(const QString& path) {
    singleInput->setText(path);
    const QFileInfo info(path);
    singleOutput->setText(info.dir().filePath(info.completeBaseName() + "_watermarked." + info.suffix()));
}

void WatermarkingUI::addImageBatchPaths(const QStringList& paths) {
    if (busy)
        return;
    try {
        for (const auto& selected : paths) {
            std::vector<fs::path> files;
            const QFileInfo info(selected);
            if (info.isDir())
                files = WatermarkCore::getValidImageFiles(fs::path(selected.toStdWString()));
            else if (info.isFile() && WatermarkCore::hasSupportedImageExtension(fs::path(selected.toStdWString())))
                files.emplace_back(selected.toStdWString());
            for (const auto& file : files) {
                const QFileInfo input(QString::fromStdWString(file.wstring()));
                const QString canonical = input.canonicalFilePath();
                const QString path = canonical.isEmpty() ? input.absoluteFilePath() : canonical;
                const bool duplicate = std::ranges::any_of(batchFiles, [&](const auto& existing) { return QString::fromStdWString(existing.wstring()).compare(path, Qt::CaseInsensitive) == 0; });
                if (!duplicate)
                    batchFiles.emplace_back(path.toStdWString());
            }
        }
        if (batchOutput->text().isEmpty() && !batchFiles.empty())
            batchOutput->setText(QFileInfo(QString::fromStdWString(batchFiles.front().wstring())).dir().filePath("watermark_output"));
        refreshBatchQueue();
    } catch (const std::exception& error) { QMessageBox::warning(this, "Image files", QString::fromUtf8(error.what())); }
}

ImageOptions WatermarkingUI::imageOptions(const bool batch) const {
    const SettingsControls& controls = batch ? batchSettings : singleSettings;
    return ImageOptions{controls.password->text(), controls.p->currentData().toInt(), static_cast<float>(controls.psnr->value()), deviceComboBox ? deviceComboBox->currentIndex() : 0};
}

void WatermarkingUI::setBusy(const bool running) {
    busy = running;
    if (!running && videoProbePending && !videoProbeTimer->isActive())
        videoProbeTimer->start(0);
    videoInputs->setEnabled(!running);
    videoStart->setEnabled(!running);
    videoCancel->setEnabled(running && videoWorker != nullptr && videoWorker->isRunning());
    videoBatchInputs->setEnabled(!running);
    videoBatchAddFiles->setEnabled(!running);
    videoBatchAddFolder->setEnabled(!running);
    videoBatchRemove->setEnabled(!running);
    videoBatchClear->setEnabled(!running);
    updateVideoBatchControls();
    videoBatchCancel->setEnabled(running && videoBatchWorker && videoBatchWorker->isRunning() && !videoBatchWorker->cancellationRequested());
    videoBatchSkip->setEnabled(running && videoBatchActive && videoBatchWorker && videoBatchWorker->isRunning() && !videoBatchWorker->cancellationRequested());
    singleSettings.password->setEnabled(!running);
    singleSettings.p->setEnabled(!running);
    singleSettings.psnr->setEnabled(!running);
    batchSettings.password->setEnabled(!running);
    batchSettings.p->setEnabled(!running);
    batchSettings.psnr->setEnabled(!running);
    singleInput->setEnabled(!running);
    singleOutput->setEnabled(!running);
    singleMode->setEnabled(!running);
    batchOutputRow->setEnabled(!running);
    for (auto* button : {batchAddFiles, batchAddFolder, batchRemove, batchClear})
        button->setEnabled(!running);
    batchMode->setEnabled(!running);
    imageBatchDevices->setEnabled(!running);
    if (deviceComboBox)
        deviceComboBox->setEnabled(!running);
    previewButton->setEnabled(!running);
    saveButton->setEnabled(!running && singleMode->currentIndex() == 0 && singleWorker != nullptr && singleWorker->session() != nullptr);
    batchStartButton->setEnabled(!running && !batchFiles.empty() && !imageBatchDevices->selectedDevices().empty());
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
        setStatus(singleDetectionResult, "Not run", "idle");
        singleDetectionPreview->setImage({});
        const QString path = singleInput->text().trimmed();
        singleDetectionFile->setText(path.isEmpty() ? "No image selected" : QFileInfo(path).fileName());
        singleDetectionFile->setToolTip(path);
        setStatus(singleStatus, "Choose an image, then detect the watermark.", "idle");
    } else {
        setStatus(singleStatus, singleImageView->hasImages() ? "Showing the previous preview. Preview again to apply the current settings or device." : "Choose an image, then preview the watermark.",
            "idle");
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
    setStatus(singleStatus, detect ? "Loading image and detecting watermark..." : "Loading image and embedding watermark...", "running");
    if (detect)
        setStatus(singleDetectionResult, "Running", "running");
    setBusy(true);
    connect(singleWorker, &QThread::finished, this, [this, detect] {
        if (closing)
            return;
        setBusy(false);
        if (!singleWorker->error().isEmpty()) {
            setStatus(singleStatus, singleWorker->error(), "error");
            if (detect)
                setStatus(singleDetectionResult, "Failed", "error");
            QMessageBox::critical(this, detect ? "Detection failed" : "Preview failed", singleWorker->error());
            delete singleWorker;
            singleWorker = nullptr;
            return;
        }
        if (detect) {
            const QString value = QString::fromStdString(formatCorrelationPercent(*singleWorker->correlation()));
            setStatus(singleDetectionResult, value, "done");
            singleDetectionPreview->setImage(singleWorker->original());
            setStatus(singleStatus, "Detection complete.", "done");
            return;
        }
        singleImageView->setImages(singleWorker->original(), singleWorker->preview());
        saveButton->setEnabled(true);
        setStatus(singleStatus, "watermark ready. Save writes this exact result.", "done");
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
    batchTable->setRowCount(0);
    batchProgress->setValue(0);
    batchElapsed->setText("Elapsed: 0.0 s");
    batchTable->setRowCount(static_cast<int>(batchFiles.size()));
    for (size_t index = 0; index < batchFiles.size(); ++index) {
        auto* file = new QTableWidgetItem(QString::fromStdWString(batchFiles[index].filename().wstring()));
        file->setToolTip(QString::fromStdWString(batchFiles[index].wstring()));
        batchTable->setItem(static_cast<int>(index), 0, file);
        auto* state = new QTableWidgetItem("Queued");
        state->setData(Qt::UserRole, static_cast<int>(ImageBatchState::Queued));
        batchTable->setItem(static_cast<int>(index), 1, state);
        batchTable->setItem(static_cast<int>(index), 2, new QTableWidgetItem());
        batchTable->setItem(static_cast<int>(index), 3, new QTableWidgetItem());
    }
    batchCompleted = 0;
    batchProgress->setRange(0, std::max(1, static_cast<int>(batchFiles.size())));
    batchStartButton->setEnabled(!batchFiles.empty() && !imageBatchDevices->selectedDevices().empty());
    setStatus(batchStatus, QString("%1 images queued").arg(batchFiles.size()), "idle");
}

void WatermarkingUI::startBatch() {
    if (busy)
        return;
    if (batchFiles.empty() || batchSettings.password->text().isEmpty() || (batchMode->currentIndex() == 0 && batchOutput->text().trimmed().isEmpty())) {
        QMessageBox::warning(this, "Missing input", "Add images, choose an output folder and enter a watermark password.");
        return;
    }
    refreshBatchQueue();
    if (batchFiles.empty()) {
        QMessageBox::warning(this, "No images", "The folder contains no supported image files.");
        return;
    }
    invalidateSingleResult();
    const bool embed = batchMode->currentIndex() == 0;
    const fs::path outputDir(batchOutput->text().trimmed().toStdWString());
    ImageBatchSettings settings;
    settings.password = batchSettings.password->text().toStdString();
    settings.p = batchSettings.p->currentData().toInt();
    settings.psnr = static_cast<float>(batchSettings.psnr->value());
    settings.detect = !embed;
    settings.deviceIndices = imageBatchDevices->selectedDevices();
    if (settings.deviceIndices.empty()) {
        QMessageBox::warning(this, "Batch devices", "Select at least one image batch device.");
        return;
    }
    std::vector<std::string> inputs;
    for (const auto& path : batchFiles) {
        const auto text = path.u8string();
        inputs.emplace_back(text.begin(), text.end());
    }
    std::vector<ImageBatchJob> jobs;
    try {
        jobs = prepareImageBatch(inputs, batchOutput->text().trimmed().toStdString(), !embed);
    } catch (const std::exception& error) {
        QMessageBox::warning(this, "Image batch", QString::fromUtf8(error.what()));
        return;
    }
    batchWorker = new BatchImageWorker(std::move(settings), std::move(jobs), this);
    setStatus(batchStatus, QString("Processing %1 images...").arg(static_cast<int>(batchFiles.size())), "running");
    setBusy(true);
    connect(batchWorker, &BatchImageWorker::summaryReady, this, [this, outputDir, embed](const int, const int, const double seconds, const bool canceled, const QString& fatalError) {
        if (closing)
            return;
        batchWorker->wait();
        batchTimer->stop();
        pollImageBatch();
        const double elapsed = fatalError.isEmpty() ? seconds : batchClock.elapsed() / 1000.0;
        batchElapsed->setText(QString("Elapsed: %1 s").arg(elapsed, 0, 'f', 1));
        int succeeded = 0, failed = 0;
        for (int index = 0; index < batchTable->rowCount(); ++index) {
            auto* state = batchTable->item(index, 1);
            auto value = static_cast<ImageBatchState>(state->data(Qt::UserRole).toInt());
            if (value != ImageBatchState::Completed && value != ImageBatchState::Failed && value != ImageBatchState::Cancelled) {
                value = canceled ? ImageBatchState::Cancelled : ImageBatchState::Failed;
                state->setData(Qt::UserRole, static_cast<int>(value));
                state->setText(imageBatchStateName(value));
                state->setForeground(canceled ? themePalette().muted : themePalette().error);
                batchTable->item(index, 3)->setText(fatalError);
            }
            succeeded += value == ImageBatchState::Completed;
            failed += value == ImageBatchState::Failed;
        }
        const int remaining = batchTable->rowCount() - succeeded - failed;
        const QString heading = fatalError.isEmpty() ? (canceled ? "Batch stopped" : "Batch complete") : "Batch failed";
        const QString details = QString("Succeeded: %1\nFailed: %2\nNot processed: %3\nTime: %4 seconds%5%6")
                                    .arg(succeeded)
                                    .arg(failed)
                                    .arg(remaining)
                                    .arg(elapsed, 0, 'f', 2)
                                    .arg(embed ? "\nOutput: " + QString::fromStdWString(outputDir.wstring()) : "")
                                    .arg(fatalError.isEmpty() ? "" : "\n\n" + fatalError);
        batchWorker->deleteLater();
        batchWorker = nullptr;
        setBusy(false);
        setStatus(batchStatus, heading + QString(" (%1 succeeded, %2 failed)").arg(succeeded).arg(failed), fatalError.isEmpty() && failed == 0 ? "done" : "error");
        QMessageBox::information(this, heading, details);
    });
    batchClock.start();
    batchTimer->start();
    batchWorker->start();
}

void WatermarkingUI::pollImageBatch() {
    if (!batchWorker)
        return;
    for (const auto& update : batchWorker->takeUpdates()) {
        if (!update.id || update.id > static_cast<std::uint64_t>(batchTable->rowCount()))
            continue;
        const int row = static_cast<int>(update.id - 1);
        auto* state = batchTable->item(row, 1);
        if (update.revision <= state->data(Qt::UserRole + 1).toULongLong())
            continue;
        state->setData(Qt::UserRole, static_cast<int>(update.state));
        state->setData(Qt::UserRole + 1, QVariant::fromValue<qulonglong>(update.revision));
        state->setText(imageBatchStateName(update.state));
        state->setForeground(update.state == ImageBatchState::Completed   ? themePalette().accent
                             : update.state == ImageBatchState::Failed    ? themePalette().error
                             : update.state == ImageBatchState::Cancelled ? themePalette().muted
                                                                          : themePalette().text);
        auto* device = batchTable->item(row, 2);
        device->setText(update.deviceIndex >= 0 && update.deviceIndex < imageBatchDevices->count() ? imageBatchDevices->itemText(update.deviceIndex) : QString());
        device->setToolTip(device->text());
        const QString detail = update.correlation ? QString("Correlation: %1").arg(QString::fromStdString(formatCorrelationPercent(*update.correlation))) : QString::fromStdString(update.detail);
        batchTable->item(row, 3)->setText(detail);
        batchTable->item(row, 3)->setToolTip(detail);
    }
    batchCompleted = 0;
    int active = 0;
    for (int row = 0; row < batchTable->rowCount(); ++row) {
        const auto state = static_cast<ImageBatchState>(batchTable->item(row, 1)->data(Qt::UserRole).toInt());
        batchCompleted += state == ImageBatchState::Completed || state == ImageBatchState::Failed;
        active += state == ImageBatchState::Loading || state == ImageBatchState::Processing || state == ImageBatchState::Saving;
    }
    batchProgress->setValue(batchCompleted);
    if (batchCancelButton->isEnabled())
        batchStatus->setText(QString("%1 of %2 images finished  |  %3 active").arg(batchCompleted).arg(batchTable->rowCount()).arg(active));
}

void WatermarkingUI::cancelBatch() {
    if (batchWorker == nullptr)
        return;
    batchWorker->cancel();
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
        if (closing)
            return;
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
        QString("%1 | p=%2 | PSNR=%3 dB | Correlation=%4\nEmbed: %5 ms (%6 FPS)  |  Detect: %7 ms (%8 FPS)")
            .arg(file)
            .arg(p)
            .arg(psnr, 0, 'f', 1)
            .arg(QString::fromStdString(formatCorrelationPercent(correlation)))
            .arg(embedTime, 0, 'f', 2)
            .arg(embedFps, 0, 'f', 1)
            .arg(detectTime, 0, 'f', 2)
            .arg(detectFps, 0, 'f', 1),
        "running");
}

void WatermarkingUI::updateImageViews() {
    if (!videoPreviewImage.isNull() && videoPreviewEnabled->isChecked())
        videoPreview->setPixmap(QPixmap::fromImage(videoPreviewImage).scaled(videoPreview->contentsRect().size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
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
    const bool batch = tab == 2;
    const bool video = tab == 1;
    const bool videoBatch = tabs->currentWidget() == videoBatchPage;
    QStringList videoBatchPaths;
    QStringList imageBatchPaths;
    if (!busy && videoBatch && drop->mimeData()->hasUrls())
        for (const auto& url : drop->mimeData()->urls())
            if (url.isLocalFile() && QFileInfo(url.toLocalFile()).exists())
                videoBatchPaths.push_back(url.toLocalFile());
    if (!busy && batch && drop->mimeData()->hasUrls())
        for (const auto& url : drop->mimeData()->urls())
            if (url.isLocalFile() && (QFileInfo(url.toLocalFile()).isDir() || WatermarkCore::hasSupportedImageExtension(fs::path(url.toLocalFile().toStdWString()))))
                imageBatchPaths.push_back(url.toLocalFile());
    QString path = !busy && tab == 0 ? supportedDropPath(drop->mimeData(), false) : QString();
    if (!imageBatchPaths.isEmpty())
        path = imageBatchPaths.front();
    if (!videoBatchPaths.isEmpty())
        path = videoBatchPaths.front();
    if (!busy && video && drop->mimeData()->hasUrls())
        for (const auto& url : drop->mimeData()->urls())
            if (url.isLocalFile() && QFileInfo(url.toLocalFile()).isFile()) {
                path = url.toLocalFile();
                break;
            }
    if (path.isEmpty()) {
        dropOverlay->hide();
        drop->ignore();
        return true;
    }

    drop->acceptProposedAction();
    if (type == QEvent::Drop) {
        dropOverlay->hide();
        if (videoBatch)
            addVideoBatchPaths(videoBatchPaths);
        else if (batch)
            addImageBatchPaths(imageBatchPaths);
        else if (video)
            selectVideo(path);
        else
            selectSingleImage(path);
    } else {
        QWidget* page = tabs->currentWidget();
        dropOverlay->setGeometry(QRect(page->mapTo(tabs, QPoint(0, 0)), page->size()));
        dropOverlay->setText(videoBatch ? QString("Drop videos or folders to add to the queue\n%1 items").arg(videoBatchPaths.size())
                             : batch    ? QString("Drop images or folders to add to the queue\n%1 items").arg(imageBatchPaths.size())
                                        : QString(video ? "Drop video to select input\n%1" : "Drop image to select input\n%1").arg(QFileInfo(path).fileName()));
        dropOverlay->raise();
        dropOverlay->show();
    }
    return true;
}

void WatermarkingUI::closeEvent(QCloseEvent* event) {
    // Completion signals can remain queued after wait returns, ignore them during close
    closing = true;
    batchTimer->stop();
    videoTimer->stop();
    videoBatchTimer->stop();
    if (videoBatchWorker) {
        videoBatchWorker->cancel();
        videoBatchWorker->wait();
    }
    videoProbeTimer->stop();
    videoProbePending = false;
    if (videoProbeWorker) {
        videoProbeWorker->cancel();
        videoProbeWorker->wait();
    }
    if (videoWorker != nullptr && videoWorker->isRunning()) {
        videoWorker->cancel();
        videoWorker->wait();
    }
    if (singleWorker != nullptr && singleWorker->isRunning()) {
        singleWorker->requestInterruption();
        singleWorker->wait();
    }
    if (batchWorker != nullptr && batchWorker->isRunning()) {
        batchWorker->cancel();
        batchWorker->wait();
    }
    if (benchmarkWorker != nullptr && benchmarkWorker->isRunning()) {
        benchmarkWorker->requestInterruption();
        benchmarkWorker->wait();
    }
    QMainWindow::closeEvent(event);
}
