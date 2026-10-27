#include "AnimatedButton.hpp"
#include "MultiDeviceComboBox.hpp"
#include "ThemePalette.hpp"
#include "WatermarkingUI.h"
#include <algorithm>
#include <CorrelationFormat.hpp>
#include <exception>
#include <filesystem>
#include <functional>
#include <limits>
#include <QAbstractItemView>
#include <QDesktopServices>
#include <QDir>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QItemSelectionModel>
#include <QMessageBox>
#include <QSignalBlocker>
#include <QTableWidgetItem>
#include <QUrl>
#include <QVBoxLayout>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include <VideoBatch.hpp>

/*!
 * \brief Multi-device video batch controls, queue progress and per-file results
 * \author Dimitris Karatzas
 */

namespace {
QString hardwareDefaults(const int device) {
    try {
        return QString::fromStdString(WatermarkCore::defaultVideoEncoderOptions(true, device));
    } catch (const std::exception&) { return {}; }
}

bool terminal(const WatermarkCore::VideoBatchState state) {
    using State = WatermarkCore::VideoBatchState;
    return state == State::Completed || state == State::Failed || state == State::Skipped || state == State::Cancelled;
}
} // namespace

void WatermarkingUI::createVideoBatchTab() {
    videoBatchPage = new QWidget(tabs);
    videoBatchPage->setObjectName("workflowPage");
    auto* layout = new QVBoxLayout(videoBatchPage);
    layout->setContentsMargins(18, 20, 18, 18);
    layout->setSpacing(12);
    videoBatchInputs = new QWidget(videoBatchPage);
    videoBatchInputs->setProperty("workflowContainer", true);
    auto* inputs = new QVBoxLayout(videoBatchInputs);
    inputs->setContentsMargins(0, 0, 0, 0);
    inputs->setSpacing(10);
    auto* form = new QFormLayout;
    form->setLabelAlignment(Qt::AlignRight);
    videoBatchMode = new QComboBox(videoBatchInputs);
    videoBatchMode->setObjectName("videoBatchMode");
    videoBatchMode->addItems({"Embed watermark", "Detect watermark"});
    form->addRow("Operation", videoBatchMode);
    auto* outputRow = new QWidget(videoBatchInputs);
    outputRow->setProperty("workflowContainer", true);
    auto* outputLayout = new QHBoxLayout(outputRow);
    outputLayout->setContentsMargins(0, 0, 0, 0);
    videoBatchOutput = new QLineEdit(outputRow);
    videoBatchOutput->setObjectName("videoBatchOutput");
    auto* browse = new AnimatedButton("Browse", outputRow);
    outputLayout->addWidget(videoBatchOutput, 1);
    outputLayout->addWidget(browse);
    form->addRow("Output folder", outputRow);
    videoBatchFormatRow = new QWidget(videoBatchInputs);
    videoBatchFormatRow->setProperty("workflowContainer", true);
    auto* formatLayout = new QHBoxLayout(videoBatchFormatRow);
    formatLayout->setContentsMargins(0, 0, 0, 0);
    formatLayout->addWidget(new QLabel("Output format", videoBatchFormatRow));
    videoBatchFormat = new QComboBox(videoBatchFormatRow);
    videoBatchFormat->setObjectName("videoBatchFormat");
    videoBatchFormat->addItem("Matroska (.mkv)", "mkv");
    videoBatchFormat->addItem("MP4 (.mp4)", "mp4");
    formatLayout->addWidget(videoBatchFormat);
    videoBatchDevices = new MultiDeviceComboBox(videoBatchInputs);
    videoBatchDevices->setObjectName("videoBatchDevices");
    const auto available = WatermarkCore::getAvailableDevices();
    for (int device = 0; device < static_cast<int>(available.size()); ++device)
        videoBatchDevices->addDevice(QString::fromStdString(available[device]), device, device == (deviceComboBox ? deviceComboBox->currentIndex() : 0));
    form->addRow("Devices", videoBatchDevices);
    videoBatchDevices->selectionChanged = [this] { updateVideoBatchSelection(); };
    videoBatchAcceleration = new QCheckBox("Accelerate final video", videoBatchInputs);
    videoBatchAcceleration->setObjectName("videoBatchAcceleration");
    videoBatchAcceleration->setChecked(true);
    videoBatchAcceleration->setToolTip(
        "When only one video remains, an idle selected GPU starts a second attempt. Only the first complete result is kept. Turn off if extra work slows CPU encoding or disk access.");
    videoBatchAcceleration->setVisible(available.size() > 1);
    formatLayout->addWidget(videoBatchAcceleration);
    formatLayout->addStretch();
    inputs->addLayout(form);
    inputs->addWidget(videoBatchFormatRow);
    inputs->addWidget(createSettingsRow(videoBatchSettings, videoBatchInputs));
    videoBatchSettings.password->setObjectName("videoBatchPassword");
    videoBatchSettings.p->setObjectName("videoBatchWindow");
    videoBatchSettings.psnr->setObjectName("videoBatchPsnr");
    auto* interval = new QHBoxLayout;
    interval->addWidget(new QLabel("Watermark every", videoBatchInputs));
    videoBatchInterval = new QSpinBox(videoBatchInputs);
    videoBatchInterval->setObjectName("videoBatchInterval");
    videoBatchInterval->setRange(1, std::numeric_limits<int>::max());
    videoBatchInterval->setValue(1);
    interval->addWidget(videoBatchInterval);
    interval->addWidget(new QLabel("frame(s)", videoBatchInputs));
    interval->addStretch();
    inputs->addLayout(interval);
    videoBatchEncoderArea = new QWidget(videoBatchInputs);
    videoBatchEncoderArea->setProperty("workflowContainer", true);
    auto* encoders = new QVBoxLayout(videoBatchEncoderArea);
    encoders->setContentsMargins(0, 0, 0, 0);
    videoBatchHardware = new QCheckBox("Hardware encoding", videoBatchEncoderArea);
    videoBatchHardware->setObjectName("videoBatchHardware");
    videoBatchHardware->setVisible(WatermarkCore::getBackendName() != "eigen");
    encoders->addWidget(videoBatchHardware);
    videoBatchEncoderDeviceRow = new QWidget(videoBatchEncoderArea);
    videoBatchEncoderDeviceRow->setProperty("workflowContainer", true);
    auto* encoderDevices = new QFormLayout(videoBatchEncoderDeviceRow);
    encoderDevices->setContentsMargins(0, 0, 0, 0);
    videoBatchEncoderDevice = new QComboBox(videoBatchEncoderDeviceRow);
    videoBatchEncoderDevice->setObjectName("videoBatchEncoderDevice");
    encoderDevices->addRow("Encoder options for", videoBatchEncoderDevice);
    encoders->addWidget(videoBatchEncoderDeviceRow);
    connect(videoBatchEncoderDevice, &QComboBox::currentIndexChanged, this, [this] {
        if (videoBatchEncoderDevice->currentIndex() >= 0)
            updateVideoBatchDevice(videoBatchEncoderDevice->currentData().toInt());
    });
    videoBatchSoftwareOptions = QString::fromStdString(WatermarkCore::defaultVideoEncoderOptions(false));
    videoBatchOptionsDevice = deviceComboBox ? deviceComboBox->currentIndex() : 0;
    videoBatchEncoderOptions = new QLineEdit(videoBatchSoftwareOptions, videoBatchEncoderArea);
    videoBatchEncoderOptions->setObjectName("videoBatchEncoderOptions");
    auto* options = new QFormLayout;
    options->addRow("FFmpeg encoder options", videoBatchEncoderOptions);
    encoders->addLayout(options);
    inputs->addWidget(videoBatchEncoderArea);
    auto* hint = new QLabel("One worker per selected device. Detection saves one CSV per video, existing files are kept.", videoBatchInputs);
    hint->setObjectName("mutedLabel");
    hint->setWordWrap(true);
    inputs->addWidget(hint);
    layout->addWidget(videoBatchInputs);
    auto* files = new QHBoxLayout;
    videoBatchAddFiles = new AnimatedButton("Add videos", videoBatchPage);
    videoBatchAddFiles->setObjectName("videoBatchAddFiles");
    videoBatchAddFolder = new AnimatedButton("Add folder", videoBatchPage);
    videoBatchRemove = new AnimatedButton("Remove selected", videoBatchPage);
    videoBatchClear = new AnimatedButton("Clear", videoBatchPage);
    files->addWidget(videoBatchAddFiles);
    files->addWidget(videoBatchAddFolder);
    files->addWidget(videoBatchRemove);
    files->addWidget(videoBatchClear);
    files->addStretch();
    layout->addLayout(files);
    auto* actions = new QHBoxLayout;
    videoBatchStart = new AnimatedButton("Start batch", videoBatchPage);
    videoBatchStart->setObjectName("primaryButton");
    videoBatchCancel = new AnimatedButton("Cancel batch", videoBatchPage);
    videoBatchSkip = new AnimatedButton("Skip current", videoBatchPage);
    auto* open = new AnimatedButton("Open output folder", videoBatchPage);
    videoBatchCancel->setEnabled(false);
    videoBatchSkip->setEnabled(false);
    actions->addWidget(videoBatchStart);
    actions->addWidget(videoBatchCancel);
    actions->addWidget(videoBatchSkip);
    actions->addStretch();
    actions->addWidget(open);
    layout->addLayout(actions);
    videoBatchStatus = new QLabel("Add videos or drop files and folders to build the queue.", videoBatchPage);
    videoBatchStatus->setObjectName("batchStatus");
    videoBatchStatus->setWordWrap(true);
    auto* statusRow = new QHBoxLayout;
    statusRow->addWidget(videoBatchStatus, 1);
    videoBatchElapsed = new QLabel("Elapsed: 0.0 s", videoBatchPage);
    videoBatchElapsed->setObjectName("videoBatchElapsed");
    statusRow->addWidget(videoBatchElapsed);
    layout->addLayout(statusRow);
    videoBatchProgress = new QProgressBar(videoBatchPage);
    videoBatchProgress->setRange(0, 1000);
    videoBatchProgress->setValue(0);
    layout->addWidget(videoBatchProgress);
    videoBatchTable = new QTableWidget(videoBatchPage);
    videoBatchTable->setObjectName("videoBatchQueue");
    videoBatchTable->setColumnCount(5);
    videoBatchTable->setHorizontalHeaderLabels({"File", "Status", "Progress", "Device", "Result"});
    videoBatchTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
    videoBatchTable->setSelectionBehavior(QAbstractItemView::SelectRows);
    videoBatchTable->setSelectionMode(QAbstractItemView::ExtendedSelection);
    videoBatchTable->setAlternatingRowColors(true);
    videoBatchTable->setWordWrap(false);
    videoBatchTable->verticalHeader()->hide();
    auto* header = videoBatchTable->horizontalHeader();
    header->setSectionResizeMode(0, QHeaderView::Stretch);
    header->setSectionResizeMode(1, QHeaderView::ResizeToContents);
    header->setSectionResizeMode(2, QHeaderView::Fixed);
    header->resizeSection(2, 120);
    header->resizeSection(3, 160);
    header->setSectionResizeMode(4, QHeaderView::Stretch);
    layout->addWidget(videoBatchTable, 1);
    auto* messages = new QCheckBox("Show processing messages", videoBatchPage);
    layout->addWidget(messages);
    videoBatchLog = new QPlainTextEdit(videoBatchPage);
    videoBatchLog->setReadOnly(true);
    videoBatchLog->setMaximumBlockCount(300);
    videoBatchLog->setMaximumHeight(110);
    videoBatchLog->hide();
    layout->addWidget(videoBatchLog);
    connect(messages, &QCheckBox::toggled, videoBatchLog, &QWidget::setVisible);
    tabs->insertTab(3, videoBatchPage, "Batch videos");
    videoBatchTimer = new QTimer(this);
    videoBatchTimer->setInterval(100);
    connect(videoBatchTimer, &QTimer::timeout, this, &WatermarkingUI::pollVideoBatch);
    connect(browse, &QPushButton::clicked, this, [this] {
        const QString path = QFileDialog::getExistingDirectory(this, "Choose batch output folder", videoBatchOutput->text());
        if (!path.isEmpty())
            videoBatchOutput->setText(path);
    });
    connect(videoBatchAddFiles, &QPushButton::clicked, this, [this] {
        addVideoBatchPaths(QFileDialog::getOpenFileNames(this, "Add videos", QString(), "Videos (*.mkv *.mp4 *.mov *.avi *.webm *.m4v *.ts *.mts *.m2ts *.mpg *.mpeg *.wmv);;All files (*)"));
    });
    connect(videoBatchAddFolder, &QPushButton::clicked, this, [this] {
        const QString path = QFileDialog::getExistingDirectory(this, "Add videos from a folder");
        if (!path.isEmpty())
            addVideoBatchPaths({path});
    });
    connect(videoBatchClear, &QPushButton::clicked, this, [this] {
        videoBatchFiles.clear();
        refreshVideoBatchTable();
    });
    connect(videoBatchRemove, &QPushButton::clicked, this, [this] {
        std::set<int, std::greater<int>> rows;
        for (const auto& index : videoBatchTable->selectionModel()->selectedRows())
            rows.insert(index.row());
        for (const int row : rows)
            videoBatchFiles.removeAt(row);
        refreshVideoBatchTable();
    });
    connect(videoBatchMode, &QComboBox::currentIndexChanged, this, [this] {
        updateVideoBatchControls();
        if (!busy)
            refreshVideoBatchTable();
    });
    connect(videoBatchHardware, &QCheckBox::toggled, this, [this](const bool hardware) {
        if (hardware) {
            videoBatchSoftwareOptions = videoBatchEncoderOptions->text();
            videoBatchEncoderOptions->setText(videoBatchHardwareOptions.value(videoBatchOptionsDevice, hardwareDefaults(videoBatchOptionsDevice)));
        } else {
            videoBatchHardwareOptions.insert(videoBatchOptionsDevice, videoBatchEncoderOptions->text());
            videoBatchEncoderOptions->setText(videoBatchSoftwareOptions);
        }
        updateVideoBatchControls();
    });
    connect(videoBatchStart, &QPushButton::clicked, this, &WatermarkingUI::startVideoBatch);
    connect(videoBatchCancel, &QPushButton::clicked, this, [this] {
        if (videoBatchWorker) {
            videoBatchWorker->cancel();
            videoBatchCancel->setEnabled(false);
            videoBatchSkip->setEnabled(false);
            videoBatchStatus->setText("Cancelling batch...");
        }
    });
    connect(videoBatchSkip, &QPushButton::clicked, this, [this] {
        if (!videoBatchWorker)
            return;
        const auto rows = videoBatchTable->selectionModel()->selectedRows();
        if (rows.isEmpty() && videoBatchActive && videoBatchWorker->skipCurrent(videoBatchActive))
            videoBatchSkipRequested.insert(videoBatchActive);
        for (const auto& row : rows) {
            const auto id = static_cast<std::uint64_t>(row.row() + 1);
            if (videoBatchWorker->skipCurrent(id))
                videoBatchSkipRequested.insert(id);
        }
        updateVideoBatchSkip();
    });
    connect(videoBatchTable->selectionModel(), &QItemSelectionModel::selectionChanged, this, [this] { updateVideoBatchSkip(); });
    connect(open, &QPushButton::clicked, this, [this] {
        const QString path = videoBatchOutput->text().trimmed();
        if (QFileInfo(path).isDir())
            QDesktopServices::openUrl(QUrl::fromLocalFile(path));
    });
    connect(videoSettings.password, &QLineEdit::textChanged, videoBatchSettings.password, &QLineEdit::setText);
    connect(videoBatchSettings.password, &QLineEdit::textChanged, videoSettings.password, &QLineEdit::setText);
    connect(videoSettings.p, &QComboBox::currentIndexChanged, videoBatchSettings.p, &QComboBox::setCurrentIndex);
    connect(videoBatchSettings.p, &QComboBox::currentIndexChanged, videoSettings.p, &QComboBox::setCurrentIndex);
    connect(videoSettings.psnr, &QDoubleSpinBox::valueChanged, videoBatchSettings.psnr, &QDoubleSpinBox::setValue);
    connect(videoBatchSettings.psnr, &QDoubleSpinBox::valueChanged, videoSettings.psnr, &QDoubleSpinBox::setValue);
    updateVideoBatchSelection();
    updateVideoBatchControls();
}

void WatermarkingUI::addVideoBatchPaths(const QStringList& paths) {
    if (busy)
        return;
    try {
        for (const auto& selected : paths) {
            QStringList files;
            const QFileInfo info(selected);
            if (info.isDir()) {
                for (const auto& file : WatermarkCore::getValidVideoFiles(std::filesystem::path(selected.toStdWString())))
                    files.push_back(QString::fromStdWString(file.wstring()));
            } else {
                files.push_back(info.absoluteFilePath());
            }
            for (const auto& file : files) {
                const QString canonical = QFileInfo(file).canonicalFilePath();
                const QString path = canonical.isEmpty() ? QFileInfo(file).absoluteFilePath() : canonical;
                if (!videoBatchFiles.contains(path, Qt::CaseInsensitive))
                    videoBatchFiles.push_back(path);
            }
        }
        if (videoBatchOutput->text().isEmpty() && !videoBatchFiles.isEmpty())
            videoBatchOutput->setText(QFileInfo(videoBatchFiles.front()).dir().filePath("watermark_output"));
        refreshVideoBatchTable();
    } catch (const std::exception& error) { QMessageBox::warning(this, "Video files", QString::fromUtf8(error.what())); }
}

void WatermarkingUI::refreshVideoBatchTable() {
    videoBatchJobs.clear();
    videoBatchTable->setRowCount(0);
    for (int row = 0; row < videoBatchFiles.size(); ++row) {
        videoBatchTable->insertRow(row);
        auto* file = new QTableWidgetItem(QFileInfo(videoBatchFiles[row]).fileName());
        file->setToolTip(videoBatchFiles[row]);
        videoBatchTable->setItem(row, 0, file);
        auto* state = new QTableWidgetItem("Pending");
        state->setData(Qt::UserRole, static_cast<int>(WatermarkCore::VideoBatchState::Pending));
        videoBatchTable->setItem(row, 1, state);
        auto* progress = new QProgressBar(videoBatchTable);
        progress->setRange(0, 1000);
        progress->setValue(0);
        videoBatchTable->setCellWidget(row, 2, progress);
        videoBatchTable->setItem(row, 3, new QTableWidgetItem);
        videoBatchTable->setItem(row, 4, new QTableWidgetItem);
    }
    videoBatchStatus->setText(QString("%1 videos queued").arg(videoBatchFiles.size()));
    videoBatchProgress->setValue(0);
    videoBatchElapsed->setText("Elapsed: 0.0 s");
    videoBatchSkip->setText(videoBatchDevices->selectedDevices().size() > 1 ? "Skip selected" : "Skip current");
    videoBatchSkip->setToolTip("Skip selected running rows, or the first active file when no row is selected.");
    videoBatchStart->setEnabled(!busy && !videoBatchFiles.isEmpty() && !videoBatchDevices->selectedDevices().empty());
}

void WatermarkingUI::updateVideoBatchControls() {
    const bool detect = videoBatchMode->currentIndex() == 1;
    videoBatchSettings.psnrColumn->setVisible(!detect);
    videoBatchEncoderArea->setVisible(!detect);
    videoBatchFormat->setVisible(!detect);
    videoBatchFormatRow->findChild<QLabel*>()->setVisible(!detect);
    videoBatchEncoderDeviceRow->setVisible(videoBatchHardware->isChecked() && videoBatchDevices->selectedDevices().size() > 1);
    videoBatchAcceleration->setEnabled(videoBatchDevices->selectedDevices().size() > 1);
    videoBatchStart->setEnabled(!busy && !videoBatchFiles.isEmpty() && !videoBatchDevices->selectedDevices().empty());
}

void WatermarkingUI::updateVideoBatchDevice(const int device) {
    if (videoBatchHardware->isChecked())
        videoBatchHardwareOptions.insert(videoBatchOptionsDevice, videoBatchEncoderOptions->text());
    videoBatchOptionsDevice = device;
    if (videoBatchHardware->isChecked())
        videoBatchEncoderOptions->setText(videoBatchHardwareOptions.value(device, hardwareDefaults(device)));
}

void WatermarkingUI::updateVideoBatchSelection() {
    if (videoBatchHardware->isChecked())
        videoBatchHardwareOptions.insert(videoBatchOptionsDevice, videoBatchEncoderOptions->text());
    const auto devices = videoBatchDevices->selectedDevices();
    {
        // Rebuilding the picker must not save encoder text against the wrong GPU
        const QSignalBlocker block(videoBatchEncoderDevice);
        videoBatchEncoderDevice->clear();
        for (const int device : devices)
            videoBatchEncoderDevice->addItem(videoBatchDevices->itemText(device), device);
        const int previous = videoBatchEncoderDevice->findData(videoBatchOptionsDevice);
        videoBatchEncoderDevice->setCurrentIndex(previous >= 0 ? previous : (devices.empty() ? -1 : 0));
    }
    if (!devices.empty())
        updateVideoBatchDevice(videoBatchEncoderDevice->currentData().toInt());
    videoBatchSkip->setText(devices.size() > 1 ? "Skip selected" : "Skip current");
    updateVideoBatchControls();
}

void WatermarkingUI::startVideoBatch() {
    if (busy)
        return;
    if (videoBatchFiles.isEmpty() || videoBatchSettings.password->text().isEmpty() || videoBatchOutput->text().trimmed().isEmpty()) {
        QMessageBox::warning(this, "Missing input", "Add videos, choose an output folder and enter a watermark password.");
        return;
    }
    const bool detect = videoBatchMode->currentIndex() == 1;
    WatermarkCore::VideoBatchSettings settings;
    settings.deviceIndex = videoBatchOptionsDevice;
    settings.detect = detect;
    settings.deviceIndices = videoBatchDevices->selectedDevices();
    if (settings.deviceIndices.empty()) {
        QMessageBox::warning(this, "Batch devices", "Select at least one batch device.");
        return;
    }
    settings.accelerateFinal = videoBatchAcceleration->isChecked();
    settings.video.watermarkPassword = videoBatchSettings.password->text().toStdString();
    settings.video.p = videoBatchSettings.p->currentData().toInt();
    settings.video.psnr = static_cast<float>(videoBatchSettings.psnr->value());
    settings.video.watermarkInterval = videoBatchInterval->value();
    settings.video.useHwDecoder = true;
    settings.video.useHwEncoder = !detect && videoBatchHardware->isChecked();
    settings.video.encodeOptions = videoBatchEncoderOptions->text().toStdString();
    if (settings.video.useHwEncoder) {
        videoBatchHardwareOptions.insert(videoBatchOptionsDevice, videoBatchEncoderOptions->text());
        for (const int device : settings.deviceIndices)
            settings.hardwareOptions.emplace(device, videoBatchHardwareOptions.value(device, hardwareDefaults(device)).toStdString());
        settings.video.encodeOptions.clear();
    }
    try {
        if (!detect)
            for (const auto& [device, options] : settings.hardwareOptions)
                WatermarkCore::validateVideoEncoderOptions(options, true);
        if (!detect && !settings.video.useHwEncoder)
            WatermarkCore::validateVideoEncoderOptions(settings.video.encodeOptions, false);
        std::vector<std::string> inputs;
        for (const auto& path : videoBatchFiles)
            inputs.push_back(path.toStdString());
        auto jobs = WatermarkCore::prepareVideoBatch(inputs, videoBatchOutput->text().trimmed().toStdString(), detect, videoBatchFormat->currentData().toString().toStdString());
        videoBatchFiles.clear();
        for (const auto& job : jobs)
            videoBatchFiles.push_back(QString::fromStdString(job.inputPath));
        refreshVideoBatchTable();
        videoBatchJobs = std::move(jobs);
    } catch (const std::exception& error) {
        QMessageBox::warning(this, "Video batch", QString::fromUtf8(error.what()));
        return;
    }
    invalidateSingleResult();
    delete videoBatchWorker;
    auto* worker = new VideoBatchWorker(std::move(settings), videoBatchJobs, this);
    videoBatchWorker = worker;
    const auto run = ++videoBatchRun;
    videoBatchActive = 0;
    videoBatchSkipRequested.clear();
    videoBatchFinished = 0;
    videoBatchLog->clear();
    videoBatchStatus->setText("Starting batch...");
    setBusy(true);
    connect(worker, &QThread::finished, this, [this, worker, run] {
        if (closing || videoBatchWorker != worker || run != videoBatchRun)
            return;
        videoBatchTimer->stop();
        worker->wait();
        pollVideoBatch();
        const double seconds = worker->error().isEmpty() ? worker->summary().elapsedSeconds : videoBatchClock.elapsed() / 1000.0;
        videoBatchElapsed->setText(QString("Elapsed: %1 s").arg(seconds, 0, 'f', 1));
        if (!worker->error().isEmpty()) {
            for (int row = 0; row < videoBatchTable->rowCount(); ++row) {
                auto* state = videoBatchTable->item(row, 1);
                if (!terminal(static_cast<WatermarkCore::VideoBatchState>(state->data(Qt::UserRole).toInt()))) {
                    state->setText("Failed");
                    state->setForeground(themePalette().error);
                    state->setData(Qt::UserRole, static_cast<int>(WatermarkCore::VideoBatchState::Failed));
                    videoBatchTable->item(row, 4)->setText(worker->error());
                    static_cast<QProgressBar*>(videoBatchTable->cellWidget(row, 2))->setRange(0, 1000);
                }
            }
            videoBatchStatus->setText(worker->error());
        } else {
            const auto& result = worker->summary();
            videoBatchStatus->setText(QString("%1 done, %2 failed, %3 skipped, %4 cancelled  |  %5 s")
                    .arg(result.succeeded)
                    .arg(result.failed)
                    .arg(result.skipped)
                    .arg(result.cancelled)
                    .arg(result.elapsedSeconds, 0, 'f', 1));
            if (!result.cancelled)
                videoBatchProgress->setValue(1000);
        }
        videoBatchActive = 0;
        setBusy(false);
    });
    videoBatchClock.start();
    videoBatchTimer->start();
    worker->start();
    videoBatchCancel->setEnabled(true);
}

void WatermarkingUI::updateVideoBatchSkip() {
    bool enabled = false;
    if (busy && videoBatchWorker && videoBatchWorker->isRunning() && !videoBatchWorker->cancellationRequested()) {
        const auto rows = videoBatchTable->selectionModel()->selectedRows();
        if (rows.isEmpty())
            enabled = videoBatchActive && !videoBatchSkipRequested.contains(videoBatchActive);
        for (const auto& row : rows) {
            const auto id = static_cast<std::uint64_t>(row.row() + 1);
            const auto* item = videoBatchTable->item(row.row(), 1);
            if (item && !videoBatchSkipRequested.contains(id)) {
                const auto state = static_cast<WatermarkCore::VideoBatchState>(item->data(Qt::UserRole).toInt());
                enabled = enabled || state == WatermarkCore::VideoBatchState::Probing || state == WatermarkCore::VideoBatchState::Processing;
            }
        }
    }
    videoBatchSkip->setEnabled(enabled);
}

void WatermarkingUI::pollVideoBatch() {
    if (!videoBatchWorker)
        return;
    if (videoBatchWorker->isRunning())
        videoBatchElapsed->setText(QString("Elapsed: %1 s").arg(videoBatchClock.elapsed() / 1000.0, 0, 'f', 1));
    auto update = videoBatchWorker->takeUpdate();
    for (const auto& job : update.jobs) {
        if (!job.id || job.id > static_cast<std::uint64_t>(videoBatchTable->rowCount()))
            continue;
        const int row = static_cast<int>(job.id - 1);
        auto* state = videoBatchTable->item(row, 1);
        if (job.revision <= state->data(Qt::UserRole + 1).toULongLong())
            continue;
        const auto previous = static_cast<WatermarkCore::VideoBatchState>(state->data(Qt::UserRole).toInt());
        state->setText(job.activeDevices.size() > 1 && !terminal(job.state) ? "Parallel attempts" : WatermarkCore::videoBatchStateName(job.state));
        state->setData(Qt::UserRole, static_cast<int>(job.state));
        state->setData(Qt::UserRole + 1, QVariant::fromValue<qulonglong>(job.revision));
        const auto& palette = themePalette();
        state->setForeground(job.state == WatermarkCore::VideoBatchState::Failed      ? palette.error
                             : job.state == WatermarkCore::VideoBatchState::Completed ? palette.accent
                             : job.state == WatermarkCore::VideoBatchState::Skipped   ? palette.warning
                             : job.state == WatermarkCore::VideoBatchState::Cancelled ? palette.muted
                                                                                      : palette.text);
        if (videoBatchSkipRequested.contains(job.id) && !terminal(job.state))
            state->setText("Skipping");
        state->setToolTip(QString("Attempt %1").arg(job.attempt));
        auto* progress = static_cast<QProgressBar*>(videoBatchTable->cellWidget(row, 2));
        progress->setProperty("fps", job.progress.fps);
        if (!terminal(job.state) && job.progress.fraction < 0)
            progress->setRange(0, 0);
        else {
            progress->setRange(0, 1000);
            progress->setValue(job.state == WatermarkCore::VideoBatchState::Completed ? 1000 : static_cast<int>(std::max(0.0, job.progress.fraction) * 1000));
        }
        if (!terminal(previous) && terminal(job.state) && job.state != WatermarkCore::VideoBatchState::Cancelled)
            ++videoBatchFinished;
        QStringList devices;
        const auto indices = terminal(job.state) ? std::vector<int>{job.deviceIndex} : job.activeDevices;
        for (const int device : indices)
            if (device >= 0 && device < videoBatchDevices->count())
                devices.push_back(videoBatchDevices->itemText(device));
        auto* device = videoBatchTable->item(row, 3);
        device->setText(devices.join(" + "));
        device->setToolTip(device->text());
        QString detail = QString::fromStdString(job.detail);
        if (job.detections)
            detail = QString("%1 samples, mean %2  |  %3").arg(job.detections).arg(QString::fromStdString(WatermarkCore::formatCorrelationPercent(job.meanCorrelation))).arg(detail);
        videoBatchTable->item(row, 4)->setText(detail);
        videoBatchTable->item(row, 4)->setToolTip(detail);
    }
    double activeFraction = 0.0, fps = 0.0;
    int active = 0;
    videoBatchActive = 0;
    for (int row = 0; row < videoBatchTable->rowCount(); ++row) {
        const auto state = static_cast<WatermarkCore::VideoBatchState>(videoBatchTable->item(row, 1)->data(Qt::UserRole).toInt());
        if (state == WatermarkCore::VideoBatchState::Probing || state == WatermarkCore::VideoBatchState::Processing) {
            if (!videoBatchActive)
                videoBatchActive = static_cast<std::uint64_t>(row + 1);
            ++active;
            const auto* progress = static_cast<QProgressBar*>(videoBatchTable->cellWidget(row, 2));
            activeFraction += progress->maximum() ? progress->value() / 1000.0 : 0.0;
            fps += progress->property("fps").toDouble();
        }
    }
    for (const auto& log : update.logs)
        videoBatchLog->appendPlainText(log);
    if (videoBatchWorker->isRunning())
        videoBatchStatus->setText(videoBatchWorker->cancellationRequested()
                                      ? "Cancelling batch..."
                                      : QString("%1 of %2 videos finished  |  %3 active  |  %4 FPS").arg(videoBatchFinished).arg(videoBatchJobs.size()).arg(active).arg(fps, 0, 'f', 1));
    if (!videoBatchJobs.empty()) {
        const int progress = static_cast<int>(1000.0 * (videoBatchFinished + activeFraction) / videoBatchJobs.size());
        // retries may lower file progress, keep the batch bar from moving backwards
        videoBatchProgress->setValue(std::max(videoBatchProgress->value(), progress));
    }
    updateVideoBatchSkip();
}
