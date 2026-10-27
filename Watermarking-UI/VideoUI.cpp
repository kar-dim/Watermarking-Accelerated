#include "AnimatedButton.hpp"
#include "WatermarkingUI.h"
#include <algorithm>
#include <climits>
#include <CorrelationFormat.hpp>
#include <exception>
#include <functional>
#include <QAbstractItemView>
#include <QChar>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QMessageBox>
#include <QPixmap>
#include <QSize>
#include <QSizePolicy>
#include <QStyle>
#include <QTableWidgetItem>
#include <QTransform>
#include <QVBoxLayout>
#include <utility>
#include <WatermarkCore.hpp>

/*!
 * \brief Single-video controls, processing preview and progress display
 * \author Dimitris Karatzas
 */
namespace {
constexpr auto videoFilter = "Videos (*.mkv *.mp4 *.mov *.avi *.webm *.m4v *.ts *.mts *.m2ts *.mpg *.mpeg *.wmv);;All files (*)";
constexpr auto outputVideoFilter = "Matroska (*.mkv);;MP4 (*.mp4)";

QWidget* fileRow(QLineEdit*& input, QWidget* parent, const std::function<void()>& browse) {
    auto* row = new QWidget(parent);
    row->setProperty("workflowContainer", true);
    auto* layout = new QHBoxLayout(row);
    layout->setContentsMargins(0, 0, 0, 0);
    input = new QLineEdit(row);
    auto* button = new AnimatedButton("Browse", row);
    button->setObjectName("secondaryButton");
    layout->addWidget(input, 1);
    layout->addWidget(button);
    QObject::connect(button, &QPushButton::clicked, row, browse);
    return row;
}

QString durationText(const double seconds) {
    if (seconds < 0.0)
        return "unknown";
    const auto total = static_cast<qint64>(seconds);
    return QString("%1:%2:%3").arg(total / 3600, 2, 10, QChar('0')).arg(total / 60 % 60, 2, 10, QChar('0')).arg(total % 60, 2, 10, QChar('0'));
}
} // namespace

void WatermarkingUI::createVideoTab() {
    videoSoftwareOptions = QString::fromStdString(WatermarkCore::defaultVideoEncoderOptions(false));
    videoOptionsDevice = deviceComboBox ? deviceComboBox->currentIndex() : 0;
    videoPage = new QWidget(tabs);
    videoPage->setObjectName("workflowPage");
    auto* layout = new QVBoxLayout(videoPage);
    layout->setContentsMargins(18, 20, 18, 18);
    layout->setSpacing(12);
    videoInputs = new QWidget(videoPage);
    videoInputs->setProperty("workflowContainer", true);
    auto* inputs = new QVBoxLayout(videoInputs);
    inputs->setContentsMargins(0, 0, 0, 0);
    inputs->setSpacing(12);
    auto* files = new QFormLayout;
    files->setLabelAlignment(Qt::AlignRight);
    videoMode = new QComboBox(videoInputs);
    videoMode->setObjectName("videoMode");
    videoMode->addItems({"Embed watermark", "Detect watermark"});
    files->addRow("Operation", videoMode);
    files->addRow("Input video", fileRow(videoInput, videoInputs, [this] {
        const QString path = QFileDialog::getOpenFileName(this, "Choose a video", videoInput->text(), videoFilter);
        if (!path.isEmpty())
            selectVideo(path);
    }));
    videoInput->setObjectName("videoInput");
    videoOutputRow = fileRow(videoOutput, videoInputs, [this] {
        const QString path = QFileDialog::getSaveFileName(this, "Choose output video", videoOutput->text(), outputVideoFilter);
        if (!path.isEmpty())
            videoOutput->setText(path);
    });
    videoOutput->setObjectName("videoOutput");
    files->addRow("Output video", videoOutputRow);
    inputs->addLayout(files);
    inputs->addWidget(createSettingsRow(videoSettings, videoInputs));
    videoSettings.password->setObjectName("videoPassword");
    videoSettings.p->setObjectName("videoWindow");
    videoSettings.psnr->setObjectName("videoPsnr");
    auto* intervalRow = new QHBoxLayout;
    intervalRow->addWidget(new QLabel("Watermark every", videoInputs));
    videoInterval = new QSpinBox(videoInputs);
    videoInterval->setObjectName("videoInterval");
    videoInterval->setRange(1, INT_MAX);
    videoInterval->setValue(1);
    intervalRow->addWidget(videoInterval);
    intervalRow->addWidget(new QLabel("frame(s)", videoInputs));
    intervalRow->addStretch();
    inputs->addLayout(intervalRow);
    videoEncoderArea = new QWidget(videoInputs);
    videoEncoderArea->setProperty("workflowContainer", true);
    auto* encoders = new QVBoxLayout(videoEncoderArea);
    encoders->setContentsMargins(0, 0, 0, 0);
    encoders->setSpacing(6);
    videoHardware = new QCheckBox("Hardware encoding", videoEncoderArea);
    videoHardware->setObjectName("videoHardware");
    videoHardware->setVisible(WatermarkCore::getBackendName() != "eigen");
    encoders->addWidget(videoHardware);
    videoEncoderOptions = new QLineEdit(videoEncoderArea);
    videoEncoderOptions->setObjectName("videoEncoderOptions");
    videoEncoderOptions->setText(videoSoftwareOptions);
    auto* options = new QFormLayout;
    options->setLabelAlignment(Qt::AlignRight);
    options->addRow("FFmpeg encoder options", videoEncoderOptions);
    encoders->addLayout(options);
    auto* decodeHint =
        new QLabel(WatermarkCore::getBackendName() == "eigen" ? "Decoding and encoding use the CPU." : "Hardware decoding is automatic, with CPU fallback when needed.", videoEncoderArea);
    decodeHint->setObjectName("mutedLabel");
    decodeHint->setWordWrap(true);
    encoders->addWidget(decodeHint);
    inputs->addWidget(videoEncoderArea);
    layout->addWidget(videoInputs);

    auto* actions = new QHBoxLayout;
    videoStart = new AnimatedButton("Embed video", videoPage);
    videoStart->setObjectName("primaryButton");
    videoCancel = new AnimatedButton("Cancel", videoPage);
    videoCancel->setObjectName("secondaryButton");
    videoCancel->setEnabled(false);
    videoExport = new AnimatedButton("Export results", videoPage);
    videoExport->setObjectName("secondaryButton");
    videoExport->hide();
    videoExport->setEnabled(false);
    actions->addWidget(videoStart);
    actions->addWidget(videoCancel);
    actions->addWidget(videoExport);
    actions->addStretch();
    videoPreviewEnabled = new QCheckBox("Live preview", videoPage);
    videoPreviewEnabled->setChecked(true);
    actions->addWidget(videoPreviewEnabled);
    layout->addLayout(actions);
    videoInfo = new QLabel("Choose a video to embed or detect a watermark.", videoPage);
    videoInfo->setObjectName("mutedLabel");
    videoInfo->setWordWrap(true);
    videoInfo->setTextInteractionFlags(Qt::TextSelectableByMouse);
    layout->addWidget(videoInfo);
    videoProgress = new QProgressBar(videoPage);
    videoProgress->setValue(0);
    layout->addWidget(videoProgress);
    videoStatus = new QLabel("Ready", videoPage);
    videoStatus->setObjectName("videoStatus");
    videoStatus->setWordWrap(true);
    layout->addWidget(videoStatus);
    videoPreview = new QLabel("Processing preview will appear here", videoPage);
    videoPreview->setObjectName("imageCanvas");
    videoPreview->setAlignment(Qt::AlignCenter);
    // A refreshed pixmap must not keep raising the label's requested size
    videoPreview->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
    videoPreview->setMinimumSize(400, 180);
    layout->addWidget(videoPreview, 1);
    videoDetectionSummary = new QLabel(videoPage);
    videoDetectionSummary->setObjectName("mutedLabel");
    videoDetectionSummary->hide();
    layout->addWidget(videoDetectionSummary);
    videoResults = new QTableWidget(videoPage);
    videoResults->setObjectName("videoResults");
    videoResults->setColumnCount(3);
    videoResults->setHorizontalHeaderLabels({"Frame", "Time", "Correlation"});
    videoResults->setEditTriggers(QAbstractItemView::NoEditTriggers);
    videoResults->setSelectionBehavior(QAbstractItemView::SelectRows);
    videoResults->setAlternatingRowColors(true);
    videoResults->verticalHeader()->hide();
    videoResults->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    videoResults->hide();
    layout->addWidget(videoResults);
    auto* logToggle = new QCheckBox("Show processing messages", videoPage);
    layout->addWidget(logToggle);
    videoLog = new QPlainTextEdit(videoPage);
    videoLog->setReadOnly(true);
    videoLog->setMaximumBlockCount(300);
    videoLog->setMaximumHeight(110);
    videoLog->hide();
    layout->addWidget(videoLog);
    layout->addStretch();
    connect(logToggle, &QCheckBox::toggled, videoLog, &QWidget::setVisible);
    connect(videoPreviewEnabled, &QCheckBox::toggled, this, &WatermarkingUI::updateVideoPreviewLayout);
    tabs->insertTab(1, videoPage, "Single video");
    videoTimer = new QTimer(this);
    videoTimer->setInterval(100);
    connect(videoTimer, &QTimer::timeout, this, &WatermarkingUI::pollVideo);
    videoProbeTimer = new QTimer(this);
    videoProbeTimer->setSingleShot(true);
    connect(videoProbeTimer, &QTimer::timeout, this, &WatermarkingUI::probeVideoInput);
    connect(videoInput, &QLineEdit::textChanged, this, [this] {
        if (busy)
            return;
        if (videoProbeWorker)
            videoProbeWorker->cancel();
        videoProbePending = true;
        videoProbeTimer->start(250);
        videoInfo->setText(videoInput->text().trimmed().isEmpty() ? "Choose a video to embed or detect a watermark." : "Reading source details...");
    });
    connect(videoMode, &QComboBox::currentIndexChanged, this, [this, files](const int mode) {
        files->setRowVisible(videoOutputRow, mode == 0);
        updateVideoControls();
    });
    connect(videoHardware, &QCheckBox::toggled, this, [this](const bool hardware) {
        if (hardware) {
            videoSoftwareOptions = videoEncoderOptions->text();
            videoEncoderOptions->setText(videoHardwareOptions.value(videoOptionsDevice, videoHardwareDefaults()));
        } else {
            videoHardwareOptions.insert(videoOptionsDevice, videoEncoderOptions->text());
            videoEncoderOptions->setText(videoSoftwareOptions);
        }
        updateVideoControls();
    });
    if (deviceComboBox)
        connect(deviceComboBox, &QComboBox::currentIndexChanged, this, &WatermarkingUI::updateVideoDevice);
    connect(videoStart, &QPushButton::clicked, this, &WatermarkingUI::startVideo);
    connect(videoCancel, &QPushButton::clicked, this, [this] {
        if (videoWorker) {
            videoWorker->cancel();
            videoCancel->setEnabled(false);
            videoStatus->setText("Cancelling...");
        }
    });
    connect(videoExport, &QPushButton::clicked, this, [this] {
        const QString path = QFileDialog::getSaveFileName(this, "Export detection results", "video_detection.csv", "CSV (*.csv)");
        if (!path.isEmpty() && videoWorker && !videoWorker->exportResults(path))
            QMessageBox::critical(this, "Export failed", "Could not save the detection results.");
    });
    connect(videoInput, &QLineEdit::editingFinished, this, [this] {
        if (videoOutput->text().isEmpty())
            selectVideo(videoInput->text().trimmed());
    });
    updateVideoControls();
}

void WatermarkingUI::selectVideo(const QString& path) {
    videoInput->setText(path);
    const QFileInfo input(path);
    videoOutput->setText(input.absolutePath() + "/" + input.completeBaseName() + "_watermarked.mkv");
}

void WatermarkingUI::probeVideoInput() {
    if (busy) {
        return;
    }
    if (videoProbeWorker)
        return;
    videoProbePending = false;
    const QString path = videoInput->text().trimmed();
    if (path.isEmpty())
        return;
    if (!QFileInfo(path).isFile()) {
        videoInfo->setText("Choose an existing video file.");
        return;
    }
    auto* worker = new VideoProbeWorker(path, this);
    videoProbeWorker = worker;
    connect(worker, &QThread::finished, this, [this, worker] {
        worker->wait();
        // The input may have changed while this probe was running
        if (!worker->cancelled() && videoInput->text().trimmed() == worker->path()) {
            if (!worker->error().isEmpty()) {
                videoInfo->setText(worker->error());
            } else {
                const auto& source = worker->result();
                const QString depth = source.bitDepth > 0 ? QString("%1-bit").arg(source.bitDepth) : "unknown depth";
                videoInfo->setText(QString("%1 x %2  |  %3  |  %4 %5  |  %6  |  %7 FPS  |  Duration %8\nAudio tracks: %9  |  Subtitle tracks: %10  |  Attachments: %11  |  Chapters: %12")
                        .arg(source.width)
                        .arg(source.height)
                        .arg(QString::fromStdString(source.codec))
                        .arg(QString::fromStdString(source.pixelFormat))
                        .arg(QString::fromStdString(source.dynamicRange))
                        .arg(depth)
                        .arg(source.framesPerSecond > 0 ? QString::number(source.framesPerSecond, 'f', 3) : "unknown")
                        .arg(durationText(source.durationSeconds))
                        .arg(source.audioTracks)
                        .arg(source.subtitleTracks)
                        .arg(source.attachments)
                        .arg(source.chapters));
            }
        }
        videoProbeWorker = nullptr;
        worker->deleteLater();
        if (videoProbePending && !busy && !videoProbeTimer->isActive())
            videoProbeTimer->start(0);
    });
    worker->start();
}

void WatermarkingUI::updateVideoControls() {
    const bool detect = videoMode->currentIndex() == 1;
    videoSettings.psnrColumn->setVisible(!detect);
    videoEncoderArea->setVisible(!detect);
    videoExport->setVisible(detect);
    videoResults->setVisible(detect);
    videoDetectionSummary->setVisible(detect);
    videoStart->setText(detect ? "Detect watermark" : "Embed video");
    const QString defaults = videoHardware->isChecked() ? videoHardwareDefaults() : QString::fromStdString(WatermarkCore::defaultVideoEncoderOptions(false));
    videoEncoderOptions->setPlaceholderText(defaults.isEmpty() ? "Enter hardware encoder options for the selected GPU" : defaults);
    updateVideoPreviewLayout();
}

void WatermarkingUI::updateVideoPreviewLayout() {
    const bool preview = videoPreviewEnabled->isChecked();
    const bool detect = videoMode->currentIndex() == 1;
    videoPreview->setVisible(preview);
    videoResults->setMaximumHeight(preview ? 165 : QWIDGETSIZE_MAX);
    auto* layout = static_cast<QVBoxLayout*>(videoPage->layout());
    layout->setStretchFactor(videoPreview, preview ? 1 : 0);
    layout->setStretchFactor(videoResults, detect && !preview ? 1 : 0);
    layout->setStretch(layout->count() - 1, !detect && !preview ? 1 : 0);
    if (videoWorker) {
        const QSize size = videoPreview->contentsRect().size();
        videoWorker->setPreview(preview && videoPage->isVisible() && videoPreview->isVisible(), size.width(), size.height());
    }
    if (preview)
        updateImageViews();
}

QString WatermarkingUI::videoHardwareDefaults() const {
    try {
        return QString::fromStdString(WatermarkCore::defaultVideoEncoderOptions(true, videoOptionsDevice));
    } catch (const std::exception&) { return {}; }
}

void WatermarkingUI::updateVideoDevice(const int device) {
    if (videoHardware->isChecked())
        videoHardwareOptions.insert(videoOptionsDevice, videoEncoderOptions->text());
    videoOptionsDevice = device;
    if (videoHardware->isChecked())
        videoEncoderOptions->setText(videoHardwareOptions.value(device, videoHardwareDefaults()));
    updateVideoControls();
}

void WatermarkingUI::startVideo() {
    if (busy)
        return;
    const bool detect = videoMode->currentIndex() == 1;
    if (!QFileInfo(videoInput->text().trimmed()).isFile() || videoSettings.password->text().isEmpty() || (!detect && videoOutput->text().trimmed().isEmpty())) {
        QMessageBox::warning(this, "Missing input", "Choose an input video and enter a watermark password. Embedding also needs an output video path.");
        return;
    }
    if (!detect) {
        try {
            WatermarkCore::validateVideoOutputPath(videoOutput->text().trimmed().toStdString());
            WatermarkCore::validateVideoEncoderOptions(videoEncoderOptions->text().toStdString(), videoHardware->isChecked());
        } catch (const std::exception& error) {
            QMessageBox::warning(this, "Video settings", QString::fromUtf8(error.what()));
            return;
        }
        if (QFileInfo::exists(videoOutput->text().trimmed()) &&
            QMessageBox::question(this, "Replace output", "The output file already exists. Replace it after processing completes?") != QMessageBox::Yes)
            return;
    }
    invalidateSingleResult();
    videoProbeTimer->stop();
    videoProbePending = false;
    if (videoProbeWorker)
        videoProbeWorker->cancel();
    delete videoWorker;
    videoWorker = nullptr;
    WatermarkCore::VideoSettings settings{};
    settings.videoFile = videoInput->text().trimmed().toStdString();
    settings.encodeOutputPath = videoOutput->text().trimmed().toStdString();
    settings.watermarkPassword = videoSettings.password->text().toStdString();
    settings.p = videoSettings.p->currentData().toInt();
    settings.psnr = static_cast<float>(videoSettings.psnr->value());
    settings.watermarkInterval = videoInterval->value();
    settings.useHwDecoder = true;
    settings.useHwEncoder = !detect && videoHardware->isChecked();
    settings.encodeOptions = videoEncoderOptions->text().toStdString();
    videoWorker = new VideoWorker(std::move(settings), deviceComboBox ? deviceComboBox->currentIndex() : 0, detect, this);
    videoLog->clear();
    videoResults->setRowCount(0);
    videoDetectionSummary->clear();
    videoPreviewImage = {};
    videoPreview->setText("Waiting for a processing preview");
    videoInfo->setText("Opening video...");
    videoStatus->setText("Setting up...");
    videoStatus->setProperty("state", "running");
    videoStatus->style()->unpolish(videoStatus);
    videoStatus->style()->polish(videoStatus);
    videoProgress->setRange(0, 0);
    videoExport->setEnabled(false);
    setBusy(true);
    connect(videoWorker, &QThread::finished, this, [this] {
        if (closing)
            return;
        videoTimer->stop();
        pollVideo();
        videoWorker->wait();
        videoProgress->setRange(0, 1000);
        const bool failed = !videoWorker->error().isEmpty();
        const bool cancelled = videoWorker->cancelled();
        if (!failed && !cancelled)
            videoProgress->setValue(1000);
        videoStatus->setText(failed ? videoWorker->error() : cancelled ? "Cancelled" : videoStatus->text());
        videoStatus->setProperty("state", failed ? "error" : cancelled ? "idle" : "done");
        videoStatus->style()->unpolish(videoStatus);
        videoStatus->style()->polish(videoStatus);
        videoExport->setEnabled(videoResults->rowCount() > 0);
        setBusy(false);
        if (failed)
            QMessageBox::critical(this, "Video processing failed", videoWorker->error());
    });
    pollVideo();
    videoTimer->start();
    videoWorker->start();
    videoCancel->setEnabled(true);
}

void WatermarkingUI::pollVideo() {
    if (!videoWorker)
        return;
    const QSize previewSize = videoPreview->contentsRect().size();
    videoWorker->setPreview(videoPreviewEnabled->isChecked() && videoPage->isVisible() && videoPreview->isVisible(), previewSize.width(), previewSize.height());
    auto update = videoWorker->takeUpdate();
    const auto& progress = update.progress;
    if (!progress.source.empty())
        videoInfo->setText(QString::fromStdString(progress.source + "\n" + progress.pipeline));
    if (progress.fraction >= 0.0) {
        videoProgress->setRange(0, 1000);
        videoProgress->setValue(static_cast<int>(progress.fraction * 1000));
    }
    const QString stage = progress.state == WatermarkCore::VideoState::Completed    ? "Complete"
                          : videoWorker->cancellationRequested()                    ? "Cancelling"
                          : progress.state == WatermarkCore::VideoState::Setup      ? "Setting up"
                          : progress.state == WatermarkCore::VideoState::Finalizing ? "Finalizing"
                                                                                    : "Processing";
    videoStatus->setText(QString("%1  |  %2 frames  |  %3 FPS  |  Elapsed %4  |  ETA %5")
            .arg(stage)
            .arg(progress.frames)
            .arg(progress.fps, 0, 'f', 1)
            .arg(durationText(progress.elapsedSeconds))
            .arg(durationText(progress.remainingSeconds)));
    if (!update.preview.isNull()) {
        const auto& matrix = progress.displayTransform;
        videoPreviewImage = update.preview.transformed(QTransform(matrix[0], matrix[1], matrix[2], matrix[3], 0, 0));
        updateImageViews();
    }
    for (const QString& line : update.logs) {
        videoLog->appendPlainText(line);
        if (line.startsWith("Preview disabled:")) {
            videoPreviewImage = {};
            videoPreview->setText(line);
        }
    }
    for (const auto& result : update.results) {
        if (videoResults->rowCount() == 200)
            videoResults->removeRow(0);
        const int row = videoResults->rowCount();
        videoResults->insertRow(row);
        videoResults->setItem(row, 0, new QTableWidgetItem(QString::number(result.frame)));
        videoResults->setItem(row, 1, new QTableWidgetItem(result.seconds < 0 ? "Unknown" : QString::number(result.seconds, 'f', 3) + " s"));
        videoResults->setItem(row, 2, new QTableWidgetItem(QString::fromStdString(WatermarkCore::formatCorrelationPercent(result.correlation))));
    }
    if (!update.results.empty())
        videoResults->scrollToBottom();
    if (update.detections)
        videoDetectionSummary->setText(QString("%1 sampled frames  |  Mean correlation %2  |  Showing recent results, export CSV for all frames")
                .arg(update.detections)
                .arg(QString::fromStdString(WatermarkCore::formatCorrelationPercent(update.meanCorrelation))));
}
