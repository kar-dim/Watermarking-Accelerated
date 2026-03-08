#include "BenchmarkWorker.hpp"
#include "WatermarkingBenchUI.h"
#include <QCloseEvent>
#include <QComboBox>
#include <QCoreApplication>
#include <QMainWindow>
#include <QMessageBox>
#include <QPixmap>
#include <QPushButton>
#include <QRect>
#include <QScreen>
#include <QString>
#include <QStyle>
#include <QVBoxLayout>
#include <QWidget>
#include <WatermarkCore.hpp>

using namespace WatermarkCore;

WatermarkingBenchUI::WatermarkingBenchUI(QWidget* parent) : QMainWindow(parent) {
    // initialize the UI elements
    startButton = new QPushButton("START BENCHMARK", this);
    startButton->setMinimumHeight(50);

    statusLabel = new QLabel("Ready", this);
    statusLabel->setAlignment(Qt::AlignCenter);
    statusLabel->setObjectName("statusLabel");

    progressBar = new QProgressBar(this);
    progressBar->setVisible(false);

    imageView = new QLabel(this);
    imageView->setAlignment(Qt::AlignCenter);
    // if image is larger than the view, scale it down to fit while maintaining aspect ratio
    imageView->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
    imageView->setMinimumSize(400, 300);
    imageView->setVisible(false);

    QVBoxLayout* layout = new QVBoxLayout;
    layout->addWidget(statusLabel);
    layout->addWidget(progressBar);
    layout->addWidget(imageView, 1);

    // for OpenCL backend only, add a device selection combobox to choose between different OpenCL devices (if multiple are available)
    if (isOpenCLBackend()) {
        QHBoxLayout* deviceLayout = new QHBoxLayout;

        deviceLabel = new QLabel("OpenCL Device:", this);
        deviceLabel->setObjectName("deviceLabel");
        deviceLabel->setAlignment(Qt::AlignRight | Qt::AlignVCenter);

        deviceComboBox = new QComboBox(this);
        deviceComboBox->setObjectName("deviceComboBox");
        for (const auto& devName : getAvailableDevices())
            deviceComboBox->addItem(QString::fromStdString(devName));

        deviceLayout->addStretch();
        deviceLayout->addWidget(deviceLabel);
        deviceLayout->addWidget(deviceComboBox);
        deviceLayout->addStretch();
        deviceLayout->setContentsMargins(0, 10, 0, 10);

        layout->addLayout(deviceLayout);
    }

    layout->addWidget(startButton, 0, Qt::AlignCenter);

    QWidget* central = new QWidget(this);
    central->setLayout(layout);
    setCentralWidget(central);

    // make the window small and center it on the screen initially, it will be resized and re-centered during benchmarking to better show the results
    adjustSize();
    centerWindowOnScreen();

    // connect the start button to the benchmark function
    connect(startButton, &QPushButton::clicked, this, &WatermarkingBenchUI::startBenchmark);
}

// do not close the UI while the background bench thread is running (when user clicks "X" randomly)
// we ask the worker to stop and wait for it
void WatermarkingBenchUI::closeEvent(QCloseEvent* event) {
    if (worker && worker->isRunning()) {
        worker->requestInterruption();
        worker->wait();
    }
    event->accept();
}

void WatermarkingBenchUI::startBenchmark() {
    // disable/alter UI elements (and enable progress bar) to prevent interference during benchmarking
    startButton->setEnabled(false);
    progressBar->setValue(0);
    progressBar->setVisible(true);
    int selectedDevice = 0;
    if (deviceComboBox) {
        selectedDevice = deviceComboBox->currentIndex();
        deviceComboBox->setEnabled(false);
    }
    statusLabel->setProperty("benchmarkState", "running");
    statusLabel->style()->unpolish(statusLabel);
    statusLabel->style()->polish(statusLabel);
    statusLabel->setText("Benchmarking...");
    imageView->setVisible(true);
    // resize the window to a larger size to better show the results during benchmarking, and move to center
    const QRect screenGeometry = QGuiApplication::primaryScreen()->availableGeometry();
    resize(screenGeometry.width() * 0.75, screenGeometry.height() * 0.75);
    centerWindowOnScreen();

    // initialize benchmark worker
    worker = new BenchmarkWorker(selectedDevice, this);

    // signal to update progress bar
    connect(worker, &BenchmarkWorker::progressUpdated, progressBar, [this](const int current, const int total) {
        progressBar->setMaximum(total);
        progressBar->setValue(current);
    });

    // signal to update the status label and image view with the latest results
    connect(worker, &BenchmarkWorker::resultReady, this, &WatermarkingBenchUI::onResultReady);

    // signal to handle benchmark completion, show the final score and re-enable the start button and device selection
    connect(worker, &BenchmarkWorker::benchmarkFinished, this, [this](const double finalEmbedFps, const double finalDetectFps, const int finalScore) {
        // re-enable the UI elements
        startButton->setEnabled(true);
        progressBar->setVisible(false);
        if (deviceComboBox)
            deviceComboBox->setEnabled(true);
        // failure case
        if (finalScore == 0) {
            statusLabel->setText("<b>BENCHMARK FAILED</b><br>");
            statusLabel->setProperty("benchmarkState", "failure");
            QMessageBox::critical(this, "Benchmark Failed",
                                  "The benchmark encountered an error and could not complete.\n\nPossible causes:\n"
                                  "- No temporary directory user rights\n- (GPU only case): OpenCL/CUDA driver crash\n- Out of memory");
        } else {
            // show the final score for both pipelines
            const QString backendName = QString::fromStdString(WatermarkCore::getDeviceName(deviceComboBox ? deviceComboBox->currentIndex() : -1));
            const QString finalMessage = QString("<b>BENCHMARK COMPLETE</b><br>Hardware: %1<br>Avg Embed: <b>%2 FPS</b> | Avg Detect: <b>%3 FPS</b><br><br>SCORE: <b>%4</b>")
                                             .arg(backendName)
                                             .arg(finalEmbedFps, 0, 'f', 1)
                                             .arg(finalDetectFps, 0, 'f', 1)
                                             .arg(finalScore);
            statusLabel->setText(finalMessage);
            statusLabel->setProperty("benchmarkState", "success");
        }
        statusLabel->style()->unpolish(statusLabel);
        statusLabel->style()->polish(statusLabel);
        // hide the image view and shrink
        imageView->setVisible(false);
        this->adjustSize();
        centerWindowOnScreen();
        worker->deleteLater();
        worker = nullptr; // we must nullify because we check it later
    });

    worker->start();
}

void WatermarkingBenchUI::onResultReady(const QImage& img, const int p, const float psnr, const double embedTime, const double detectTime, const double embedFps, const double detectFps,
                                        const QString& file, const float correlation) {
    // clang-format off
    // update the status label with the current parameters and performance metrics
    const QString statusText = QString("File: %1 | Block (p): %2 | PSNR: %3 dB | Corr: %4\nEmbed: %5 ms (%6 FPS)  ||  Detect: %7 ms (%8 FPS)")
                                   .arg(file).arg(p).arg(psnr, 0, 'f', 1).arg(correlation, 0, 'f', 4).arg(embedTime, 0, 'f', 2)
                                   .arg(embedFps, 0, 'f', 1).arg(detectTime, 0, 'f', 2).arg(detectFps, 0, 'f', 1);
    // clang-format on
    // update the image view with the new image (scaled to fit the view while maintaining aspect ratio)
    statusLabel->setText(statusText);
    imageView->setPixmap(QPixmap::fromImage(img).scaled(imageView->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
}

// helper function to center the window on the screen, called after resizing to ensure it stays centered
void WatermarkingBenchUI::centerWindowOnScreen() {
    const QRect screenGeometry = QGuiApplication::primaryScreen()->availableGeometry();
    move(screenGeometry.x() + (screenGeometry.width() - width()) / 2, screenGeometry.y() + (screenGeometry.height() - height()) / 2);
}
