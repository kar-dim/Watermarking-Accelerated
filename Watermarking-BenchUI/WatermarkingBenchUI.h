#pragma once
#include "BenchmarkWorker.hpp"
#include <QLabel>
#include <QMainWindow>
#include <QProgressBar>
#include <QPushButton>
#include <QSpinBox>
#include <QString>
#include <QWidget>

class WatermarkingBenchUI : public QMainWindow {
    Q_OBJECT

  public:
    explicit WatermarkingBenchUI(QWidget* parent = nullptr);

  private slots:
    void startBenchmark();
    void onResultReady(const QImage& img, const int p, const float psnr, const double embedTime, const double detectTime, const double embedFps, const double detectFps, const QString& file,
                       const float correlation);

  private:
    QPushButton* startButton;
    QLabel* statusLabel;
    QProgressBar* progressBar;
    QLabel* imageView;
    BenchmarkWorker* worker = nullptr;
    QLabel* deviceLabel = nullptr; // opencl
    QSpinBox* deviceSpinBox = nullptr;

    void centerWindowOnScreen();
};