#pragma once
#include "BenchmarkWorker.hpp"
#include <QCloseEvent>
#include <QComboBox>
#include <QImage>
#include <QLabel>
#include <QMainWindow>
#include <QProgressBar>
#include <QPushButton>
#include <QString>
#include <QWidget>

/*!
 *  \brief  Main window of the Benchmark UI project
 *  \author Dimitris Karatzas
 */
class WatermarkingBenchUI : public QMainWindow {
    Q_OBJECT

  public:
    explicit WatermarkingBenchUI(QWidget* parent = nullptr);

  protected:
    void closeEvent(QCloseEvent* event) override;

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
    QLabel* deviceLabel = nullptr;       // opencl
    QComboBox* deviceComboBox = nullptr; // opencl

    void centerWindowOnScreen();
};