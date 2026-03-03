#pragma once
#include <array>
#include <QImage>
#include <QString>
#include <QTemporaryDir>
#include <QThread>
#include <WatermarkEngine.hpp>

class BenchmarkWorker : public QThread {
    Q_OBJECT
  public:
    explicit BenchmarkWorker(int openclDevice = 0, QObject* parent = nullptr);

  signals:
    void resultReady(const QImage& img, const int p, const float psnr, const double embedTime, const double detectTime, const double embedFps, const double detectFps, const QString& fileName,
                     const float correlation);
    void progressUpdated(const int currentStep, const int totalSteps);
    void errorOccurred(const QString& fileName, const QString& errorMessage);
    void benchmarkFinished(const double finalEmbedFps, const double finalDetectFps, const int finalScore);

  protected:
    void run() override;

  private:
    static constexpr std::array<int, 4> pValues{3, 5, 7, 9};
    static constexpr std::array<float, 11> psnrValues{10.0f, 15.0f, 20.0f, 25.0f, 30.0f, 35.0f, 40.0f, 45.0f, 50.0f, 55.0f, 60.0f};

    QTemporaryDir tempDir;
    QString inputFolder;
    int deviceIndex;

    QImage convertToQtFormat(WatermarkEngine::ImageSession* session) const;
};