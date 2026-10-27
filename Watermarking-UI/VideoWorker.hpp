#pragma once

/*!
 * \brief Background video worker with bounded preview and result delivery
 * \author Dimitris Karatzas
 */

#include <atomic>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <QImage>
#include <QString>
#include <QStringList>
#include <QTemporaryFile>
#include <QTextStream>
#include <QThread>
#include <stop_token>
#include <WatermarkCore.hpp>

struct VideoUpdate {
    WatermarkCore::VideoProgress progress;
    QImage preview;
    QStringList logs;
    std::deque<WatermarkCore::VideoDetection> results;
    std::uint64_t detections = 0;
    double meanCorrelation = 0.0;
};

class VideoProbeWorker : public QThread {
  public:
    VideoProbeWorker(QString path, QObject* parent = nullptr);
    ~VideoProbeWorker() override {
        cancel();
        wait();
    }
    void cancel() { stop_.request_stop(); }
    const QString& path() const { return path_; }
    const WatermarkCore::VideoSourceInfo& result() const { return result_; }
    const QString& error() const { return error_; }
    bool cancelled() const { return stop_.stop_requested(); }

  protected:
    void run() override;

  private:
    QString path_;
    WatermarkCore::VideoSourceInfo result_;
    QString error_;
    std::stop_source stop_;
};

class VideoWorker : public QThread, private WatermarkCore::VideoObserver {
  public:
    VideoWorker(WatermarkCore::VideoSettings settings, int device, bool detect, QObject* parent = nullptr);
    ~VideoWorker() override {
        cancel();
        wait();
    }
    void cancel();
    void setPreview(bool enabled, int width, int height);
    VideoUpdate takeUpdate();
    bool exportResults(const QString& destination);
    const QString& error() const { return error_; }
    bool cancelled() const { return cancelled_; }
    bool cancellationRequested() const { return stop_.stop_requested(); }

  protected:
    void run() override;

  private:
    void onProgress(const WatermarkCore::VideoProgress& progress) override;
    void onDetection(const WatermarkCore::VideoDetection& result) override;
    void onLog(const std::string& text) override;
    bool wantsPreview() const override;
    void previewSize(int& width, int& height) const override;
    void onPreview(const std::uint8_t* rgba, int width, int height, int pitch) override;

    WatermarkCore::VideoSettings settings_;
    int device_;
    bool detect_;
    std::stop_source stop_;
    std::atomic<bool> previewEnabled_{false};
    std::atomic<bool> previewPending_{false};
    std::atomic<int> previewWidth_{960}, previewHeight_{540};
    std::mutex mutex_;
    VideoUpdate pending_;
    double correlationSum_ = 0.0;
    QTemporaryFile csv_;
    std::unique_ptr<QTextStream> csvStream_;
    QString error_;
    bool cancelled_ = false;
};
