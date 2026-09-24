#pragma once

#include <filesystem>
#include <optional>
#include <QImage>
#include <QString>
#include <QThread>
#include <vector>
#include <WatermarkCore.hpp>

/*!
 *  \brief  Worker threads for asynchronous single image and batch watermarking
 *  \author Dimitris Karatzas
 */

// Common parameters for watermark operations
struct ImageOptions {
    QString password;
    int p = 3;
    float psnr = 40.0f;
    int deviceIndex = 0;
};

// Worker thread for processing a single image without blocking the UI
class SingleImageWorker : public QThread {
  public:
    SingleImageWorker(ImageOptions options, QString inputPath, bool detect, QObject* parent = nullptr);

    // Result accessors for previews, error messages, and detection correlation
    [[nodiscard]] const QImage& preview() const { return preview_; }
    [[nodiscard]] const QImage& original() const { return original_; }
    [[nodiscard]] const QString& error() const { return error_; }
    [[nodiscard]] std::optional<float> correlation() const { return correlation_; }
    [[nodiscard]] WatermarkCore::ImageSession* session() const { return session_.get(); }

  protected:
    void run() override;

  private:
    // Internal worker parameters, session handle, and cached results
    ImageOptions options_;
    QString inputPath_;
    bool detect_ = false;
    WatermarkCore::ImageHandle session_;
    QImage preview_;
    QImage original_;
    QString error_;
    std::optional<float> correlation_;
};

// Worker thread for batch processing a folder of images
class BatchImageWorker : public QThread {
    Q_OBJECT
  public:
    BatchImageWorker(ImageOptions options, std::vector<std::filesystem::path> files, std::filesystem::path outputDir, bool embed, QObject* parent = nullptr);

  signals:
    // Notify the UI when a file state changes or when the whole batch finishes
    void itemState(int index, const QString& state, const QString& detail);
    void summaryReady(int succeeded, int failed, double seconds, bool canceled, const QString& fatalError);

  protected:
    void run() override;

  private:
    // Batch options, file list, destination directory, and embed flag
    ImageOptions options_;
    std::vector<std::filesystem::path> files_;
    std::filesystem::path outputDir_;
    bool embed_;
};
