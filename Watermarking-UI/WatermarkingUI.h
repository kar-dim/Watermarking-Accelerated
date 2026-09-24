#pragma once

#include "BenchmarkWorker.hpp"
#include "ImageWorker.hpp"
#include "ComparisonView.hpp"
#include <QCloseEvent>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QEvent>
#include <QImage>
#include <QLabel>
#include <QLineEdit>
#include <QMainWindow>
#include <QProgressBar>
#include <QPushButton>
#include <QScrollArea>
#include <QResizeEvent>
#include <QTabWidget>
#include <QTableWidget>
#include <QString>
#include <QWidget>
#include <filesystem>
#include <vector>

/*!
 *  \brief  Main graphical user interface for watermarking and benchmarking
 *  \author Dimitris Karatzas
 */

// Main window widget for watermarking and benchmark tasks
class WatermarkingUI : public QMainWindow {
    Q_OBJECT

  public:
    explicit WatermarkingUI(QWidget* parent = nullptr);

  protected:
    // Window lifecycle, sizing, and drag drop event handlers
    void closeEvent(QCloseEvent* event) override;
    void resizeEvent(QResizeEvent* event) override;
    bool eventFilter(QObject* watched, QEvent* event) override;

  private:
    // Input controls for watermarking parameters
    struct SettingsControls {
        QLineEdit* password = nullptr;
        QComboBox* p = nullptr;
        QDoubleSpinBox* psnr = nullptr;
        QWidget* psnrColumn = nullptr;
    };
    QWidget* createSettingsRow(SettingsControls& controls, QWidget* parent);
    // Helper methods for settings management and UI workflow actions
    ImageOptions imageOptions(bool batch) const;
    void setBusy(bool busy);
    void invalidateSingleResult();
    void selectSingleImage(const QString& path);
    void selectBatchFolder(const QString& path);
    void updateImageViews();
    void startSingleOperation();
    void saveSingleImage();
    void refreshBatchQueue();
    void startBatch();
    void cancelBatch();
    void startBenchmark();
    void resetBenchmarkView();
    void finishBenchmarkCanceled();
    void onBenchmarkResult(const QImage& image, int p, float psnr, double embedTime, double detectTime, double embedFps, double detectFps, const QString& file, float correlation);

    // Tab widget and shared compute device picker
    QTabWidget* tabs = nullptr;
    SettingsControls singleSettings;
    SettingsControls batchSettings;
    QComboBox* deviceComboBox = nullptr;

    // Single image tab controls
    QLineEdit* singleInput = nullptr;
    QLineEdit* singleOutput = nullptr;
    QComboBox* singleMode = nullptr;
    QPushButton* previewButton = nullptr;
    QPushButton* saveButton = nullptr;
    QLabel* singleStatus = nullptr;
    ComparisonView* singleImageView = nullptr;
    QWidget* singleDetectArea = nullptr;
    QLabel* singleDetectionResult = nullptr;
    SingleImageWorker* singleWorker = nullptr;

    // Batch processing tab controls
    QLineEdit* batchFolder = nullptr;
    QComboBox* batchMode = nullptr;
    QPushButton* batchStartButton = nullptr;
    QPushButton* batchCancelButton = nullptr;
    QLabel* batchStatus = nullptr;
    QLabel* batchOutputHint = nullptr;
    QProgressBar* batchProgress = nullptr;
    QTableWidget* batchTable = nullptr;
    BatchImageWorker* batchWorker = nullptr;
    std::vector<std::filesystem::path> batchFiles;
    int batchCompleted = 0;

    // Benchmark tab controls
    QPushButton* benchmarkStartButton = nullptr;
    QLabel* benchmarkStatus = nullptr;
    QProgressBar* benchmarkProgress = nullptr;
    QLabel* benchmarkImageView = nullptr;
    QLabel* dropOverlay = nullptr;
    QImage benchmarkPreviewImage;
    BenchmarkWorker* benchmarkWorker = nullptr;
    bool busy = false;
};
