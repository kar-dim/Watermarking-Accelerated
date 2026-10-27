#pragma once
#include <QSet>
#include <QElapsedTimer>

#include "BenchmarkWorker.hpp"
#include "ImageWorker.hpp"
#include "ComparisonView.hpp"
#include "VideoWorker.hpp"
#include "VideoBatchWorker.hpp"
#include <QCheckBox>
#include <QPlainTextEdit>
#include <QSpinBox>
#include <QTimer>
#include <QCloseEvent>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QEvent>
#include <QImage>
#include <QHash>
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

class ImagePreviewView;
class MultiDeviceComboBox;

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
    void addImageBatchPaths(const QStringList& paths);
    void updateImageViews();
    void startSingleOperation();
    void saveSingleImage();
    void refreshBatchQueue();
    void startBatch();
    void pollImageBatch();
    void cancelBatch();
    void startBenchmark();
    void resetBenchmarkView();
    void finishBenchmarkCanceled();
    void onBenchmarkResult(const QImage& image, int p, float psnr, double embedTime, double detectTime, double embedFps, double detectFps, const QString& file, float correlation);
    void createVideoTab();
    void startVideo();
    void pollVideo();
    void selectVideo(const QString& path);
    void probeVideoInput();
    void updateVideoControls();
    void updateVideoPreviewLayout();
    void updateVideoDevice(int device);
    QString videoHardwareDefaults() const;
    void createVideoBatchTab();
    void addVideoBatchPaths(const QStringList& paths);
    void refreshVideoBatchTable();
    void updateVideoBatchControls();
    void updateVideoBatchDevice(int device);
    void updateVideoBatchSelection();
    void updateVideoBatchSkip();
    void startVideoBatch();
    void pollVideoBatch();

    // Tab widget and shared compute device picker
    QTabWidget* tabs = nullptr;
    SettingsControls singleSettings;
    SettingsControls batchSettings;
    QComboBox* deviceComboBox = nullptr;
    QWidget* devicePickerArea = nullptr;

    // Single image tab controls
    QLineEdit* singleInput = nullptr;
    QLineEdit* singleOutput = nullptr;
    QComboBox* singleMode = nullptr;
    QPushButton* previewButton = nullptr;
    QPushButton* saveButton = nullptr;
    QLabel* singleStatus = nullptr;
    ComparisonView* singleImageView = nullptr;
    QWidget* singleDetectArea = nullptr;
    ImagePreviewView* singleDetectionPreview = nullptr;
    QLabel* singleDetectionFile = nullptr;
    QLabel* singleDetectionResult = nullptr;
    SingleImageWorker* singleWorker = nullptr;

    // Batch processing tab controls
    QLineEdit* batchOutput = nullptr;
    QWidget* batchOutputRow = nullptr;
    QPushButton* batchAddFiles = nullptr;
    QPushButton* batchAddFolder = nullptr;
    QPushButton* batchRemove = nullptr;
    QPushButton* batchClear = nullptr;
    QComboBox* batchMode = nullptr;
    QPushButton* batchStartButton = nullptr;
    QPushButton* batchCancelButton = nullptr;
    QLabel* batchStatus = nullptr;
    QLabel* batchElapsed = nullptr;
    QElapsedTimer batchClock;
    QTimer* batchTimer = nullptr;
    QLabel* batchOutputHint = nullptr;
    QProgressBar* batchProgress = nullptr;
    QTableWidget* batchTable = nullptr;
    BatchImageWorker* batchWorker = nullptr;
    MultiDeviceComboBox* imageBatchDevices = nullptr;
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
    SettingsControls videoSettings;
    QWidget* videoPage = nullptr;
    QWidget* videoInputs = nullptr;
    QWidget* videoOutputRow = nullptr;
    QWidget* videoEncoderArea = nullptr;
    QComboBox* videoMode = nullptr;
    QLineEdit* videoInput = nullptr;
    QLineEdit* videoOutput = nullptr;
    QLineEdit* videoEncoderOptions = nullptr;
    QSpinBox* videoInterval = nullptr;
    QCheckBox* videoHardware = nullptr;
    QCheckBox* videoPreviewEnabled = nullptr;
    QPushButton* videoStart = nullptr;
    QPushButton* videoCancel = nullptr;
    QPushButton* videoExport = nullptr;
    QLabel* videoStatus = nullptr;
    QLabel* videoInfo = nullptr;
    QLabel* videoPreview = nullptr;
    QLabel* videoDetectionSummary = nullptr;
    QProgressBar* videoProgress = nullptr;
    QTableWidget* videoResults = nullptr;
    QPlainTextEdit* videoLog = nullptr;
    QTimer* videoTimer = nullptr;
    QTimer* videoProbeTimer = nullptr;
    VideoProbeWorker* videoProbeWorker = nullptr;
    bool videoProbePending = false;
    VideoWorker* videoWorker = nullptr;
    QImage videoPreviewImage;
    QString videoSoftwareOptions;
    QHash<int, QString> videoHardwareOptions;
    int videoOptionsDevice = 0;
    SettingsControls videoBatchSettings;
    QWidget* videoBatchPage = nullptr;
    QWidget* videoBatchInputs = nullptr;
    QWidget* videoBatchEncoderArea = nullptr;
    QWidget* videoBatchFormatRow = nullptr;
    QComboBox* videoBatchMode = nullptr;
    QComboBox* videoBatchFormat = nullptr;
    QLineEdit* videoBatchOutput = nullptr;
    QLineEdit* videoBatchEncoderOptions = nullptr;
    QSpinBox* videoBatchInterval = nullptr;
    QCheckBox* videoBatchHardware = nullptr;
    MultiDeviceComboBox* videoBatchDevices = nullptr;
    QCheckBox* videoBatchAcceleration = nullptr;
    QWidget* videoBatchEncoderDeviceRow = nullptr;
    QComboBox* videoBatchEncoderDevice = nullptr;
    QPushButton* videoBatchAddFiles = nullptr;
    QPushButton* videoBatchAddFolder = nullptr;
    QPushButton* videoBatchRemove = nullptr;
    QPushButton* videoBatchClear = nullptr;
    QPushButton* videoBatchStart = nullptr;
    QPushButton* videoBatchCancel = nullptr;
    QPushButton* videoBatchSkip = nullptr;
    QLabel* videoBatchStatus = nullptr;
    QLabel* videoBatchElapsed = nullptr;
    QElapsedTimer videoBatchClock;
    QProgressBar* videoBatchProgress = nullptr;
    QTableWidget* videoBatchTable = nullptr;
    QPlainTextEdit* videoBatchLog = nullptr;
    QTimer* videoBatchTimer = nullptr;
    VideoBatchWorker* videoBatchWorker = nullptr;
    QStringList videoBatchFiles;
    std::vector<WatermarkCore::VideoBatchJob> videoBatchJobs;
    std::uint64_t videoBatchRun = 0;
    std::uint64_t videoBatchActive = 0;
    QSet<std::uint64_t> videoBatchSkipRequested;
    std::size_t videoBatchFinished = 0;
    QString videoBatchSoftwareOptions;
    QHash<int, QString> videoBatchHardwareOptions;
    int videoBatchOptionsDevice = 0;
    bool busy = false;
    bool closing = false;
};
