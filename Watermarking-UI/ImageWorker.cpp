#include "ImageWorker.hpp"
#include "ImagePreview.hpp"
#include <algorithm>
#include <chrono>
#include <exception>
#include <filesystem>
#include <future>
#include <queue>
#include <stdexcept>
#include <utility>
#include <omp.h>
#include <WatermarkCore.hpp>

using namespace WatermarkCore;
namespace fs = std::filesystem;

/*!
 *  \brief  Implementation of single image and batch watermarking workers
 *  \author Dimitris Karatzas
 */

// Initialize worker thread with options, input path, and detection mode
SingleImageWorker::SingleImageWorker(ImageOptions options, QString inputPath, const bool detect, QObject* parent)
    : QThread(parent), options_(std::move(options)), inputPath_(std::move(inputPath)), detect_(detect) {}

// Run watermark embed or detect on the designated background thread
void SingleImageWorker::run() {
    try {
        initializeEnvironment(options_.deviceIndex);
        auto session = createImageSession(options_.password.toStdString(), options_.p, options_.psnr);
        // The core supplies oriented display pixels only when embedding
        loadImage(session.get(), inputPath_.toStdString(), !detect_);
        if (detect_) {
            correlation_ = detectLoadedImage(session.get(), MaskMethod::ME);
            return;
        }
        embedImage(session.get(), MaskMethod::ME);
        finish();
        // Copy the embedded pixels once, saving later uses the same session output
        preview_ = imagePreviewFromSession(session.get());
        if (preview_.isNull())
            throw std::runtime_error("Could not create the watermarked preview");
        original_ = originalPreviewFromSession(session.get());
        if (original_.isNull())
            throw std::runtime_error("Could not load the original image for comparison");
        session_ = std::move(session);
    } catch (const std::exception& error) { error_ = QString::fromUtf8(error.what()); }
}

// Initialize batch worker with image file list and output directory
BatchImageWorker::BatchImageWorker(ImageOptions options, std::vector<fs::path> files, fs::path outputDir, const bool embed, QObject* parent)
    : QThread(parent), options_(std::move(options)), files_(std::move(files)), outputDir_(std::move(outputDir)), embed_(embed) {}

// Execute batch pipeline with asynchronous prefetching and disk saving
void BatchImageWorker::run() {
    const auto started = std::chrono::steady_clock::now();
    int succeeded = 0;
    int failed = 0;
    QString fatalError;

    struct PendingSave {
        int index;
        std::future<void> task;
    };
    std::vector<ExportHandle> exportBuffers;
    std::queue<PendingSave> saves;
    // Drain the oldest pending asynchronous disk save and notify the UI
    const auto completeOldestSave = [&]() {
        PendingSave pending = std::move(saves.front());
        saves.pop();
        try {
            pending.task.get();
            ++succeeded;
            emit itemState(pending.index, "Done", "Saved");
        } catch (const std::exception& error) {
            ++failed;
            emit itemState(pending.index, "Error", QString::fromUtf8(error.what()));
        }
    };

    try {
        if (files_.empty())
            throw std::runtime_error("No image files were selected");
        initializeEnvironment(options_.deviceIndex);
        if (embed_)
            fs::create_directories(outputDir_);
        auto session = createImageSession(options_.password.toStdString(), options_.p, options_.psnr);

        if (embed_) {
            const size_t poolSize = std::min(files_.size(), static_cast<size_t>(omp_get_max_threads()));
            for (size_t index = 0; index < poolSize; ++index)
                exportBuffers.push_back(createReusableExportBuffer());
        }
        size_t nextBuffer = 0;
        // Preload the next image while the current image uses the selected device
        const int device = getCurrentDeviceIndex();
        const auto prefetch = [this, device](const size_t index) { return std::async(std::launch::async, [path = files_[index], device] { return preloadImageFromDisk(path.string(), device); }); };
        std::future<PreloadedHandle> nextImage = prefetch(0);
        for (size_t index = 0; index < files_.size() && !isInterruptionRequested(); ++index) {
            emit itemState(static_cast<int>(index), "Working", "");
            bool nextStarted = false;
            try {
                auto image = nextImage.get();
                if (index + 1 < files_.size()) {
                    nextImage = prefetch(index + 1);
                    nextStarted = true;
                }
                bindPreloadedImage(session.get(), std::move(image));
                if (embed_) {
                    embedImage(session.get(), MaskMethod::ME);
                    if (saves.size() == exportBuffers.size())
                        completeOldestSave();
                    ExportedImage* buffer = exportBuffers[nextBuffer].get();
                    exportForSave(session.get(), buffer, MaskMethod::ME);
                    const fs::path outputPath = outputDir_ / files_[index].filename();
                    saves.push(PendingSave{static_cast<int>(index), std::async(std::launch::async, flushToDiskAsync, buffer, outputPath.string(), MaskMethod::ME)});
                    nextBuffer = (nextBuffer + 1) % exportBuffers.size();
                } else {
                    const float correlation = detectLoadedImage(session.get(), MaskMethod::ME);
                    ++succeeded;
                    emit itemState(static_cast<int>(index), "Done", QString("Correlation: %1").arg(correlation, 0, 'f', 4));
                }
            } catch (const std::exception& error) {
                ++failed;
                emit itemState(static_cast<int>(index), "Error", QString::fromUtf8(error.what()));
                if (index + 1 < files_.size() && !nextStarted)
                    nextImage = prefetch(index + 1);
            }
        }
    } catch (const std::exception& error) { fatalError = QString::fromUtf8(error.what()); }

    // Ensure all background save operations complete before finishing
    while (!saves.empty())
        completeOldestSave();

    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    emit summaryReady(succeeded, failed, seconds, isInterruptionRequested(), fatalError);
}
