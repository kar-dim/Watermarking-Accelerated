#include "ImageWorker.hpp"
#include "ImagePreview.hpp"
#include <algorithm>
#include <chrono>
#include <exception>
#include <filesystem>
#include <optional>
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
            correlation_ = detectLoadedImage(session.get());
            return;
        }
        embedImage(session.get());
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

    // background saves, one per thread
    std::optional<ImageSaver> saver;
    // notify the UI of the finished saves (in the order they finished)
    const auto reportSaves = [&](const std::vector<SaveResult>& results) {
        for (const auto& [index, error] : results) {
            if (error.empty()) {
                ++succeeded;
                emit itemState(static_cast<int>(index), "Done", "Saved");
            } else {
                ++failed;
                emit itemState(static_cast<int>(index), "Error", QString::fromStdString(error));
            }
        }
    };

    try {
        if (files_.empty())
            throw std::runtime_error("No image files were selected");
        initializeEnvironment(options_.deviceIndex);
        if (embed_)
            fs::create_directories(outputDir_);
        auto session = createImageSession(options_.password.toStdString(), options_.p, options_.psnr);
        if (embed_)
            saver.emplace(std::min(files_.size(), static_cast<size_t>(omp_get_max_threads())));
        // the next images load in the background while the current image uses the selected device
        ImagePrefetcher images(files_, getCurrentDeviceIndex());
        for (size_t index = 0; index < files_.size() && !isInterruptionRequested(); ++index) {
            emit itemState(static_cast<int>(index), "Working", "");
            try {
                bindPreloadedImage(session.get(), images.next());
                if (embed_) {
                    embedImage(session.get());
                    saver->save(session.get(), (outputDir_ / files_[index].filename()).string(), index);
                    reportSaves(saver->takeFinished());
                } else {
                    const float correlation = detectLoadedImage(session.get());
                    ++succeeded;
                    emit itemState(static_cast<int>(index), "Done", QString("Correlation: %1").arg(correlation, 0, 'f', 4));
                }
            } catch (const std::exception& error) {
                ++failed;
                emit itemState(static_cast<int>(index), "Error", QString::fromUtf8(error.what()));
            }
        }
    } catch (const std::exception& error) { fatalError = QString::fromUtf8(error.what()); }

    // Ensure all background save operations complete before finishing
    if (saver)
        reportSaves(saver->finish());

    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    emit summaryReady(succeeded, failed, seconds, isInterruptionRequested(), fatalError);
}
