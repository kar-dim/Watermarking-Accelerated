#include "ImageWorker.hpp"
#include "ImagePreview.hpp"
#include <exception>
#include <stdexcept>
#include <utility>
#include <WatermarkCore.hpp>

using namespace WatermarkCore;

/*!
 *  \brief  Implementation of single image and batch watermarking workers
 *  \author Dimitris Karatzas
 */

SingleImageWorker::SingleImageWorker(ImageOptions options, QString inputPath, const bool detect, QObject* parent)
    : QThread(parent), options_(std::move(options)), inputPath_(std::move(inputPath)), detect_(detect) {}

void SingleImageWorker::run() {
    try {
        initializeEnvironment(options_.deviceIndex);
        auto session = createImageSession(options_.password.toStdString(), options_.p, options_.psnr);
        loadImage(session.get(), inputPath_.toStdString(), true);
        if (detect_) {
            correlation_ = detectLoadedImage(session.get());
            const QImage original = originalPreviewFromSession(session.get());
            if (!original.isNull())
                original_ = original.width() > 2048 || original.height() > 1536 ? original.scaled(2048, 1536, Qt::KeepAspectRatio, Qt::SmoothTransformation) : original;
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

BatchImageWorker::BatchImageWorker(ImageBatchSettings settings, std::vector<ImageBatchJob> jobs, QObject* parent) : QThread(parent), settings_(std::move(settings)), jobs_(std::move(jobs)) {}

void BatchImageWorker::onUpdate(const ImageBatchUpdate& update) {
    const std::lock_guard lock(mutex_);
    auto& saved = pending_[update.id];
    if (update.revision > saved.revision)
        saved = update;
}

std::vector<ImageBatchUpdate> BatchImageWorker::takeUpdates() {
    const std::lock_guard lock(mutex_);
    std::vector<ImageBatchUpdate> values;
    values.reserve(pending_.size());
    for (auto& [id, update] : pending_)
        values.push_back(std::move(update));
    pending_.clear();
    return values;
}

void BatchImageWorker::run() {
    try {
        const auto result = runImageBatch(settings_, jobs_, control_, this);
        emit summaryReady(static_cast<int>(result.succeeded), static_cast<int>(result.failed), result.elapsedSeconds, control_.cancelled(), {});
    } catch (const std::exception& error) { emit summaryReady(0, 0, 0, control_.cancelled(), QString::fromUtf8(error.what())); }
}
