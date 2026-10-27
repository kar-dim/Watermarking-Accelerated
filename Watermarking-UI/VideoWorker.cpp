#include "VideoObserver.hpp"
#include "VideoWorker.hpp"
#include <exception>
#include <memory>
#include <QByteArray>
#include <QFile>
#include <QSaveFile>
#include <stdexcept>
#include <utility>

using namespace WatermarkCore;

/*!
 * \brief Background video processing and detection result export
 * \author Dimitris Karatzas
 */

VideoProbeWorker::VideoProbeWorker(QString path, QObject* parent) : QThread(parent), path_(std::move(path)) {}

void VideoProbeWorker::run() {
    try {
        result_ = probeVideoSource(path_.toStdString(), stop_.get_token());
    } catch (const VideoCancelled&) {
    } catch (const std::exception& error) { error_ = QString::fromUtf8(error.what()); }
}

VideoWorker::VideoWorker(VideoSettings settings, const int device, const bool detect, QObject* parent) : QThread(parent), settings_(std::move(settings)), device_(device), detect_(detect) {
    settings_.observer = this;
    settings_.stopToken = stop_.get_token();
}

void VideoWorker::cancel() { stop_.request_stop(); }

void VideoWorker::setPreview(const bool enabled, const int width, const int height) {
    previewWidth_.store(width);
    previewHeight_.store(height);
    previewEnabled_.store(enabled);
}

VideoUpdate VideoWorker::takeUpdate() {
    const std::lock_guard lock(mutex_);
    VideoUpdate result;
    // Keep the last progress for quiet polls, drain previews and messages only once
    result.progress = pending_.progress;
    result.preview = std::exchange(pending_.preview, {});
    result.logs = std::exchange(pending_.logs, {});
    result.results = std::exchange(pending_.results, {});
    result.detections = pending_.detections;
    result.meanCorrelation = pending_.meanCorrelation;
    previewPending_.store(false);
    return result;
}

void VideoWorker::run() {
    try {
        if (stop_.stop_requested())
            throw VideoCancelled();
        initializeEnvironment(device_);
        if (!detect_) {
            validateVideoOutputPath(settings_.encodeOutputPath);
            validateVideoEncoderOptions(settings_.encodeOptions, settings_.useHwEncoder);
        }
        if (detect_) {
            if (!csv_.open())
                throw std::runtime_error("Could not create the detection results file");
            csvStream_ = std::make_unique<QTextStream>(&csv_);
            *csvStream_ << "frame,time_seconds,correlation\n";
            csvStream_->setRealNumberPrecision(9);
        }
        auto session = initVideo(settings_);
        if (detect_)
            detectVideo(session.get());
        else {
            optimizeThreadsForVideoEmbedding();
            embedVideo(session.get());
        }
    } catch (const VideoCancelled&) { cancelled_ = true; } catch (const std::exception& error) {
        if (stop_.stop_requested())
            cancelled_ = true;
        else
            error_ = QString::fromUtf8(error.what());
    }
    if (csvStream_) {
        csvStream_->flush();
        if (csvStream_->status() != QTextStream::Ok && error_.isEmpty())
            error_ = "Could not write the detection results";
        csvStream_.reset();
        csv_.close();
    }
}

void VideoWorker::onProgress(const VideoProgress& progress) {
    const std::lock_guard lock(mutex_);
    pending_.progress = progress;
}

void VideoWorker::onDetection(const VideoDetection& result) {
    *csvStream_ << result.frame << ',' << (result.seconds < 0 ? QString() : QString::number(result.seconds, 'f', 6)) << ',' << result.correlation << '\n';
    if (csvStream_->status() != QTextStream::Ok)
        throw std::runtime_error("Could not write the detection results");
    const std::lock_guard lock(mutex_);
    // The CSV keeps every sample, only the results waiting for the UI are bounded
    if (pending_.results.size() == 200)
        pending_.results.pop_front();
    pending_.results.push_back(result);
    correlationSum_ += result.correlation;
    ++pending_.detections;
    pending_.meanCorrelation = correlationSum_ / pending_.detections;
}

void VideoWorker::onLog(const std::string& text) {
    const std::lock_guard lock(mutex_);
    if (pending_.logs.size() == 100)
        pending_.logs.removeFirst();
    pending_.logs.push_back(QString::fromUtf8(text).trimmed());
}

bool VideoWorker::wantsPreview() const { return previewEnabled_.load() && !previewPending_.load(); }

void VideoWorker::previewSize(int& width, int& height) const {
    width = previewWidth_.load();
    height = previewHeight_.load();
}

void VideoWorker::onPreview(const std::uint8_t* rgba, const int width, const int height, const int pitch) {
    // The core reuses these pixels as soon as this callback returns
    const QImage owned = QImage(rgba, width, height, pitch, QImage::Format_RGBA8888).copy();
    const std::lock_guard lock(mutex_);
    pending_.preview = owned;
    previewPending_.store(true);
}

bool VideoWorker::exportResults(const QString& destination) {
    if (isRunning() || csv_.fileName().isEmpty())
        return false;
    QFile source(csv_.fileName());
    QSaveFile output(destination);
    if (!source.open(QIODevice::ReadOnly) || !output.open(QIODevice::WriteOnly))
        return false;
    while (!source.atEnd()) {
        const QByteArray block = source.read(64 * 1024);
        if (block.isEmpty() && source.error() != QFile::NoError)
            return false;
        if (output.write(block) != block.size())
            return false;
    }
    return output.commit();
}
