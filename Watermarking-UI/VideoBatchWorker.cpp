#include "VideoBatch.hpp"
#include "VideoBatchWorker.hpp"
#include <exception>
#include <QFileInfo>
#include <QString>
#include <QThread>
#include <string>
#include <utility>
#include <vector>

/*!
 * \brief Delivery of bounded video batch progress to the Qt UI
 * \author Dimitris Karatzas
 */

VideoBatchWorker::VideoBatchWorker(WatermarkCore::VideoBatchSettings settings, std::vector<WatermarkCore::VideoBatchJob> jobs, QObject* parent)
    : QThread(parent), settings_(std::move(settings)), jobs_(std::move(jobs)) {}

void VideoBatchWorker::run() {
    try {
        summary_ = WatermarkCore::runVideoBatch(settings_, jobs_, control_, this);
    } catch (const std::exception& error) { error_ = QString::fromUtf8(error.what()); }
}

void VideoBatchWorker::onUpdate(const WatermarkCore::VideoBatchUpdate& update) {
    const std::lock_guard lock(mutex_);
    // Coalesce fast worker updates instead of adding one Qt event for every change
    pending_[update.id] = update;
}

void VideoBatchWorker::onLog(const std::uint64_t id, const std::string& message) {
    const std::lock_guard lock(mutex_);
    if (logs_.size() == 100)
        logs_.removeFirst();
    const QString file = id > 0 && id <= jobs_.size() ? QFileInfo(QString::fromStdString(jobs_[id - 1].inputPath)).fileName() : QString::number(id);
    logs_.push_back(file + ": " + QString::fromStdString(message).trimmed());
}

VideoBatchMailbox VideoBatchWorker::takeUpdate() {
    const std::lock_guard lock(mutex_);
    VideoBatchMailbox result;
    result.jobs.reserve(pending_.size());
    for (auto& [id, update] : pending_)
        result.jobs.push_back(std::move(update));
    pending_.clear();
    result.logs = std::move(logs_);
    return result;
}
