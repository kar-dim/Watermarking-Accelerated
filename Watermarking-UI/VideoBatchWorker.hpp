#pragma once

#include <map>
#include <mutex>
#include <QString>
#include <QStringList>
#include <QThread>
#include <vector>
#include <VideoBatch.hpp>

/*!
 * \brief Background video batch processing with coalesced per-file updates
 * \author Dimitris Karatzas
 */

struct VideoBatchMailbox {
    std::vector<WatermarkCore::VideoBatchUpdate> jobs;
    QStringList logs;
};

class VideoBatchWorker : public QThread, private WatermarkCore::VideoBatchObserver {
  public:
    VideoBatchWorker(WatermarkCore::VideoBatchSettings settings, std::vector<WatermarkCore::VideoBatchJob> jobs, QObject* parent = nullptr);
    ~VideoBatchWorker() override {
        cancel();
        wait();
    }
    void cancel() { control_.cancel(); }
    bool skipCurrent(const std::uint64_t id) { return control_.skipCurrent(id); }
    bool cancellationRequested() const { return control_.cancelled(); }
    VideoBatchMailbox takeUpdate();
    const WatermarkCore::VideoBatchSummary& summary() const { return summary_; }
    const QString& error() const { return error_; }

  protected:
    void run() override;

  private:
    void onUpdate(const WatermarkCore::VideoBatchUpdate& update) override;
    void onLog(std::uint64_t id, const std::string& message) override;
    WatermarkCore::VideoBatchSettings settings_;
    std::vector<WatermarkCore::VideoBatchJob> jobs_;
    WatermarkCore::VideoBatchControl control_;
    WatermarkCore::VideoBatchSummary summary_;
    QString error_;
    std::mutex mutex_;
    std::map<std::uint64_t, WatermarkCore::VideoBatchUpdate> pending_;
    QStringList logs_;
};
