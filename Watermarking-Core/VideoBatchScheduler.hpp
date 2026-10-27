#pragma once
#include "include/VideoBatch.hpp"
#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <utility>
#include <vector>

/*!
 * \brief Whole-file claims, bounded retries and one optional final-file backup
 * \author Dimitris Karatzas
 */
namespace WatermarkCore {
class VideoBatchScheduler {
  public:
    struct Claim {
        std::uint64_t id = 0, attempt = 0;
        int device = -1;
        bool backup = false;
        std::stop_token stop;
    };

    VideoBatchScheduler(const std::vector<VideoBatchJob>& jobs, const std::vector<int>& devices, const bool accelerateFinal) : accelerateFinal_(accelerateFinal && devices.size() > 1) {
        for (const int device : devices)
            if (!devices_.emplace(device, true).second)
                throw std::invalid_argument("A batch lists each device once");
        if (devices_.empty())
            throw std::invalid_argument("Select at least one batch device");
        for (const auto& job : jobs) {
            Record record;
            record.view.id = job.id;
            record.view.revision = 1;
            if (!records_.emplace(job.id, std::move(record)).second || !job.id)
                throw std::invalid_argument("Batch job IDs must be unique and nonzero");
            order_.push_back(job.id);
        }
    }

    std::optional<Claim> claim(const int device) {
        std::unique_lock lock(mutex_);
        for (;;) {
            if (cancelled_ || !devices_.at(device))
                return {};
            Record* next = nullptr;
            for (const auto state : {VideoBatchState::Retrying, VideoBatchState::Pending}) {
                for (const auto id : order_) {
                    auto& record = records_.at(id);
                    if (record.view.state == state && !record.tried.contains(device)) {
                        next = &record;
                        break;
                    }
                }
                if (next)
                    break;
            }
            bool backup = false;
            if (!next && accelerateFinal_) {
                // Keep GPUs free for queued files, duplicate work only for the last unfinished video
                Record* candidate = nullptr;
                bool blocked = false;
                for (auto& [id, record] : records_) {
                    if (record.view.state == VideoBatchState::Pending || record.view.state == VideoBatchState::Retrying) {
                        blocked = true;
                        break;
                    }
                    if (!terminal(record.view.state)) {
                        if (candidate) {
                            blocked = true;
                            break;
                        }
                        candidate = &record;
                    }
                }
                if (!blocked && candidate && candidate->active.size() == 1 && !candidate->backupStarted && !candidate->skip && !candidate->tried.contains(device)) {
                    next = candidate;
                    next->backupStarted = true;
                    backup = true;
                }
            }
            if (next) {
                const auto attempt = ++next->attempts;
                // Retried files and final backups must not run twice on the same device
                next->tried.insert(device);
                Active execution;
                execution.device = device;
                execution.view.id = next->view.id;
                execution.view.attempt = attempt;
                execution.view.deviceIndex = device;
                execution.view.state = VideoBatchState::Probing;
                const auto stop = execution.stop.get_token();
                next->active.emplace(attempt, std::move(execution));
                if (!backup)
                    next->view = preserveRevision(*next, next->active.at(attempt).view);
                refresh(*next);
                changed_.notify_all();
                return Claim{next->view.id, attempt, device, backup, stop};
            }
            // Completed row may still have a losing attempt shutting down
            const bool unsettled = std::ranges::any_of(records_, [](const auto& item) { return !terminal(item.second.view.state) || !item.second.active.empty(); });
            if (!unsettled)
                return {};
            changed_.wait(lock);
        }
    }

    void progress(const Claim& claim, const VideoBatchUpdate& view) {
        const std::lock_guard lock(mutex_);
        auto& record = records_.at(claim.id);
        auto found = record.active.find(claim.attempt);
        if (found == record.active.end() || terminal(record.view.state))
            return;
        found->second.view = view;
        found->second.view.deviceIndex = claim.device;
        found->second.view.attempt = claim.attempt;
        // One visible row follows the furthest attempt while each attempt keeps its own progress
        const Active* furthest = &record.active.begin()->second;
        for (const auto& [attempt, active] : record.active)
            if (active.view.progress.frames > furthest->view.progress.frames)
                furthest = &active;
        record.view = preserveRevision(record, furthest->view);
        record.view.id = claim.id;
        refresh(record);
    }

    bool complete(const Claim& claim, const VideoBatchUpdate& result, const std::function<void()>& publish) {
        const std::lock_guard lock(mutex_);
        auto& record = records_.at(claim.id);
        auto& execution = record.active.at(claim.attempt);
        bool accepted = false;
        if (!terminal(record.view.state) && !cancelled_ && !record.skip && !execution.stop.stop_requested()) {
            // Publication and winner selection share this lock
            publish();
            record.view = preserveRevision(record, result);
            record.view.id = claim.id;
            record.view.state = VideoBatchState::Completed;
            record.view.progress.fraction = 1.0;
            record.view.attempt = claim.attempt;
            record.view.deviceIndex = claim.device;
            stop(record);
            accepted = true;
        }
        record.active.erase(claim.attempt);
        settle(record);
        refresh(record);
        changed_.notify_all();
        return accepted;
    }

    void fail(const Claim& claim, const std::string& error, const bool retryable, const bool lost = false) {
        const std::lock_guard lock(mutex_);
        auto& record = records_.at(claim.id);
        record.active.erase(claim.attempt);
        if (lost)
            devices_.at(claim.device) = false;
        if (!terminal(record.view.state)) {
            record.view.detail = error;
            record.view.deviceIndex = claim.device;
            record.view.attempt = claim.attempt;
            if (record.active.empty()) {
                if (cancelled_ || record.skip)
                    settle(record);
                else
                    record.view.state = retryable && eligible(record) ? VideoBatchState::Retrying : VideoBatchState::Failed;
            } else {
                record.view = preserveRevision(record, record.active.begin()->second.view);
            }
        }
        refresh(record);
        failUnreachable();
        changed_.notify_all();
    }

    void retire(const int device, const std::string& error) {
        const std::lock_guard lock(mutex_);
        devices_.at(device) = false;
        for (auto& [id, record] : records_) {
            bool abandoned = false;
            for (auto attempt = record.active.begin(); attempt != record.active.end();) {
                if (attempt->second.device == device) {
                    attempt->second.stop.request_stop();
                    attempt = record.active.erase(attempt);
                    abandoned = true;
                } else
                    ++attempt;
            }
            if (!abandoned)
                continue;
            if (!terminal(record.view.state)) {
                if (!record.active.empty())
                    record.view = preserveRevision(record, record.active.begin()->second.view);
                else if (cancelled_ || record.skip)
                    settle(record);
                else {
                    record.view.state = eligible(record) ? VideoBatchState::Retrying : VideoBatchState::Failed;
                    record.view.detail = error;
                }
            }
            refresh(record);
        }
        failUnreachable(error);
        changed_.notify_all();
    }

    bool skip(const std::uint64_t id) {
        const std::lock_guard lock(mutex_);
        for (const auto candidate : order_) {
            auto& record = records_.at(candidate);
            if ((!id || candidate == id) && !terminal(record.view.state) && !record.active.empty()) {
                record.skip = true;
                stop(record);
                changed_.notify_all();
                return true;
            }
        }
        return false;
    }

    void cancel() {
        const std::lock_guard lock(mutex_);
        cancelled_ = true;
        for (auto& [id, record] : records_) {
            if (!terminal(record.view.state)) {
                stop(record);
                if (record.active.empty())
                    settle(record);
                refresh(record);
            }
        }
        changed_.notify_all();
    }

    std::vector<VideoBatchUpdate> snapshot() const {
        const std::lock_guard lock(mutex_);
        std::vector<VideoBatchUpdate> result;
        result.reserve(order_.size());
        for (const auto id : order_)
            result.push_back(records_.at(id).view);
        return result;
    }

    static bool terminal(const VideoBatchState state) {
        return state == VideoBatchState::Completed || state == VideoBatchState::Failed || state == VideoBatchState::Skipped || state == VideoBatchState::Cancelled;
    }

  private:
    struct Active {
        int device = -1;
        std::stop_source stop;
        VideoBatchUpdate view;
    };
    struct Record {
        VideoBatchUpdate view;
        std::uint64_t attempts = 0;
        bool backupStarted = false, skip = false;
        std::set<int> tried;
        std::map<std::uint64_t, Active> active;
    };
    static VideoBatchUpdate preserveRevision(const Record& record, VideoBatchUpdate view) {
        view.revision = record.view.revision;
        return view;
    }
    static void stop(Record& record) {
        for (auto& [id, execution] : record.active)
            execution.stop.request_stop();
    }
    void settle(Record& record) {
        if (!terminal(record.view.state) && record.active.empty() && (cancelled_ || record.skip)) {
            record.view.state = cancelled_ ? VideoBatchState::Cancelled : VideoBatchState::Skipped;
            record.view.detail = cancelled_ ? "Batch cancelled" : "Skipped by user";
        }
    }
    static void refresh(Record& record) {
        record.view.activeDevices.clear();
        for (const auto& [id, execution] : record.active)
            record.view.activeDevices.push_back(execution.device);
        ++record.view.revision;
    }
    bool eligible(const Record& record) const {
        return std::ranges::any_of(devices_, [&](const auto& device) { return device.second && !record.tried.contains(device.first); });
    }
    void failUnreachable(const std::string& reason = "No remaining selected device can process this file") {
        for (auto& [id, record] : records_)
            if ((record.view.state == VideoBatchState::Retrying || record.view.state == VideoBatchState::Pending) && !eligible(record)) {
                record.view.state = VideoBatchState::Failed;
                record.view.detail = reason;
                refresh(record);
            }
    }
    mutable std::mutex mutex_;
    std::condition_variable changed_;
    std::map<std::uint64_t, Record> records_;
    std::vector<std::uint64_t> order_;
    std::map<int, bool> devices_;
    bool accelerateFinal_ = false, cancelled_ = false;
};
} // namespace WatermarkCore
