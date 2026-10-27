#pragma once
#include "../Watermarking-Core/VideoBatchScheduler.hpp"
#include "VideoBatch.hpp"
#include <atomic>
#include <chrono>
#include <cstdint>
#include <future>
#include <gtest/gtest.h>
#include <mutex>
#include <set>
#include <stdexcept>
#include <vector>

/*!
 * \brief GPU-free checks for concurrent claims, retries and final-file publication
 * \author Dimitris Karatzas
 */
namespace {
using WatermarkCore::VideoBatchJob;
using WatermarkCore::VideoBatchScheduler;
using WatermarkCore::VideoBatchState;
using WatermarkCore::VideoBatchUpdate;

TEST(VideoBatchSchedulerTest, ConcurrentWorkersClaimEachLogicalFileOnce) {
    std::vector<VideoBatchJob> jobs;
    for (std::uint64_t id = 1; id <= 40; ++id)
        jobs.push_back({id, "input", "output"});
    VideoBatchScheduler queue(jobs, {0, 1, 2}, false);
    std::mutex mutex;
    std::set<std::uint64_t> completed;
    std::atomic<int> publications{0};
    std::vector<std::future<void>> workers;
    for (int device = 0; device < 3; ++device)
        workers.push_back(std::async(std::launch::async, [&, device] {
            while (const auto claim = queue.claim(device)) {
                VideoBatchUpdate result;
                result.id = claim->id;
                EXPECT_TRUE(queue.complete(*claim, result, [&] { ++publications; }));
                const std::lock_guard lock(mutex);
                EXPECT_TRUE(completed.insert(claim->id).second);
            }
        }));
    for (auto& worker : workers)
        worker.get();
    EXPECT_EQ(publications, 40);
    EXPECT_EQ(completed.size(), 40);
    for (const auto& view : queue.snapshot()) {
        EXPECT_EQ(view.state, VideoBatchState::Completed);
        EXPECT_TRUE(view.activeDevices.empty());
    }
}

TEST(VideoBatchSchedulerTest, RetriesOncePerDeviceAndRetiresLostWorkers) {
    VideoBatchScheduler queue(
        {
            {1, "in", "out"},
            {2, "in", "out"}
    },
        {0, 1}, false);
    const auto first = queue.claim(0);
    ASSERT_TRUE(first);
    queue.fail(*first, "device lost", true, true);
    EXPECT_FALSE(queue.claim(0));
    const auto retry = queue.claim(1);
    ASSERT_TRUE(retry);
    EXPECT_EQ(retry->id, 1);
    EXPECT_EQ(retry->attempt, 2);
    queue.fail(*retry, "unsupported format", true);
    const auto next = queue.claim(1);
    ASSERT_TRUE(next);
    EXPECT_EQ(next->id, 2);
    EXPECT_TRUE(queue.complete(*next, {}, [] {}));
    EXPECT_FALSE(queue.claim(1));
    const auto views = queue.snapshot();
    EXPECT_EQ(views[0].state, VideoBatchState::Failed);
    EXPECT_EQ(views[1].state, VideoBatchState::Completed);
}

TEST(VideoBatchSchedulerTest, CorruptFilesDoNotRetryAndUnavailableDevicesFailPendingJobs) {
    VideoBatchScheduler queue(
        {
            {1, "in", "out"},
            {2, "in", "out"}
    },
        {0, 1}, false);
    const auto claim = queue.claim(0);
    ASSERT_TRUE(claim);
    queue.fail(*claim, "bad input", false);
    EXPECT_EQ(queue.snapshot()[0].state, VideoBatchState::Failed);
    queue.retire(0, "unavailable");
    queue.retire(1, "unavailable");
    EXPECT_EQ(queue.snapshot()[1].state, VideoBatchState::Failed);
    EXPECT_FALSE(queue.claim(0));
}

TEST(VideoBatchSchedulerTest, RetiringABusyWorkerReleasesItsClaimForAnotherDevice) {
    VideoBatchScheduler queue(
        {
            {1, "in", "out"},
            {2, "in", "out"}
    },
        {0, 1, 2}, false);
    const auto abandoned = queue.claim(0);
    const auto healthy = queue.claim(1);
    ASSERT_TRUE(abandoned);
    ASSERT_TRUE(healthy);
    queue.retire(0, "worker exited");
    EXPECT_TRUE(abandoned->stop.stop_requested());
    const auto retry = queue.claim(2);
    ASSERT_TRUE(retry);
    EXPECT_EQ(retry->id, abandoned->id);
    EXPECT_TRUE(queue.complete(*retry, {}, [] {}));
    EXPECT_TRUE(queue.complete(*healthy, {}, [] {}));
    EXPECT_FALSE(queue.claim(0));
    EXPECT_FALSE(queue.claim(1));
    for (const auto& view : queue.snapshot()) {
        EXPECT_EQ(view.state, VideoBatchState::Completed);
        EXPECT_TRUE(view.activeDevices.empty());
    }
}

TEST(VideoBatchSchedulerTest, PublicationFailureLeavesTheOtherAttemptEligible) {
    VideoBatchScheduler queue(
        {
            {1, "in", "out"}
    },
        {0, 1, 2}, true);
    const auto primary = queue.claim(0);
    const auto backup = queue.claim(1);
    ASSERT_TRUE(primary);
    ASSERT_TRUE(backup);
    EXPECT_TRUE(backup->backup);
    auto third = std::async(std::launch::async, [&] { return queue.claim(2); });
    EXPECT_EQ(third.wait_for(std::chrono::milliseconds(30)), std::future_status::timeout);
    EXPECT_THROW(queue.complete(*backup, {}, [] { throw std::runtime_error("publication failed"); }), std::runtime_error);
    queue.fail(*backup, "publication failed", false);
    EXPECT_FALSE(primary->stop.stop_requested());
    EXPECT_EQ(third.wait_for(std::chrono::milliseconds(30)), std::future_status::timeout);
    EXPECT_TRUE(queue.complete(*primary, {}, [] {}));
    EXPECT_FALSE(third.get());
    EXPECT_EQ(queue.snapshot()[0].state, VideoBatchState::Completed);
}

TEST(VideoBatchSchedulerTest, SimultaneousFinishesPublishOneWinnerAndStopTheLoser) {
    VideoBatchScheduler queue(
        {
            {1, "in", "out"}
    },
        {0, 1}, true);
    const auto primary = queue.claim(0), backup = queue.claim(1);
    ASSERT_TRUE(primary);
    ASSERT_TRUE(backup);
    std::promise<void> start;
    auto ready = start.get_future().share();
    std::atomic<int> accepted{0}, published{0};
    auto finish = [&](const VideoBatchScheduler::Claim& claim) {
        ready.wait();
        accepted += queue.complete(claim, {}, [&] { ++published; });
    };
    auto one = std::async(std::launch::async, finish, *primary);
    auto two = std::async(std::launch::async, finish, *backup);
    start.set_value();
    one.get();
    two.get();
    EXPECT_EQ(accepted, 1);
    EXPECT_EQ(published, 1);
    EXPECT_TRUE(primary->stop.stop_requested());
    EXPECT_TRUE(backup->stop.stop_requested());
    EXPECT_TRUE(queue.snapshot()[0].activeDevices.empty());
}

TEST(VideoBatchSchedulerTest, SkipCancelsBothAttemptsAndBatchCancelStopsPendingClaims) {
    VideoBatchScheduler queue(
        {
            {1, "in", "out"}
    },
        {0, 1}, true);
    const auto primary = queue.claim(0), backup = queue.claim(1);
    ASSERT_TRUE(primary);
    ASSERT_TRUE(backup);
    EXPECT_FALSE(queue.skip(2));
    EXPECT_TRUE(queue.skip(1));
    EXPECT_TRUE(primary->stop.stop_requested());
    EXPECT_TRUE(backup->stop.stop_requested());
    queue.fail(*primary, "cancelled", true);
    queue.fail(*backup, "cancelled", true);
    EXPECT_EQ(queue.snapshot()[0].state, VideoBatchState::Skipped);
    VideoBatchScheduler cancelled(
        {
            {1, "in", "out"},
            {2, "in", "out"}
    },
        {0, 1}, false);
    const auto active = cancelled.claim(0);
    ASSERT_TRUE(active);
    cancelled.cancel();
    EXPECT_TRUE(active->stop.stop_requested());
    cancelled.fail(*active, "cancelled", true);
    EXPECT_FALSE(cancelled.claim(1));
    for (const auto& view : cancelled.snapshot())
        EXPECT_EQ(view.state, VideoBatchState::Cancelled);
}
} // namespace
