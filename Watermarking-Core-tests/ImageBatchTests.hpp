#pragma once
#include "ImageBatch.hpp"
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <numeric>

/*!
 * \brief Image batch ownership, output safety, cancellation and isolated-device parity checks
 * \author Dimitris Karatzas
 */

TEST_F(WatermarkTest, ImageBatchPlannerReservesNamesAndKeepsEveryInput) {
    const auto first = tempDir / "one" / u8"\u03b5\u03b9\u03ba\u03cc\u03bd\u03b1.png";
    const auto second = tempDir / "two" / first.filename();
    fs::create_directories(first.parent_path());
    fs::create_directories(second.parent_path());
    fs::copy_file(colorImage, first);
    fs::copy_file(colorImage, second);
    const auto text = [](const fs::path& path) {
        const auto value = path.u8string();
        return std::string(value.begin(), value.end());
    };
    const auto jobs = prepareImageBatch({text(first), text(second), text(first)}, text(first.parent_path()), false);
    ASSERT_EQ(jobs.size(), 2u);
    EXPECT_NE(jobs[0].outputPath, jobs[1].outputPath);
    EXPECT_NE(jobs[0].outputPath, text(first));
    EXPECT_NE(jobs[1].outputPath, text(first));
    const auto detection = prepareImageBatch({text(first)}, {}, true);
    ASSERT_EQ(detection.size(), 1u);
    EXPECT_TRUE(detection[0].outputPath.empty());
    EXPECT_THROW(prepareImageBatch({text(first)}, {}, false), std::invalid_argument);
}

TEST_F(WatermarkTest, ImageBatchRejectsInvalidSelectionsAndOverlappingOutputs) {
    ImageBatchSettings settings;
    settings.password = defaultPassword;
    const auto jobs = prepareImageBatch({colorImage.string()}, (tempDir / "output").string(), false);
    ImageBatchControl control;
    settings.deviceIndices = {0, 0};
    EXPECT_THROW(runImageBatch(settings, jobs, control), std::invalid_argument);
    settings.deviceIndices = {99999};
    EXPECT_THROW(runImageBatch(settings, jobs, control), std::invalid_argument);
    settings.deviceIndices = {0};
    auto invalid = jobs;
    invalid[0].outputPath = invalid[0].inputPath;
    EXPECT_THROW(runImageBatch(settings, invalid, control), std::invalid_argument);
    invalid = jobs;
    invalid[0].id = 0;
    EXPECT_THROW(runImageBatch(settings, invalid, control), std::invalid_argument);
}

TEST_F(WatermarkTest, ImageBatchMatchesIsolatedDevicesForEmbedAndDetect) {
    const std::vector<fs::path> sources{colorImage, alphaImage, grayImage, "samples/images/720p_exif6.jpg"};
    ImageBatchSettings settings;
    settings.password = defaultPassword;
    settings.p = defaultP;
    settings.psnr = defaultPsnr;
    std::vector<std::string> inputs;
    for (int index = 0; index < 20; ++index) {
        const auto input = tempDir / ("input-" + std::to_string(index) + sources[index % sources.size()].extension().string());
        fs::copy_file(sources[index % sources.size()], input);
        inputs.push_back(input.string());
    }
    const auto bytes = [](const fs::path& path) {
        std::ifstream file(path, std::ios::binary);
        return std::vector<char>(std::istreambuf_iterator<char>{file}, {});
    };
    // Each GPU supplies its own baseline to allow for device-specific rounding
    std::map<int, std::vector<std::vector<char>>> expected;
    std::map<int, std::vector<float>> correlations;
    const auto names = getAvailableDevices();
    for (int device = 0; device < static_cast<int>(names.size()); ++device) {
        settings.deviceIndices = {device};
        const auto jobs = prepareImageBatch(inputs, (tempDir / ("single-" + std::to_string(device))).string(), false);
        ImageBatchControl control;
        const auto result = runImageBatch(settings, jobs, control);
        ASSERT_EQ(result.succeeded, inputs.size());
        ASSERT_EQ(result.failed, 0u);
        for (const auto& job : jobs)
            expected[device].push_back(bytes(job.outputPath));
        settings.detect = true;
        ImageBatchControl detectControl;
        const auto detected = runImageBatch(settings, prepareImageBatch(inputs, {}, true), detectControl);
        ASSERT_EQ(detected.succeeded, inputs.size());
        for (const auto& row : detected.results) {
            ASSERT_TRUE(row.correlation);
            correlations[device].push_back(*row.correlation);
        }
        settings.detect = false;
    }
    settings.deviceIndices.resize(names.size());
    std::iota(settings.deviceIndices.begin(), settings.deviceIndices.end(), 0);
    const auto jobs = prepareImageBatch(inputs, (tempDir / "combined").string(), false);
    ImageBatchControl control;
    const auto result = runImageBatch(settings, jobs, control);
    ASSERT_EQ(result.succeeded, inputs.size());
    EXPECT_EQ(result.failed, 0u);
    std::set<int> used;
    for (std::size_t i = 0; i < jobs.size(); ++i) {
        const auto& row = result.results[i];
        used.insert(row.deviceIndex);
        EXPECT_EQ(bytes(jobs[i].outputPath), expected.at(row.deviceIndex)[i]) << jobs[i].inputPath;
    }
    EXPECT_EQ(used.size(), names.size());
    EXPECT_EQ(std::distance(fs::directory_iterator(tempDir / "combined"), fs::directory_iterator{}), inputs.size());
    settings.detect = true;
    ImageBatchControl detectControl;
    const auto detected = runImageBatch(settings, prepareImageBatch(inputs, {}, true), detectControl);
    ASSERT_EQ(detected.succeeded, inputs.size());
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        ASSERT_TRUE(detected.results[i].correlation);
        EXPECT_EQ(*detected.results[i].correlation, correlations.at(detected.results[i].deviceIndex)[i]);
    }
    initializeEnvironment(0);
}

TEST_F(WatermarkTest, ImageBatchContinuesAfterBadInputAndPreservesExistingOutput) {
    const auto bad = tempDir / "bad.png";
    std::ofstream(bad) << "broken image";
    const auto output = tempDir / "outputs";
    auto jobs = prepareImageBatch({colorImage.string(), bad.string(), grayImage.string()}, output.string(), false);
    fs::create_directories(output);
    std::ofstream(jobs.front().outputPath) << "existing output";
    ImageBatchSettings settings;
    settings.password = defaultPassword;
    ImageBatchControl control;
    const auto result = runImageBatch(settings, jobs, control);
    EXPECT_EQ(result.succeeded, 1u);
    EXPECT_EQ(result.failed, 2u);
    std::ifstream kept(jobs.front().outputPath);
    std::string value;
    std::getline(kept, value);
    EXPECT_EQ(value, "existing output");
    EXPECT_EQ(std::distance(fs::directory_iterator(output), fs::directory_iterator{}), 2);
}

TEST_F(WatermarkTest, ImageBatchCancellationDrainsSavesAndCancelsPendingImages) {
    std::vector<std::string> inputs;
    for (int i = 0; i < 30; ++i) {
        const auto file = tempDir / ("cancel-" + std::to_string(i) + ".png");
        fs::copy_file(colorImage, file);
        inputs.push_back(file.string());
    }
    ImageBatchSettings settings;
    settings.password = defaultPassword;
    for (int device = 0; device < static_cast<int>(getAvailableDevices().size()); ++device)
        settings.deviceIndices.push_back(device);
    ImageBatchControl control;
    struct Cancel : ImageBatchObserver {
        ImageBatchControl& control;
        explicit Cancel(ImageBatchControl& control) : control(control) {}
        void onUpdate(const ImageBatchUpdate& row) override {
            if (row.state == ImageBatchState::Saving)
                control.cancel();
        }
    } cancel(control);
    const auto output = tempDir / "cancel-output";
    const auto result = runImageBatch(settings, prepareImageBatch(inputs, output.string(), false), control, &cancel);
    EXPECT_GT(result.cancelled, 0u);
    EXPECT_EQ(result.failed, 0u);
    EXPECT_EQ(result.succeeded + result.cancelled, inputs.size());
    EXPECT_EQ(std::distance(fs::directory_iterator(output), fs::directory_iterator{}), result.succeeded);
    for (const auto& file : fs::directory_iterator(output))
        EXPECT_FALSE(file.path().filename().string().starts_with(".watermark-"));
}

TEST_F(WatermarkTest, ImageBatchObserverFailureStopsAndJoinsWorkers) {
    ImageBatchSettings settings;
    settings.password = defaultPassword;
    ImageBatchControl control;
    struct Throw : ImageBatchObserver {
        void onUpdate(const ImageBatchUpdate&) override { throw std::runtime_error("observer stopped"); }
    } observer;
    EXPECT_THROW(runImageBatch(settings, prepareImageBatch({colorImage.string()}, (tempDir / "observer-output").string(), false), control, &observer), std::runtime_error);
    EXPECT_TRUE(control.cancelled());
}

TEST_F(WatermarkTest, ImageSaveBudgetAllowsOversizedImagesOneAtATime) {
    ImageSaver saver(3, 1);
    for (std::size_t id = 0; id < 5; ++id) {
        loadImage(session.get(), colorImage.string());
        embedImage(session.get());
        saver.saveExact(session.get(), (tempDir / ("budget-" + std::to_string(id) + ".png")).string(), id);
    }
    const auto results = saver.finish();
    ASSERT_EQ(results.size(), 5u);
    for (const auto& result : results) {
        EXPECT_TRUE(result.error.empty()) << result.error;
        EXPECT_TRUE(fs::exists(tempDir / ("budget-" + std::to_string(result.id) + ".png")));
    }
}
