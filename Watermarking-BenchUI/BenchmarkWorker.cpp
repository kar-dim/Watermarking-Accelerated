#include "BenchmarkWorker.hpp"
#include "common_utils.hpp"
#include "WatermarkCore.hpp"
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <future>
#include <numeric>
#include <QDir>
#include <QFile>
#include <QImage>
#include <QIODevice>
#include <QString>
#include <QThread>
#include <ratio>
#include <tuple>
#include <utility>
#include <vector>
#include <WatermarkTypes.hpp>

using namespace CommonUtils;
using namespace WatermarkCore;
namespace fs = std::filesystem;

BenchmarkWorker::BenchmarkWorker(const int openclDevice, QObject* parent) : QThread(parent), deviceIndex(openclDevice) {}

void BenchmarkWorker::run() {
    // read the image files (which are embedded in the executable) and write them to temporary folder
    if (tempDir.isValid()) {
        const QDir resourceDir(":/samples");
        for (const QString& fileName : resourceDir.entryList(QDir::Files)) {
            QFile resFile(":/samples/" + fileName);
            if (resFile.open(QIODevice::ReadOnly)) {
                QFile outFile(tempDir.path() + "/" + fileName);
                if (outFile.open(QIODevice::WriteOnly))
                    outFile.write(resFile.readAll());
            }
        }
        // we can load the images from this temp folder
        inputFolder = tempDir.path();
    } else {
        emit benchmarkFinished(0.0, 0.0, 0);
        return;
    }

    // initialize watermark environment (device index is used for OpenCL only)
    initializeEnvironment(deviceIndex);

    // check if the input directory which conntains the benchmark images is valid
    fs::path inputDir(inputFolder.toStdString());
    if (!fs::exists(inputDir) || !fs::is_directory(inputDir)) {
        emit benchmarkFinished(0.0, 0.0, 0);
        return;
    }

    // get the image files, if no valid image files are found, emit a finish signal with 0 FPS
    const std::vector<fs::path> validFiles = CommonUtils::getValidImageFiles(inputDir);
    if (validFiles.empty()) {
        emit benchmarkFinished(0.0, 0.0, 0);
        return;
    }

    // initialize accumulators for the benchmark results and calculate total steps for progress tracking
    const int totalSteps = static_cast<int>(validFiles.size() * pValues.size() * psnrValues.size());
    int currentStep = 0;
    int totalEmbedFrames = 0;
    int totalDetectFrames = 0;
    double totalEmbedTimeMs = 0.0;
    double totalDetectTimeMs = 0.0;
    // initialize with a fixed watermark seed and the first set of parameters (p, psnr)
    auto session = createImageSession("password12345", pValues[0], psnrValues[0]);
    // first image load while we set up the session
    std::future<PreloadedHandle> prefetchTask = std::async(std::launch::async, preloadImageFromDisk, validFiles[0].string());

    // auto-tuned performance lambda
    auto measurePerformance = [&](auto&& task) {
        // constants
        constexpr int maxIterations = 300;
        constexpr int minWorkMs = 50; // minimum 50ms of work (for fast devices or small images)
        constexpr double targetCv = 0.10;
        constexpr double maxTimeBudgetMs = 250.0;

        int minIterations = 5;
        // warmup
        auto t1 = std::chrono::high_resolution_clock::now();
        float lastResult = task();
        auto t2 = std::chrono::high_resolution_clock::now();
        const double firstRunMs = std::chrono::duration<double, std::milli>(t2 - t1).count();
        // optimization: if the device is very slow, allow only one or two loops in order to finish quicker
        if (firstRunMs >= maxTimeBudgetMs)
            minIterations = 1;
        else if (firstRunMs > (maxTimeBudgetMs / 2.0))
            minIterations = 2;

        std::vector<double> samples;
        samples.reserve(maxIterations);
        // start auto-tuned bench
        while (samples.size() < maxIterations) {
            if (isInterruptionRequested())
                break;
            const double totalTime = std::accumulate(samples.begin(), samples.end(), 0.0);
            if (samples.size() >= minIterations) {
                if (totalTime > maxTimeBudgetMs)
                    break;
                if (totalTime > minWorkMs && calculateCV(samples) < targetCv)
                    break;
            }
            // Run next frame
            t1 = std::chrono::high_resolution_clock::now();
            lastResult = task();
            t2 = std::chrono::high_resolution_clock::now();
            samples.push_back(std::chrono::duration<double, std::milli>(t2 - t1).count());
        }

        // calculate stats based on the samples we actually ran
        const int iterations = static_cast<int>(samples.size());
        const double totalMs = std::accumulate(samples.begin(), samples.end(), 0.0);
        const double avgMs = (iterations > 0) ? (totalMs / iterations) : 0.0;
        const double fps = (avgMs > 0.0) ? (1000.0 / avgMs) : 0.0;
        return std::make_tuple(totalMs, avgMs, fps, lastResult, iterations);
    };

    // main loop
    // we check periodically if the thread is interrupted to exit gracefully
    for (size_t i = 0; i < validFiles.size(); i++) {
        if (QThread::currentThread()->isInterruptionRequested())
            return;
        const QString currentFileName = QString::fromStdString(validFiles[i].filename().string());
        try {
            // get the current image and prefetch next image in the background
            auto currentImage = prefetchTask.get();
            if (i + 1 < validFiles.size())
                prefetchTask = std::async(std::launch::async, preloadImageFromDisk, validFiles[i + 1].string());
            // lazily initialize the watermark session based on the current image dimensions
            bindPreloadedImage(session.get(), std::move(currentImage));

            // for all combinations
            for (int p : pValues) {
                for (float psnr : psnrValues) {
                    if (QThread::currentThread()->isInterruptionRequested())
                        return;
                    // update p and psnr in the session, this will trigger all necessary internal recalculations in the engine (e.g. random noise generation)
                    updateSessionParams(session.get(), p, psnr);
                    // EMBED BENCHMARK
                    auto [tEmbed, avgEmbedMs, embedFps, dummy, iterEmbed] = measurePerformance([&]() {
                        embedImage(session.get(), MaskMethod::ME);
                        return 0.0f;
                    });
                    if (QThread::currentThread()->isInterruptionRequested())
                        return;
                    // necessary uint8 to float for detection
                    prepareDetectionImage(session.get(), MaskMethod::ME);
                    // DETECT BENCHMARK
                    auto [tDetect, avgDetectMs, detectFps, currentCorrelation, iterDetect] = measurePerformance([&]() { return detectEmbeddedBuffer(session.get(), MaskMethod::ME); });
                    if (QThread::currentThread()->isInterruptionRequested())
                        return;
                    // accumulate total times and frames for final score calculation at the end of the benchmark
                    totalEmbedTimeMs += tEmbed;
                    totalDetectTimeMs += tDetect;
                    totalEmbedFrames += iterEmbed;
                    totalDetectFrames += iterDetect;
                    // GUI: convert the current watermarked image to a QImage format (interleaved RGB, transposed row-wise) for display in the GUI
                    // and emit current FPS and time for this specific frame and step completion to the GUI for display
                    emit resultReady(convertToQtFormat(session.get()), p, psnr, avgEmbedMs, avgDetectMs, embedFps, detectFps, currentFileName, currentCorrelation);
                    emit progressUpdated(++currentStep, totalSteps);
                }
            }
        } catch (...) {
            // if at least one file is not benchmarked, we consider it failure
            emit benchmarkFinished(0.0, 0.0, 0);
            return;
        }
    }
    // calculate final score (geomean average FPS of both pipelines)
    const double finalEmbedFps = (totalEmbedTimeMs == 0.0) ? 0.0 : (totalEmbedFrames * 1000.0) / totalEmbedTimeMs;
    const double finalDetectFps = (totalDetectTimeMs == 0.0) ? 0.0 : (totalDetectFrames * 1000.0) / totalDetectTimeMs;
    const int finalScore = static_cast<int>(std::round(std::sqrt(finalEmbedFps * finalDetectFps) * 100.0));
    emit benchmarkFinished(finalEmbedFps, finalDetectFps, finalScore);
}

// simple format conversion from the raw pixel data of the watermark session (column-major, planar) to a QImage (row-major, interleaved) for display in the GUI
// optimized with OpenMP and cache locality in mind, but the timer does not count this, it is only used for display purposes
QImage BenchmarkWorker::convertToQtFormat(ImageSession* session) const {
    int rows = 0, cols = 0, channels = 0;
    const uint8_t* rawData = getSessionPixelData(session, cols, rows, channels);
    QImage displayImage(cols, rows, channels == 3 ? QImage::Format_RGB888 : QImage::Format_Grayscale8);
    // rgb
    if (channels == 3) {
        const uint8_t* red = rawData;
        const uint8_t* green = rawData + (cols * rows);
        const uint8_t* blue = rawData + 2 * (cols * rows);
#pragma omp parallel for
        for (int y = 0; y < rows; y++) {
            uint8_t* scanline = displayImage.scanLine(y);
            int readIdx = y;
            for (int x = 0; x < cols; x++) {
                scanline[0] = red[readIdx];
                scanline[1] = green[readIdx];
                scanline[2] = blue[readIdx];
                scanline += 3;
                readIdx += rows;
            }
        }
    } else { // grayscale
#pragma omp parallel for
        for (int y = 0; y < rows; y++) {
            uint8_t* scanline = displayImage.scanLine(y);
            int readIdx = y;
            for (int x = 0; x < cols; x++) {
                *scanline++ = rawData[readIdx];
                readIdx += rows;
            }
        }
    }

    return displayImage;
}
