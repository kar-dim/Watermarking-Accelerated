#include "BenchmarkWorker.hpp"
#include "ImagePreview.hpp"
#include "WatermarkCore.hpp"
#include <array>
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
#include <utility>
#include <vector>

using namespace WatermarkCore;
namespace fs = std::filesystem;

namespace {
double calculateCV(const std::vector<double>& times) {
    if (times.empty())
        return 0.0;
    const double mean = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    const double sqSum = std::inner_product(times.begin(), times.end(), times.begin(), 0.0);
    const double variance = sqSum / times.size() - mean * mean;
    const double stdev = variance > 0.0 ? std::sqrt(variance) : 0.0;
    return mean > 0.0 ? stdev / mean : 0.0;
}
} // namespace

/*!
 *  \brief  Implementation of the benchmarking background thread
 *  \author Dimitris Karatzas
 */

BenchmarkWorker::BenchmarkWorker(const int deviceIndex, QObject* parent) : QThread(parent), deviceIndex(deviceIndex) {}

void BenchmarkWorker::run() {
    try {
        // Extract bundled resource samples so Core APIs load them from disk
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
            inputFolder = tempDir.path();
        } else {
            emit benchmarkFinished(0.0, 0.0, 0);
            return;
        }

        // Select GPU before creating any buffers
        initializeEnvironment(deviceIndex);
        const DeviceMemoryCleanup cleanup(deviceIndex);
        // Pre-compile OpenCL kernels before the timed loop (no-op on other backends)
        buildOpenCLKernels();
        if (isInterruptionRequested()) {
            emit benchmarkCanceled();
            return;
        }

        fs::path inputDir(inputFolder.toStdString());
        if (!fs::exists(inputDir) || !fs::is_directory(inputDir)) {
            emit benchmarkFinished(0.0, 0.0, 0);
            return;
        }

        // get the image files, if no valid image files are found, emit a finish signal with 0 FPS
        const std::vector<fs::path> validFiles = WatermarkCore::getValidImageFiles(inputDir);
        if (validFiles.empty()) {
            emit benchmarkFinished(0.0, 0.0, 0);
            return;
        }

        // initialize accumulators for the benchmark results and calculate total steps for progress tracking
        const int totalSteps = static_cast<int>(validFiles.size() * pValues.size() * psnrValues.size());
        int currentStep = 0;
        double sumLogEmbedFps = 0.0;
        double sumLogDetectFps = 0.0;
        int cellCount = 0;
        // initialize with a fixed watermark seed and the first set of parameters (p, psnr)
        auto session = createImageSession("password12345", pValues[0], psnrValues[0]);
        // first image load while we set up the session
        const int selectedDevice = getCurrentDeviceIndex();
        std::future<PreloadedHandle> prefetchTask = std::async(std::launch::async, preloadImageFromDisk, validFiles[0].string(), selectedDevice, false);

        // reserve the vector which would hold the benchmark times per image once
        constexpr int maxIterations = 300;
        std::vector<double> samples;
        samples.reserve(maxIterations);
        // GUI holds latest preview until next arrival, ping-pong buffer reused on subsequent step
        std::array<QImage, 2> previews;

        // auto-tuned performance lambda
        auto measurePerformance = [&](auto&& task) {
            // constants and initialize
            constexpr int minWorkMs = 50; // minimum 50ms of work (for fast devices or small images)
            constexpr double targetCv = 0.10;
            constexpr double maxTimeBudgetMs = 250.0;
            samples.clear();

            int minIterations = 5;
            // warmup
            auto t1 = std::chrono::high_resolution_clock::now();
            const float result = task();
            auto t2 = std::chrono::high_resolution_clock::now();
            const double firstRunMs = std::chrono::duration<double, std::milli>(t2 - t1).count();
            // optimization: if the device is very slow, allow only one or two loops in order to finish quicker
            if (firstRunMs >= maxTimeBudgetMs)
                minIterations = 1;
            else if (firstRunMs > (maxTimeBudgetMs / 2.0))
                minIterations = 2;

            // start auto-tuned bench
            double totalTime = 0.0;
            while (samples.size() < maxIterations) {
                if (isInterruptionRequested())
                    break;
                if (samples.size() >= minIterations) {
                    if (totalTime > maxTimeBudgetMs)
                        break;
                    if (totalTime > minWorkMs && calculateCV(samples) < targetCv)
                        break;
                }
                // run next frame
                t1 = std::chrono::high_resolution_clock::now();
                task();
                t2 = std::chrono::high_resolution_clock::now();
                // accumulate total time
                const double duration = std::chrono::duration<double, std::milli>(t2 - t1).count();
                totalTime += duration;
                samples.push_back(duration);
            }

            // calculate stats based on the samples we actually ran
            const int iterations = static_cast<int>(samples.size());
            const double avgMs = (iterations > 0) ? (totalTime / iterations) : 0.0;
            const double fps = (avgMs > 0.0) ? (1000.0 / avgMs) : 0.0;
            return std::make_tuple(avgMs, fps, result);
        };

        // main loop
        // we check periodically if the thread is interrupted to exit gracefully
        for (size_t i = 0; i < validFiles.size(); i++) {
            if (isInterruptionRequested()) {
                emit benchmarkCanceled();
                return;
            }
            const QString currentFileName = QString::fromStdString(validFiles[i].filename().string());
            try {
                // get the current image and prefetch next image in the background
                auto currentImage = prefetchTask.get();
                if (i + 1 < validFiles.size())
                    prefetchTask = std::async(std::launch::async, preloadImageFromDisk, validFiles[i + 1].string(), selectedDevice, false);
                // lazily initialize the watermark session based on the current image dimensions
                bindPreloadedImage(session.get(), std::move(currentImage));
                // the prefetch shares the device stream/queue and the CPU cores with the timed work, it must finish before measuring
                if (prefetchTask.valid())
                    prefetchTask.wait();

                // for all combinations
                for (int p : pValues) {
                    for (float psnr : psnrValues) {
                        if (isInterruptionRequested()) {
                            emit benchmarkCanceled();
                            return;
                        }
                        // p change rebuilds the backend, PSNR change updates only the cached embedding strength
                        updateSessionParams(session.get(), p, psnr);
                        // EMBED BENCHMARK
                        auto [avgEmbedMs, embedFps, dummy] = measurePerformance([&]() {
                            embedImage(session.get());
                            finish();
                            return 0.0f;
                        });
                        if (isInterruptionRequested()) {
                            emit benchmarkCanceled();
                            return;
                        }
                        // necessary uint8 to float for detection
                        prepareDetectionImage(session.get());
                        // DETECT BENCHMARK
                        auto [avgDetectMs, detectFps, currentCorrelation] = measurePerformance([&]() { return detectEmbeddedBuffer(session.get()); });
                        if (isInterruptionRequested()) {
                            emit benchmarkCanceled();
                            return;
                        }
                        // accumulate log(fps) per cell for geometric mean score at the end
                        if (embedFps > 0.0 && detectFps > 0.0) {
                            sumLogEmbedFps += std::log(embedFps);
                            sumLogDetectFps += std::log(detectFps);
                            cellCount++;
                        }
                        // Convert and scale outside timed sections before queueing across threads
                        QImage& preview = previews[currentStep % previews.size()];
                        updatePreviewFromSession(session.get(), preview);
                        const int maxWidth = previewMaxWidth.load(std::memory_order_relaxed);
                        const int maxHeight = previewMaxHeight.load(std::memory_order_relaxed);
                        const QImage displayImg =
                            (preview.width() > maxWidth || preview.height() > maxHeight) ? preview.scaled(maxWidth, maxHeight, Qt::KeepAspectRatio, Qt::SmoothTransformation) : preview;
                        emit resultReady(displayImg, p, psnr, avgEmbedMs, avgDetectMs, embedFps, detectFps, currentFileName, currentCorrelation);
                        emit progressUpdated(++currentStep, totalSteps);
                    }
                }
            } catch (const std::exception& error) {
                if (isInterruptionRequested())
                    emit benchmarkCanceled();
                else
                    emit benchmarkFinished(0.0, 0.0, 0, QString::fromUtf8(error.what()));
                return;
            }
        }
        if (isInterruptionRequested()) {
            emit benchmarkCanceled();
            return;
        }
        // calculate final score: geometric mean of cell FPS across all (image * p * psnr) combinations
        // geometric mean -> equal logarithmic weight to each cell
        const double finalEmbedFps = (cellCount > 0) ? std::exp(sumLogEmbedFps / cellCount) : 0.0;
        const double finalDetectFps = (cellCount > 0) ? std::exp(sumLogDetectFps / cellCount) : 0.0;
        const int finalScore = static_cast<int>(std::round(std::sqrt(finalEmbedFps * finalDetectFps) * 10.0));
        emit benchmarkFinished(finalEmbedFps, finalDetectFps, finalScore);
    } catch (const std::exception& error) {
        if (isInterruptionRequested())
            emit benchmarkCanceled();
        else
            emit benchmarkFinished(0.0, 0.0, 0, QString::fromUtf8(error.what()));
    } catch (...) { emit benchmarkFinished(0.0, 0.0, 0); }
}
