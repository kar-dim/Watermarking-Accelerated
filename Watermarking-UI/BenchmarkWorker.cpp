#include "BenchmarkWorker.hpp"
#include "common_utils.hpp"
#include "ImagePreview.hpp"
#include "WatermarkCore.hpp"
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <future>
#include <QDir>
#include <QFile>
#include <QImage>
#include <QIODevice>
#include <QString>
#include <QThread>
#include <ratio>
#include <utility>
#include <vector>
#include <WatermarkTypes.hpp>

using namespace CommonUtils;
using namespace WatermarkCore;
namespace fs = std::filesystem;

/*!
 *  \brief  Implementation of the benchmarking background thread
 *  \author Dimitris Karatzas
 */

// Initialize benchmark worker with selected compute device index
BenchmarkWorker::BenchmarkWorker(const int deviceIndex, QObject* parent) : QThread(parent), deviceIndex(deviceIndex) {}

void BenchmarkWorker::run() {
    try {
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

        // Select the GPU for this worker thread before creating any buffers
        initializeEnvironment(deviceIndex);
        buildOpenCLKernels(); // NO-OP for non-OpenCL backends, but we call it here to ensure any necessary pre-compilation is done before the benchmark loop starts
        if (isInterruptionRequested()) {
            emit benchmarkCanceled();
            return;
        }

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
                            embedImage(session.get(), MaskMethod::ME);
                            finish();
                            return 0.0f;
                        });
                        if (isInterruptionRequested()) {
                            emit benchmarkCanceled();
                            return;
                        }
                        // necessary uint8 to float for detection
                        prepareDetectionImage(session.get(), MaskMethod::ME);
                        // DETECT BENCHMARK
                        auto [avgDetectMs, detectFps, currentCorrelation] = measurePerformance([&]() { return detectEmbeddedBuffer(session.get(), MaskMethod::ME); });
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
                        // GUI: convert the current watermarked image to a QImage format (interleaved RGB, transposed row-wise) for display in the GUI
                        // and emit current FPS and time for this specific frame and step completion to the GUI for display
                        emit resultReady(imagePreviewFromSession(session.get()), p, psnr, avgEmbedMs, avgDetectMs, embedFps, detectFps, currentFileName, currentCorrelation);
                        emit progressUpdated(++currentStep, totalSteps);
                    }
                }
            } catch (...) {
                // if at least one file is not benchmarked, we consider it failure
                if (isInterruptionRequested())
                    emit benchmarkCanceled();
                else
                    emit benchmarkFinished(0.0, 0.0, 0);
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
    } catch (...) {
        if (isInterruptionRequested())
            emit benchmarkCanceled();
        else
            emit benchmarkFinished(0.0, 0.0, 0);
    }
}
