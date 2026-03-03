#include "BenchmarkWorker.hpp"
#include "common_utils.hpp"
#include "WatermarkEngine.hpp"
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <future>
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
namespace fs = std::filesystem;

BenchmarkWorker::BenchmarkWorker(int openclDevice, QObject* parent) : QThread(parent), deviceIndex(openclDevice) {
    // read the image files (which are embedded in the executable) and write them to temporary folder
    if (tempDir.isValid()) {
        const QDir resourceDir(":/samples");
        for (const QString& fileName : resourceDir.entryList(QDir::Files)) {
            QFile resFile(":/samples/" + fileName);
            if (resFile.open(QIODevice::ReadOnly)) {
                const QString outPath = tempDir.path() + "/" + fileName;
                QFile outFile(outPath);
                if (outFile.open(QIODevice::WriteOnly))
                    outFile.write(resFile.readAll());
            }
        }
        // we can load the images from this temp folder
        inputFolder = tempDir.path();
    } else {
        emit errorOccurred("System", "Failed to create temporary directory for benchmark samples.");
    }
}

void BenchmarkWorker::run() {
    // initialize watermark environment For OpenCL only
    WatermarkEngine::initializeEnvironment(deviceIndex);

    // check if the input directory which conntains the benchmark images is valid
    fs::path inputDir(inputFolder.toStdString());
    if (!fs::exists(inputDir) || !fs::is_directory(inputDir)) {
        emit errorOccurred("System", "Invalid directory path!");
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
    double totalEmbedTimeMs = 0.0;
    double totalDetectTimeMs = 0.0;
    int totalFrames = 0;
    // initialize with a fixed watermark seed and the first set of parameters (p, psnr)
    auto session = WatermarkEngine::createImageSession(12345, pValues[0], psnrValues[0]);
    // first image load while we set up the session
    std::future<WatermarkEngine::PreloadedHandle> prefetchTask = std::async(std::launch::async, WatermarkEngine::preloadImageFromDisk, validFiles[0].string());
    // more warmup iterations for GPU to ensure accurate benchmarking, OpenCL needs less than CUDA
    const int benchmarkIterations = WatermarkEngine::isGpuBackend() ? (WatermarkEngine::isOpenCLBackend() ? 20 : 100) : 5;

    // helper lambda to measure performance of a given task (embedding or detection)
    auto measurePerformance = [&](auto&& task) {
        task(); // initialization warmup: run the task once to ensure all memory allocations or arrayfire/openmp init are done
        auto start = std::chrono::high_resolution_clock::now();
        float lastResult = 0.0f;
        // benchmark loop
        for (int k = 0; k < benchmarkIterations; k++)
            lastResult = task();
        auto end = std::chrono::high_resolution_clock::now();
        // calculate average time and total time for the current task
        const double totalMs = std::chrono::duration<double, std::milli>(end - start).count();
        const double avgMs = totalMs / benchmarkIterations;
        const double fps = 1000.0 / avgMs;
        return std::make_tuple(totalMs, avgMs, fps, lastResult);
    };

    // main loop
    for (size_t i = 0; i < validFiles.size(); i++) {
        const QString currentFileName = QString::fromStdString(validFiles[i].filename().string());
        try {
            // get the current image and prefetch next image in the background
            auto currentImage = prefetchTask.get();
            if (i + 1 < validFiles.size())
                prefetchTask = std::async(std::launch::async, WatermarkEngine::preloadImageFromDisk, validFiles[i + 1].string());
            // lazily initialize the watermark session based on the current image dimensions
            WatermarkEngine::bindPreloadedImage(session.get(), std::move(currentImage));

            // for all combinations
            for (int p : pValues) {
                for (float psnr : psnrValues) {
                    // update p and psnr in the session, this will trigger all necessary internal recalculations in the engine (e.g. random noise generation)
                    WatermarkEngine::updateSessionParams(session.get(), p, psnr);
                    // EMBED BENCHMARK
                    auto [totalEmbedMs, avgEmbedMs, embedFps, dummy] = measurePerformance([&]() {
                        WatermarkEngine::embedImage(session.get(), MaskMethod::ME);
                        return 0.0f;
                    });
                    // DETECT BENCHMARK
                    // necessary uint8 to float for detection
                    WatermarkEngine::prepareDetectionImage(session.get(), MaskMethod::ME);
                    auto [totalDetectMs, avgDetectMs, detectFps, currentCorrelation] = measurePerformance([&]() { return WatermarkEngine::detectEmbeddedBuffer(session.get(), MaskMethod::ME); });
                    // accumulate total times and frames for final score calculation at the end of the benchmark
                    totalEmbedTimeMs += totalEmbedMs;
                    totalDetectTimeMs += totalDetectMs;
                    totalFrames += benchmarkIterations;
                    // GUI: convert the current watermarked image to a QImage format (interleaved RGB, transposed row-wise) for display in the GUI
                    // and emit current FPS and time for this specific frame and step completion to the GUI for display
                    emit resultReady(convertToQtFormat(session.get()), p, psnr, avgEmbedMs, avgDetectMs, embedFps, detectFps, currentFileName, currentCorrelation);
                    emit progressUpdated(++currentStep, totalSteps);
                }
            }
        } catch (const std::exception& e) {
            emit errorOccurred(currentFileName, QString::fromStdString(cleanError(e.what())));
            if (i + 1 < validFiles.size())
                prefetchTask = std::async(std::launch::async, WatermarkEngine::preloadImageFromDisk, validFiles[i + 1].string());
        }
    }

    // calculate final score (geomean average FPS of both pipelines)
    const double finalEmbedFps = (totalEmbedTimeMs == 0.0) ? 0.0 : (totalFrames * 1000.0) / totalEmbedTimeMs;
    const double finalDetectFps = (totalDetectTimeMs == 0.0) ? 0.0 : (totalFrames * 1000.0) / totalDetectTimeMs;
    const int finalScore = static_cast<int>(std::round(std::sqrt(finalEmbedFps * finalDetectFps) * 100.0));
    emit benchmarkFinished(finalEmbedFps, finalDetectFps, finalScore);
}

// simple format conversion from the raw pixel data of the watermark session (column-major, planar) to a QImage (row-major, interleaved) for display in the GUI
// optimized with OpenMP and cache locality in mind, but the timer does not count this, it is only used for display purposes
QImage BenchmarkWorker::convertToQtFormat(WatermarkEngine::ImageSession* session) const {
    int rows = 0, cols = 0, channels = 0;
    const uint8_t* rawData = WatermarkEngine::getSessionPixelData(session, cols, rows, channels);
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
