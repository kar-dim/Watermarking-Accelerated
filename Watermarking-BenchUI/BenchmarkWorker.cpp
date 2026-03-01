#include "BenchmarkWorker.hpp"
#include "common_utils.hpp"
#include "WatermarkEngine.hpp"
#include <chrono>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <future>
#include <ratio>
#include <utility>
#include <vector>
#include <WatermarkTypes.hpp>

using namespace CommonUtils;
namespace fs = std::filesystem;

BenchmarkWorker::BenchmarkWorker(const QString& folderPath, int openclDevice, QObject* parent) : QThread(parent), inputFolder(folderPath), deviceIndex(openclDevice) {}

void BenchmarkWorker::run() {
    // initialize watermark environment For OpenCL only: if the specified device is invalid, emit a warning and continue with the default device (index 0)
    const bool validDevice = WatermarkEngine::initializeEnvironment(deviceIndex);
    if (!validDevice)
        emit showWarningDialog("Invalid OpenCL Device", "The specified OpenCL device index was not found or is invalid.\n\nFalling back to default Device 0.\n\nClick OK to continue the benchmark.");

    // check if the input directory which conntains the benchmark images is valid
    fs::path inputDir(inputFolder.toStdString());
    if (!fs::exists(inputDir) || !fs::is_directory(inputDir)) {
        emit errorOccurred("System", "Invalid directory path!");
        emit benchmarkFinished(0.0);
        return;
    }

    // get the image files, if no valid image files are found, emit a finish signal with 0 FPS
    const std::vector<fs::path> validFiles = CommonUtils::getValidImageFiles(inputDir);
    if (validFiles.empty()) {
        emit benchmarkFinished(0.0);
        return;
    }

    // initialize accumulators for the benchmark results and calculate total steps for progress tracking
    const int totalSteps = static_cast<int>(validFiles.size() * pValues.size() * psnrValues.size());
    int currentStep = 0;
    double totalTimeMs = 0.0;
    int totalFrames = 0;
    // initialize with a fixed watermark seed and the first set of parameters (p, psnr)
    auto session = WatermarkEngine::createImageSession(12345, pValues[0], psnrValues[0]);
    // first image load while we set up the session
    std::future<WatermarkEngine::PreloadedHandle> prefetchTask = std::async(std::launch::async, WatermarkEngine::preloadImageFromDisk, validFiles[0].string());
    // more warmup iterations for GPU to ensure accurate benchmarking, OpenCL needs less than CUDA
    const int benchmarkIterations = WatermarkEngine::isGpuBackend() ? (WatermarkEngine::isOpenCLBackend() ? 20 : 100) : 5;

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
                    WatermarkEngine::updateSessionParams(session.get(), p, psnr);
                    // initialization warmup: run the embedding once to ensure all memory allocations or arrayfire/openmp init are done before we start measuring time
                    WatermarkEngine::embedImage(session.get(), MaskMethod::ME);
                    // warmup for cpu/gpu to force P0 state (gpu needs much more warmup due to driver)
                    auto start = std::chrono::high_resolution_clock::now();
                    for (int k = 0; k < benchmarkIterations; k++)
                        WatermarkEngine::embedImage(session.get(), MaskMethod::ME);
                    auto end = std::chrono::high_resolution_clock::now();
                    // calculate average time per embedding and FPS for the current parameters, and accumulate total time and frames for final score calculation at the end of the batch
                    const double totalBatchTimeMs = std::chrono::duration<double, std::milli>(end - start).count();
                    const double avgEmbedTimeMs = totalBatchTimeMs / static_cast<double>(benchmarkIterations);
                    const double currentFps = 1000.0 / avgEmbedTimeMs;
                    totalTimeMs += totalBatchTimeMs;
                    totalFrames += benchmarkIterations;
                    // GUI: convert the current watermarked image to a QImage format (interleaved RGB, transposed row-wise) for display in the GUI
                    // and emit current FPS and time for this specific frame and step completion to the GUI for display
                    emit resultReady(convertToQtFormat(session.get()), p, static_cast<int>(psnr), avgEmbedTimeMs, currentFps, currentFileName);
                    emit progressUpdated(++currentStep, totalSteps);
                }
            }
        } catch (const std::exception& e) {
            emit errorOccurred(currentFileName, QString::fromStdString(cleanError(e.what())));
            if (i + 1 < validFiles.size())
                prefetchTask = std::async(std::launch::async, WatermarkEngine::preloadImageFromDisk, validFiles[i + 1].string());
        }
    }

    // calculate final score (average FPS across all frames)
    emit benchmarkFinished(totalTimeMs == 0.0 ? 0.0 : (totalFrames * 1000.0) / totalTimeMs);
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
