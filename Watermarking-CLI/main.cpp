#include "common_utils.hpp"
#include "libs/inih/INIReader.h"
#include "WatermarkEngine.hpp"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <format>
#include <future>
#include <iostream>
#include <omp.h>
#include <queue>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <WatermarkTypes.hpp>

using namespace WatermarkEngine;
using namespace CommonUtils;
namespace fs = std::filesystem;

using std::cout;
using std::string;

/*!
 *  \brief  Helper functions for testing the watermark algorithms
 *  \author Dimitris Karatzas
 */
static inline string info(const string& str) { return "\033[38;5;208m" + str + "\033[0m"; }
static inline string err(const string& str) { return "\033[91m" + str + "\033[0m"; }
static inline string success(const string& str) { return "\033[92m" + str + "\033[0m"; }

// batch processing of images in a directory (for both embed and detect)
static int testForImageBatch(const INIReader& inir, const int p, const float psnr, const bool isEmbed) {
    const uint32_t watermarkSeed = inir.GetInteger("global", "watermark_seed", 0);
    checkError(watermarkSeed == 0, "No valid watermark seed specified!");
    const bool showFps = inir.GetBoolean("global", "display_fps", true);

    const fs::path inputDir(inir.Get("image", "path", ""));
    if (!fs::exists(inputDir) || !fs::is_directory(inputDir))
        throw std::runtime_error("Error: Batch path is not a valid directory!");

    // only create the below if we are embedding!
    fs::path outputDir;
    std::queue<std::future<void>> saveTasks;
    std::vector<WatermarkEngine::ExportHandle> exportPool;
    size_t bufferIndex = 0;
    size_t maxParallelSaves = 0;
    if (isEmbed) {
        outputDir = inputDir / "watermark_output";
        fs::create_directories(outputDir);
        maxParallelSaves = omp_get_max_threads();
        // preallocate exactly enough buffers for the max concurrent threads
        for (size_t i = 0; i < maxParallelSaves; i++)
            exportPool.push_back(WatermarkEngine::createReusableExportBuffer());
    }

    // get the valid images of the directory, if no valid image files are found, throw an error
    const std::vector<fs::path> validFiles = getValidImageFiles(inputDir);
    checkError(validFiles.empty(), "No valid image files found in directory!");
    cout << info(std::format("Found {} images. Starting batch {}...\n", validFiles.size(), isEmbed ? "embedding" : "detection"));

    // initialize the watermarking session once and reuse for all images in the batch
    auto session = createImageSession(watermarkSeed, p, psnr);
    int successCount = 0;
    float corr = 0.0f;
    double totalEngineTime = 0.0;

    // start the batch process (begin timer)
    const auto batchStart = std::chrono::high_resolution_clock::now();
    // preload the first image
    std::future<PreloadedHandle> prefetchTask = std::async(std::launch::async, preloadImageFromDisk, validFiles[0].string());
    for (size_t i = 0; i < validFiles.size(); i++) {
        try {
            auto currentImage = prefetchTask.get();
            // spawn background thread to read the next image
            if (i + 1 < validFiles.size())
                prefetchTask = std::async(std::launch::async, preloadImageFromDisk, validFiles[i + 1].string());
            // begin timing the watermark operation
            auto imgStart = std::chrono::high_resolution_clock::now();
            // give the buffer to the watermark engine (may trigger lazy init)
            bindPreloadedImage(session.get(), std::move(currentImage));
            // embed
            if (isEmbed) {
                embedImage(session.get(), MaskMethod::ME);
                // if we have too many active saves, wait for the oldest one to finish
                const fs::path outFile = outputDir / validFiles[i].filename();
                if (saveTasks.size() >= maxParallelSaves) {
                    saveTasks.front().get();
                    saveTasks.pop();
                }
                // get the next available buffer from the pool and do a zero copy allocation into it
                auto* currentBuffer = exportPool[bufferIndex].get();
                exportForSave(session.get(), currentBuffer, MaskMethod::ME);
                // launch the heavy save to disk task in the background and cycle to the next buffer
                saveTasks.push(std::async(std::launch::async, flushToDiskAsync, currentBuffer, outFile.string(), MaskMethod::ME));
                bufferIndex = (bufferIndex + 1) % maxParallelSaves;
            } else { // detect
                corr = detectLoadedImage(session.get(), MaskMethod::ME);
            }
            // stop timing the watermark operation
            auto imgEnd = std::chrono::high_resolution_clock::now();
            totalEngineTime += std::chrono::duration<double>(imgEnd - imgStart).count();

            // print success
            if (isEmbed)
                cout << success(std::format(" [OK] {}\n", validFiles[i].filename().string()));
            else
                cout << success(std::format(" [OK] Correlation: {:.2f}, {}\n", corr, validFiles[i].filename().string()));
            successCount++;

        } catch (const std::exception& e) {
            cout << err(std::format(" [FAILED] {} - Error: {}\n", validFiles[i].filename().string(), cleanError(e.what())));
            // If an async task fails, we must catch it and manually prime the pump for the next iteration
            if (i + 1 < validFiles.size())
                prefetchTask = std::async(std::launch::async, WatermarkEngine::preloadImageFromDisk, validFiles[i + 1].string());
        }
    }
    // finish any pending saves before exiting
    if (isEmbed) {
        while (!saveTasks.empty()) {
            saveTasks.front().get();
            saveTasks.pop();
        }
    }
    // stop batch process (and timer)
    const auto batchEnd = std::chrono::high_resolution_clock::now();
    const double totalBatchTime = std::chrono::duration<double>(batchEnd - batchStart).count();

    // print results summary
    checkError(successCount == 0, "No images were successfully processed. Please check the error messages above.");
    cout << info(std::format("\nBatch complete! Successfully processed {}/{} images.\n", successCount, validFiles.size()));
    cout << info("Total batch time: " + formatExecutionTime(false, totalBatchTime) + "\n");
    cout << info("Average engine performance: " + formatExecutionTime(showFps, totalEngineTime / successCount) + "\n");

    return EXIT_SUCCESS;
}

// single image processing, it loads the image, embeds the watermark, detects it, and optionally saves the watermarked image to disk
static int testForImageSingle(const INIReader& inir, const int p, const float psnr) {
    const string imageFile = inir.Get("image", "path", "NO_IMAGE");
    checkError(imageFile == "NO_IMAGE", "No valid image file specified!");
    const uint32_t watermarkSeed = inir.GetInteger("global", "watermark_seed", 0);
    checkError(watermarkSeed == 0, "No valid watermark seed specified!");
    const bool showFps = inir.GetBoolean("global", "display_fps", true);
    const bool saveToDisk = inir.GetBoolean("image", "save_to_disk", false);
    int loops = inir.GetInteger("image", "benchmark_loops", 5);
    loops = loops <= 0 ? 5 : loops;

    cout << "Each test will be executed " << loops << " times.\n";

    // load watermarking session
    auto s = createImageSession(watermarkSeed, p, psnr);
    const double loadTime = executionTime([&]() { loadImage(s.get(), imageFile); });
    cout << "Time to load image data from disk: " << loadTime << " seconds\n\n";

    // helper lambda for embedding and detection benchmarks
    auto runWatermarkingProcess = [&](MaskMethod method, const string& name) {
        // embed
        const double embedTime = executionTime([&]() { embedImage(s.get(), method); }, loops);
        cout << std::format("Calculation of {} mask (p = {}, PSNR = {}dB)\n{}\n\n", name, p, psnr, formatExecutionTime(showFps, embedTime / loops));
        // prepare buffer for detection (convert to float)
        prepareDetectionImage(s.get(), method);
        // detect
        float corr = 0;
        const double detectTime = executionTime([&]() { corr = detectEmbeddedBuffer(s.get(), method); }, loops);
        cout << std::format("Calculation of {} correlation:\n{}\n\n", name, formatExecutionTime(showFps, detectTime / loops));
        // optionally save to disk
        if (saveToDisk) {
            cout << "Writing to disk... ";
            saveImage(s.get(), imageFile, method);
            cout << success("Successfully saved to disk\n\n");
        }
        return corr;
    };

    // run benchmarks for ME and NVF
    const float corrNvf = runWatermarkingProcess(MaskMethod::NVF, "NVF");
    const float corrMe = runWatermarkingProcess(MaskMethod::ME, "ME");
    cout << std::format("Correlation [NVF]: {:.16f}\n", corrNvf);
    cout << std::format("Correlation [ME]:  {:.16f}\n", corrMe);
    return EXIT_SUCCESS;
}

// video processing, it opens the video, embeds or detects the watermark based on the mode specified in settings.ini,
// and optionally encodes the output video with the embedded watermark (using hardware if specified and available)
static int testForVideo(const INIReader& inir, const string& videoFile, const int p, const float psnr) {
    const bool showFps = inir.GetBoolean("global", "display_fps", true);
    const bool isEmbed = inir.Get("video", "mode", "embed") == "embed";

    // supply the relevant settings to the video session input struct
    VideoSettings settings;
    settings.videoFile = videoFile;
    settings.watermarkSeed = inir.GetInteger("global", "watermark_seed", 0);
    settings.p = p;
    settings.psnr = psnr;
    settings.watermarkInterval = std::max(1, static_cast<int>(inir.GetInteger("video", "watermark_interval", 1)));
    settings.hwDecoder = inir.Get("compute", "cuda_hw_decoder", "");
    settings.useHwEncoder = inir.GetBoolean("compute", "cuda_hw_encoder", false);
    settings.encodeOptions = settings.useHwEncoder ? inir.Get("video", "hw_encode_options", "-c:v hevc_nvenc -preset p6 -tune hq -cq 26 -b:v 0")
                                                   : inir.Get("video", "cpu_encode_options", "-c:v libx265 -preset fast -crf 23");
    settings.encodeOutputPath = inir.Get("video", "encode_output_path", "");

    // open video session
    VideoHandle session = initVideo(settings);

    int framesProcessed = 0;
    double totalTime = 0;
    // embed and encode the video, or just detect the watermark from the input video
    if (isEmbed) {
        totalTime = executionTime([&]() { framesProcessed = embedVideo(session.get()); }, 1, false);
        cout << info("\nWatermark embedding total time: " + formatExecutionTime(false, totalTime) + "\n\n");
    } else {
        totalTime = executionTime([&]() { framesProcessed = detectVideo(session.get()); }, 1, false);
        cout << info("\nWatermark detection total time: " + formatExecutionTime(false, totalTime) + "\n");
        cout << info("Average execution time per frame: " + formatExecutionTime(showFps, totalTime / framesProcessed) + "\n");
    }
    return EXIT_SUCCESS;
}

/*!
 *  \brief  This is a project implementation of my Thesis with title:
 *			EFFICIENT IMPLEMENTATION OF WATERMARKING ALGORITHMS AND
 *			WATERMARK DETECTION IN IMAGE AND VIDEO USING GPU.
 *  \author Dimitris Karatzas
 */
int main() {
    int exitCode = EXIT_SUCCESS;
    try {
        // open parameters file
        const INIReader inir("settings.ini");
        if (inir.ParseError() < 0)
            throw std::runtime_error("Could not load settings.ini");
        // initialize backend data (GPU devices, OpenMP threads, etc.)
        initializeEnvironment(inir.GetInteger("compute", "opencl_device_id", 0));
        const int p = inir.GetInteger("global", "p", -1);
        if (p != 3 && p != 5 && p != 7 && p != 9)
            throw std::runtime_error("p must be 3, 5, 7 or 9");
        const float psnr = inir.GetFloat("global", "psnr", -1.0f);
        if (psnr <= 0)
            throw std::runtime_error("PSNR must be a positive number");
        // test algorithms
        const string videoFile = inir.Get("video", "path", "");
        const string imageMode = inir.Get("image", "mode", "");
        if (!videoFile.empty())
            exitCode = testForVideo(inir, videoFile, p, psnr);
        else if (imageMode == "batch_embed")
            exitCode = testForImageBatch(inir, p, psnr, true);
        else if (imageMode == "batch_detect")
            exitCode = testForImageBatch(inir, p, psnr, false);
        else if (imageMode == "single")
            exitCode = testForImageSingle(inir, p, psnr);
        else
            throw std::runtime_error("Invalid mode specified in settings.ini. Must be 'single', 'batch' for images, or specify a video path.");
    } catch (const std::exception& ex) {
        cout << err(string("Fatal error: ") + ex.what() + "\n");
        exitCode = EXIT_FAILURE;
    }
    system("pause");
    return exitCode;
}