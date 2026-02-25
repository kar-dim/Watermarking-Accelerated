#include "common_utils.hpp"
#include "libs/inih/INIReader.h"
#include "WatermarkEngine.hpp"
#include <algorithm>
#include <cstdlib>
#include <exception>
#include <format>
#include <iostream>
#include <stdexcept>
#include <string>
#include <WatermarkTypes.hpp>

using namespace WatermarkEngine;
using namespace CommonUtils;

/*!
 *  \brief  Helper functions for testing the watermark algorithms
 *  \author Dimitris Karatzas
 */
static inline std::string info(const std::string& str) { return "\033[38;5;208m" + str + "\033[0m"; }
static inline std::string err(const std::string& str) { return "\033[91m" + str + "\033[0m"; }
static inline std::string success(const std::string& str) { return "\033[92m" + str + "\033[0m"; }

int testForImage(const INIReader& inir, int p, float psnr) {
    const std::string imageFile = inir.Get("image", "path", "NO_IMAGE");
    const std::string watermarkFile = inir.Get("global", "watermark_data_file", "");
    const bool showFps = inir.GetBoolean("global", "display_fps", true);
    const bool saveToDisk = inir.GetBoolean("image", "save_to_disk", false);
    int loops = inir.GetInteger("image", "benchmark_loops", 5);
    loops = loops <= 0 ? 5 : loops;

    std::cout << "Each test will be executed " << loops << " times.\n";

    // load watermarking session
    ImageHandle session{nullptr};
    const double loadTime = executionTime([&]() { session = initImage(imageFile, watermarkFile, p, psnr); });
    std::cout << "Time to load image data from disk: " << loadTime << " seconds\n\n";

    // helper lambda for embedding and detection benchmarks
    auto runWatermarkingProcess = [&](MaskMethod method, const std::string& name) {
        // embed
        const double embedTime = executionTime([&]() { embedImage(session.get(), method); }, loops);
        std::cout << std::format("Calculation of {} mask (p = {}, PSNR = {}dB)\n{}\n\n", name, p, psnr, formatExecutionTime(showFps, embedTime / loops));
        // prepare buffer for detection (convert to float)
        prepareDetectionImage(session.get(), method);
        // detect
        float corr = 0;
        const double detectTime = executionTime([&]() { corr = detectEmbeddedBuffer(session.get(), method); }, loops);
        std::cout << std::format("Calculation of {} correlation:\n{}\n\n", name, formatExecutionTime(showFps, detectTime / loops));
        // optionally save to disk
        if (saveToDisk) {
            std::cout << "Writing to disk... ";
            saveImage(session.get(), imageFile, method);
            std::cout << success("Successfully saved to disk\n\n");
        }
        return corr;
    };

    // run benchmarks for ME and NVF
    const float corrNvf = runWatermarkingProcess(MaskMethod::NVF, "NVF");
    const float corrMe = runWatermarkingProcess(MaskMethod::ME, "ME");
    std::cout << std::format("Correlation [NVF]: {:.16f}\n", corrNvf);
    std::cout << std::format("Correlation [ME]:  {:.16f}\n", corrMe);
    return EXIT_SUCCESS;
}

int testForVideo(const INIReader& inir, const std::string& videoFile, int p, float psnr) {
    const bool showFps = inir.GetBoolean("global", "display_fps", true);
    const bool isEmbed = inir.Get("video", "mode", "embed") == "embed";

    VideoSettings settings;
    settings.videoFile = videoFile;
    settings.watermarkDataPath = inir.Get("global", "watermark_data_file", "");
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
    if (isEmbed) {
        totalTime = executionTime([&]() { framesProcessed = embedVideo(session.get()); }, 1, false);
        std::cout << info("\nWatermark embedding total time: " + formatExecutionTime(false, totalTime) + "\n\n");
    } else {
        totalTime = executionTime([&]() { framesProcessed = detectVideo(session.get()); }, 1, false);
        std::cout << info("\nWatermark detection total time: " + formatExecutionTime(false, totalTime) + "\n");
        std::cout << info("Average execution time per frame: " + formatExecutionTime(showFps, totalTime / framesProcessed) + "\n");
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
        std::string videoFile = inir.Get("video", "path", "");
        exitCode = !videoFile.empty() ? testForVideo(inir, videoFile, p, psnr) : testForImage(inir, p, psnr);
    } catch (const std::exception& ex) {
        std::cout << err(std::string("Fatal error: ") + ex.what() + "\n");
        exitCode = EXIT_FAILURE;
    }
    system("pause");
    return exitCode;
}