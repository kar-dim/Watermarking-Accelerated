#include "common_utils.hpp"
#include "libs/inih/INIReader.h"
#include "WatermarkCore.hpp"
#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <format>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <map>
#include <omp.h>
#include <queue>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>
#include <WatermarkTypes.hpp>

using namespace WatermarkCore;
using namespace CommonUtils;
namespace fs = std::filesystem;

using std::cout;
using std::string;

namespace {
// Command-line option definition mapping an INI section to its key name
struct OptionDefinition {
    std::string_view section;
    std::string_view name;
};

// List of supported command-line settings matching INI configuration keys
constexpr std::array cliSettings = {
    OptionDefinition{"global",  "watermark_password"  },
    OptionDefinition{"global",  "p"                   },
    OptionDefinition{"global",  "psnr"                },
    OptionDefinition{"global",  "display_fps"         },
    OptionDefinition{"compute", "gpu_device_id"       },
    OptionDefinition{"compute", "opencl_device_id"    },
    OptionDefinition{"compute", "cuda_hw_decoder"     },
    OptionDefinition{"compute", "cuda_hw_encoder"     },
    OptionDefinition{"image",   "mode"                },
    OptionDefinition{"image",   "path"                },
    OptionDefinition{"image",   "output_path"         },
    OptionDefinition{"image",   "benchmark_loops"     },
    OptionDefinition{"video",   "mode"                },
    OptionDefinition{"video",   "path"                },
    OptionDefinition{"video",   "encode_output_path"  },
    OptionDefinition{"video",   "encode_codec_options"},
    OptionDefinition{"video",   "hw_encode_options"   },
    OptionDefinition{"video",   "watermark_interval"  }
};

// Builds a lookup key string from section and setting name
string settingKey(const std::string_view section, const std::string_view name) { return string(section) + "=" + string(name); }

// Holds parsed command-line flags and setting overrides
struct CommandLineOptions {
    std::map<string, string> settings;
    string settingsFile = "settings.ini";
    bool benchmark = false;
    bool benchmarkSave = false;
    bool help = false;
    bool noPause = false;
};

// Resolves a command-line option name to an OptionDefinition
OptionDefinition resolveOption(const std::string_view option) {
    const size_t separator = option.find('.');
    if (separator != std::string_view::npos) {
        // Find exact match when section prefix is explicitly specified
        const std::string_view section = option.substr(0, separator);
        const std::string_view name = option.substr(separator + 1);
        for (const auto& definition : cliSettings)
            if (definition.section == section && definition.name == name)
                return definition;
        throw std::runtime_error("Unknown command-line setting '--" + string(option) + "'");
    }

    // Find unique match when no section prefix is provided
    const OptionDefinition* match = nullptr;
    for (const auto& definition : cliSettings) {
        if (definition.name != option)
            continue;
        if (match)
            throw std::runtime_error("Ambiguous setting '--" + string(option) + "'; use '--image." + string(option) + "' or '--video." + string(option) + "'");
        match = &definition;
    }
    if (!match)
        throw std::runtime_error("Unknown command-line option '--" + string(option) + "'");
    return *match;
}

// Parses command-line arguments into flags and setting overrides
CommandLineOptions parseCommandLine(const int argc, char* argv[]) {
    CommandLineOptions result;
    for (int i = 1; i < argc; ++i) {
        const string argument = argv[i];
        if (argument == "--help" || argument == "-h") {
            result.help = true;
            continue;
        }
        if (argument == "--bench") {
            result.benchmark = true;
            result.noPause = true;
            continue;
        }
        if (argument == "--bench-save") {
            result.benchmark = true;
            result.benchmarkSave = true;
            result.noPause = true;
            continue;
        }
        if (argument == "--no-pause") {
            result.noPause = true;
            continue;
        }
        if (!argument.starts_with("--"))
            throw std::runtime_error("Unexpected positional argument '" + argument + "'");

        // Parse key and value from arguments using equals sign or space separator
        const size_t equals = argument.find('=');
        const string option = argument.substr(2, equals == string::npos ? string::npos : equals - 2);
        string value;
        if (equals != string::npos)
            value = argument.substr(equals + 1);
        else {
            if (i + 1 >= argc)
                throw std::runtime_error("Missing value for '--" + option + "'");
            value = argv[++i];
        }

        // Custom INI settings file override
        if (option == "settings") {
            result.settingsFile = std::move(value);
            continue;
        }
        // Match option to setting definition and store override
        const auto definition = resolveOption(option);
        const auto name = definition.name == "opencl_device_id" ? "gpu_device_id" : definition.name;
        result.settings[settingKey(definition.section, name)] = std::move(value);
    }
    return result;
}

// Displays command-line usage and available options
void printHelp() {
    cout << R"(
Usage: Watermarking-CLI [options]

The application reads settings.ini, then applies command-line overrides.

Control options:
  --bench                    Benchmark ME embed/detect for p=3,5,7,9 and 480p..4K.
                             Writes readme_pictures/{cuda,opencl,eigen}.csv.
  --bench-save               Also save one embedded image per benchmark case.
  --settings FILE            Read a different INI file (default: settings.ini).
  --no-pause                 Do not wait for a key before exiting.
  -h, --help                 Show this help.

Global settings:
  --watermark_password TEXT  Password used to generate the watermark.
  --p N                      Prediction window size: 3, 5, 7, or 9.
  --psnr DB                  Embed strength as a positive PSNR value in dB.
  --display_fps BOOL         Show FPS for video operations (true/false).

Compute settings:
  --gpu_device_id N          GPU device index (CUDA and OpenCL builds, defaults to 0).
  --opencl_device_id N       Legacy alias for gpu_device_id.
  --cuda_hw_decoder BOOL     Use NVDEC for video, with CPU fallback (CUDA only).
  --cuda_hw_encoder BOOL     Use NVENC for video (true/false; all builds).

Image settings:
  --image.mode MODE          single, batch_embed, or batch_detect.
  --image.path PATH          Input image, or input directory for batch mode.
  --output_path FILE         Required destination file for single mode.
  --benchmark_loops N        Positive measured loop count for --bench only.

Video settings:
  --video.mode MODE          embed or detect.
  --video.path FILE          Input video; selects video mode when provided.
  --encode_output_path FILE  Destination video file for embed mode.
  --encode_codec_options STR Software encoder and options, e.g. -c:v libx265.
  --hw_encode_options STR    NVENC encoder and options, e.g. -c:v hevc_nvenc.
  --watermark_interval N     Embed or detect every Nth video frame (N >= 1).

Every setting also accepts its section-qualified spelling, for example
--global.p=5 or --compute.gpu_device_id=1. Values may use '--key value' or
'--key=value'. The duplicated mode/path names must be section-qualified.
)";
}

// Configuration accessor that applies command-line overrides on top of INI values
class Settings {
  public:
    Settings(const INIReader& ini, const std::map<string, string>& overrides) : ini_(ini), overrides_(overrides) {}

    // Retrieves string value with command-line override precedence
    string Get(const string& section, const string& name, const string& defaultValue) const {
        const auto overrideValue = overrides_.find(settingKey(section, name));
        return overrideValue == overrides_.end() ? ini_.Get(section, name, defaultValue) : overrideValue->second;
    }

    // Retrieves integer value with command-line override precedence
    long GetInteger(const string& section, const string& name, const long defaultValue) const {
        const string* value = findOverride(section, name);
        if (!value)
            return ini_.GetInteger(section, name, defaultValue);
        errno = 0;
        char* end = nullptr;
        const long parsed = std::strtol(value->c_str(), &end, 0);
        if (errno != 0 || end == value->c_str() || *end != '\0')
            throw std::runtime_error("Invalid integer for '--" + name + "': " + *value);
        return parsed;
    }

    // Retrieves float value with command-line override precedence
    float GetFloat(const string& section, const string& name, const float defaultValue) const {
        const string* value = findOverride(section, name);
        if (!value)
            return ini_.GetFloat(section, name, defaultValue);
        errno = 0;
        char* end = nullptr;
        const float parsed = std::strtof(value->c_str(), &end);
        if (errno != 0 || end == value->c_str() || *end != '\0')
            throw std::runtime_error("Invalid number for '--" + name + "': " + *value);
        return parsed;
    }

    // Retrieves boolean value with command-line override precedence
    bool GetBoolean(const string& section, const string& name, const bool defaultValue) const {
        const string* value = findOverride(section, name);
        if (!value)
            return ini_.GetBoolean(section, name, defaultValue);
        string normalized = *value;
        std::transform(normalized.begin(), normalized.end(), normalized.begin(), [](const unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (normalized == "true" || normalized == "yes" || normalized == "on" || normalized == "1")
            return true;
        if (normalized == "false" || normalized == "no" || normalized == "off" || normalized == "0")
            return false;
        throw std::runtime_error("Invalid boolean for '--" + name + "': " + *value);
    }

  private:
    // Looks up a setting override by section and key name
    const string* findOverride(const string& section, const string& name) const {
        const auto value = overrides_.find(settingKey(section, name));
        return value == overrides_.end() ? nullptr : &value->second;
    }

    const INIReader& ini_;
    const std::map<string, string>& overrides_;
};

// Escapes and quotes a string for CSV formatting
string csvString(const string& value) {
    string escaped = "\"";
    for (const char c : value)
        escaped += c == '"' ? "\"\"" : string(1, c);
    return escaped + "\"";
}
} // namespace

/*!
 *  \brief  Helper functions for testing the watermark algorithms
 *  \author Dimitris Karatzas
 */
// batch processing of images in a directory (for both embed and detect)
static int testForImageBatch(const Settings& inir, const int p, const float psnr, const bool isEmbed) {
    const string watermarkPassword = inir.Get("global", "watermark_password", "");
    checkError(watermarkPassword.empty(), "No valid watermark password specified!");

    const fs::path inputDir(inir.Get("image", "path", ""));
    if (!fs::exists(inputDir) || !fs::is_directory(inputDir))
        throw std::runtime_error("Error: Batch path is not a valid directory!");

    // only create the below if we are embedding!
    fs::path outputDir;
    struct PendingSave {
        fs::path inputFile;
        std::future<void> future;
    };
    std::queue<PendingSave> saveTasks;
    std::vector<ExportHandle> exportPool;
    size_t bufferIndex = 0;
    size_t maxParallelSaves = 0;
    if (isEmbed) {
        outputDir = inputDir / "watermark_output";
        fs::create_directories(outputDir);
        maxParallelSaves = omp_get_max_threads();
        // preallocate exactly enough buffers for the max concurrent threads
        for (size_t i = 0; i < maxParallelSaves; i++)
            exportPool.push_back(createReusableExportBuffer());
    }

    // get the valid images of the directory, if no valid image files are found, throw an error
    const std::vector<fs::path> validFiles = getValidImageFiles(inputDir);
    checkError(validFiles.empty(), "No valid image files found in directory!");
    cout << info(std::format("Found {} images. Starting batch {}...\n\n", validFiles.size(), isEmbed ? "embedding" : "detection"));

    // initialize the watermarking session once and reuse for all images in the batch
    auto session = createImageSession(watermarkPassword, p, psnr);
    int successCount = 0;
    float corr = 0.0f;

    // Pop before calling get(): failed future must NOT remain at the front
    // else it will "poison" every later attempt to drain the queue
    const auto completeOldestSave = [&]() {
        PendingSave pending = std::move(saveTasks.front());
        saveTasks.pop();
        try {
            pending.future.get();
            cout << success(std::format(" [OK] {}\n", pending.inputFile.filename().string()));
            ++successCount;
        } catch (const std::exception& e) { cout << err(std::format(" [FAILED] {} - Save error: {}\n", pending.inputFile.filename().string(), cleanError(e.what()))); }
    };

    // start the batch process (begin timer)
    const auto batchStart = std::chrono::high_resolution_clock::now();
    // preload the first image
    const int selectedDevice = getCurrentDeviceIndex();
    std::future<PreloadedHandle> prefetchTask = std::async(std::launch::async, preloadImageFromDisk, validFiles[0].string(), selectedDevice, false);
    for (size_t i = 0; i < validFiles.size(); i++) {
        bool nextPrefetchStarted = false;
        try {
            auto currentImage = prefetchTask.get();
            // spawn background thread to read the next image
            if (i + 1 < validFiles.size()) {
                prefetchTask = std::async(std::launch::async, preloadImageFromDisk, validFiles[i + 1].string(), selectedDevice, false);
                nextPrefetchStarted = true;
            }
            // give the buffer to the watermark engine (may trigger lazy init)
            bindPreloadedImage(session.get(), std::move(currentImage));
            // embed
            if (isEmbed) {
                embedImage(session.get(), MaskMethod::ME);

                // if we have too many active saves, wait for the oldest one to finish
                const fs::path outFile = outputDir / validFiles[i].filename();
                if (saveTasks.size() >= maxParallelSaves)
                    completeOldestSave();
                // get the next available buffer from the pool and do a zero copy allocation into it
                auto* currentBuffer = exportPool[bufferIndex].get();
                exportForSave(session.get(), currentBuffer, MaskMethod::ME);
                // launch the heavy save to disk task in the background and cycle to the next buffer
                saveTasks.push(PendingSave{validFiles[i], std::async(std::launch::async, flushToDiskAsync, currentBuffer, outFile.string(), MaskMethod::ME)});
                bufferIndex = (bufferIndex + 1) % maxParallelSaves;
            } else { // detect
                corr = detectLoadedImage(session.get(), MaskMethod::ME);
                cout << success(std::format(" [OK] Correlation: {:.2f}, {}\n", corr, validFiles[i].filename().string()));
                ++successCount;
            }

        } catch (const std::exception& e) {
            cout << err(std::format(" [FAILED] {} - Error: {}\n", validFiles[i].filename().string(), cleanError(e.what())));
            // If the current prefetch failed before the next one was launched,
            // keep the pipeline moving. Do not launch the same prefetch twice.
            if (!nextPrefetchStarted && i + 1 < validFiles.size())
                prefetchTask = std::async(std::launch::async, preloadImageFromDisk, validFiles[i + 1].string(), selectedDevice, false);
        }
    }
    // finish any pending saves before exiting
    if (isEmbed) {
        while (!saveTasks.empty())
            completeOldestSave();
    }
    // stop batch process (and timer)
    const auto batchEnd = std::chrono::high_resolution_clock::now();
    const double totalBatchTime = std::chrono::duration<double>(batchEnd - batchStart).count();

    // print results summary
    checkError(successCount == 0, "No images were successfully processed. Please check the error messages above.");
    cout << info(std::format("\nBatch complete! Successfully processed {}/{} images.\n", successCount, validFiles.size()));
    cout << info("Total batch time: " + formatExecutionTime(false, totalBatchTime) + "\n");

    return EXIT_SUCCESS;
}

// embed one ME watermark and write the requested output image
static int testForImageSingle(const Settings& inir, const int p, const float psnr) {
    const string imageFile = inir.Get("image", "path", "NO_IMAGE");
    checkError(imageFile == "NO_IMAGE", "No valid image file specified!");
    const string outputPath = inir.Get("image", "output_path", "");
    checkError(outputPath.empty(), "Single image mode requires --output_path (or [image]/output_path).");
    std::error_code pathError;
    checkError(fs::equivalent(imageFile, outputPath, pathError), "Input and output image must be different files.");
    const string watermarkPassword = inir.Get("global", "watermark_password", "");
    checkError(watermarkPassword.empty(), "No valid watermark seed specified!");
    const auto started = std::chrono::steady_clock::now();
    auto s = createImageSession(watermarkPassword, p, psnr);
    loadImage(s.get(), imageFile);
    const auto dims = getImageDims(s.get());
    cout << info("Image size is: " + std::to_string(dims.first) + "x" + std::to_string(dims.second) + " (HxW)\n\n");
    embedImage(s.get(), MaskMethod::ME);
    finish();
    saveImageExact(s.get(), outputPath);
    const double totalSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    cout << success(std::format("Embedded ME watermark in '{}' (p = {}, PSNR = {} dB).\n", outputPath, p, psnr));
    cout << std::format("Total image time (load, watermark setup, embed, save): {:.3f} seconds\n", totalSeconds);
    return EXIT_SUCCESS;
}

// video processing, it opens the video, embeds or detects the watermark based on the mode specified in settings.ini,
// and optionally encodes the output video with the embedded watermark (using hardware acceleration if specified and available)
static int testForVideo(const Settings& inir, const string& videoFile, const int p, const float psnr) {
    const bool showFps = inir.GetBoolean("global", "display_fps", true);
    const string videoMode = inir.Get("video", "mode", "embed");
    checkError(videoMode != "embed" && videoMode != "detect", "Invalid video mode. Must be 'embed' or 'detect'.");
    const bool isEmbed = videoMode == "embed";

    // supply the relevant settings to the video session input struct
    VideoSettings settings;
    settings.videoFile = videoFile;
    settings.watermarkPassword = inir.Get("global", "watermark_password", "");
    checkError(settings.watermarkPassword.empty(), "No valid watermark password specified!");
    settings.p = p;
    settings.psnr = psnr;
    settings.watermarkInterval = std::max(1, static_cast<int>(inir.GetInteger("video", "watermark_interval", 1)));
    settings.useHwDecoder = inir.GetBoolean("compute", "cuda_hw_decoder", true);
    settings.useHwEncoder = inir.GetBoolean("compute", "cuda_hw_encoder", false);
    settings.encodeOptions = settings.useHwEncoder ? inir.Get("video", "hw_encode_options", "-c:v hevc_nvenc -preset p6 -tune hq -cq 26 -b:v 0")
                                                   : inir.Get("video", "encode_codec_options", "-c:v libx265 -preset fast -crf 23");
    settings.encodeOutputPath = inir.Get("video", "encode_output_path", "");

    // Video embedding also runs FFmpeg encoder threads. Limit Eigen/OpenMP to
    // physical cores so the two thread pools do not oversubscribe the CPU
    if (isEmbed)
        optimizeThreadsForVideoEmbedding();

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

// Runs ME mask benchmark across standard resolutions and prediction orders, writing CSV output
static int runCliBenchmark(const Settings& settings, const float psnr, const bool saveImages) {
    // Benchmark test image definition
    struct BenchmarkImage {
        const char* resolution;
        const char* path;
    };
    constexpr std::array images = {
        BenchmarkImage{"480p",  "samples/images/480p.png" },
        BenchmarkImage{"720p",  "samples/images/720p.png" },
        BenchmarkImage{"1080p", "samples/images/1080p.png"},
        BenchmarkImage{"4K",    "samples/images/4k.png"   }
    };
    constexpr std::array predictionOrders = {3, 5, 7, 9};

    const string backend = getBackendName();
    const string device = getDeviceName();
    // Use fewer iterations for CPU backend to keep benchmark duration reasonable
    const int defaultLoops = backend == "eigen" ? 100 : 1000;
    const int loops = settings.GetInteger("image", "benchmark_loops", defaultLoops);
    checkError(loops <= 0, "benchmark_loops must be positive.");
    const string watermarkPassword = settings.Get("global", "watermark_password", "");
    checkError(watermarkPassword.empty(), "No valid watermark password specified!");

    // Create output folder and initialize benchmark CSV file
    const fs::path outputDir = "readme_pictures";
    fs::create_directories(outputDir);
    const fs::path outputPath = outputDir / (backend + ".csv");
    std::ofstream output(outputPath, std::ios::trunc);
    if (!output)
        throw std::runtime_error("Could not write benchmark CSV: " + outputPath.string());
    output << "backend,device,p,resolution,operation,fps,seconds,loops,image\n" << std::setprecision(17);

    cout << info(std::format("CLI benchmark: {} on {} ({} loops per measurement)\n", backend, device, loops));
    // Test each prediction order
    for (const int p : predictionOrders) {
        auto session = createImageSession(watermarkPassword, p, psnr);
        // Test each resolution image
        for (const auto& image : images) {
            if (!fs::is_regular_file(image.path))
                throw std::runtime_error("Benchmark image not found: " + string(image.path));
            loadImage(session.get(), image.path);

            // Measure embedding execution time
            const double embedSeconds = executionTime(
                                            [&]() {
                                                embedImage(session.get(), MaskMethod::ME);
                                                finish();
                                            },
                                            loops) /
                                        loops;
            if (saveImages) {
                const fs::path imagesDir = outputDir / (backend + "_images");
                fs::create_directories(imagesDir);
                saveImageExact(session.get(), (imagesDir / std::format("p{}_{}.png", p, image.resolution)).string());
            }
            // Measure detection execution time and correlation
            prepareDetectionImage(session.get(), MaskMethod::ME);
            float correlation = 0.0f;
            const double detectSeconds = executionTime([&]() { correlation = detectEmbeddedBuffer(session.get(), MaskMethod::ME); }, loops) / loops;

            // Writes a single measurement record to CSV
            const auto writeResult = [&](const char* operation, const double seconds) {
                output << csvString(backend) << ',' << csvString(device) << ',' << p << ',' << csvString(image.resolution) << ',' << csvString(operation) << ',' << 1.0 / seconds << ',' << seconds
                       << ',' << loops << ',' << csvString(image.path) << '\n';
            };
            writeResult("embed", embedSeconds);
            writeResult("detect", detectSeconds);
            output.flush();
            cout << std::format("p={}, {:>5}: embed {:>10.2f} FPS, detect {:>10.2f} FPS, correlation {:.8f}\n", p, image.resolution, 1.0 / embedSeconds, 1.0 / detectSeconds, correlation);
        }
    }
    cout << success("Benchmark CSV written to " + outputPath.string() + "\n");
    return EXIT_SUCCESS;
}

/*!
 *  \brief  This is a project implementation of my Thesis with title:
 *			EFFICIENT IMPLEMENTATION OF WATERMARKING ALGORITHMS AND
 *			WATERMARK DETECTION IN IMAGE AND VIDEO USING GPU.
 *  \author Dimitris Karatzas
 */
int main(const int argc, char* argv[]) {
    int exitCode = EXIT_SUCCESS;
    // Supplying any argument implies non-interactive use
    bool pauseBeforeExit = argc == 1;
    try {
        // Parse command-line options and overrides
        const CommandLineOptions commandLine = parseCommandLine(argc, argv);
        pauseBeforeExit = !commandLine.noPause;
        if (commandLine.help) {
            printHelp();
            return EXIT_SUCCESS;
        }
        // open parameters file
        const INIReader ini(commandLine.settingsFile);
        if (ini.ParseError() < 0)
            throw std::runtime_error("Could not load " + commandLine.settingsFile);
        // Combine INI settings with command-line overrides
        const Settings inir(ini, commandLine.settings);
        // initialize backend data (GPU devices, OpenMP threads, etc.)
        // the generic key takes precedence while old INI files remain valid
        initializeEnvironment(inir.GetInteger("compute", "gpu_device_id", inir.GetInteger("compute", "opencl_device_id", 0)));
        const float psnr = inir.GetFloat("global", "psnr", -1.0f);
        if (psnr <= 0)
            throw std::runtime_error("PSNR must be a positive number");
        // Run standalone benchmark if requested
        if (commandLine.benchmark)
            return runCliBenchmark(inir, psnr, commandLine.benchmarkSave);

        const int p = inir.GetInteger("global", "p", -1);
        if (p != 3 && p != 5 && p != 7 && p != 9)
            throw std::runtime_error("p must be 3, 5, 7 or 9");
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
    // flush first
    cout.flush();
    // Pause terminal only if run interactively
    if (pauseBeforeExit)
        system("pause");
    return exitCode;
}
