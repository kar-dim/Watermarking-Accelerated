#include "common_utils.hpp"
#include "WatermarkCore.hpp"
#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <format>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <omp.h>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>
#include <windows.h>

/*!
 *  \brief  Command-line interface entry point for image/video watermarking and benchmarking
 *  \author Dimitris Karatzas
 */

using namespace CommonUtils;
using namespace WatermarkCore;
namespace fs = std::filesystem;

using std::cout;
using std::string;

namespace {

inline std::string formatExecutionTime(const bool showFps, const double seconds) { return showFps ? std::format("FPS: {:.2f} FPS", 1.0 / seconds) : std::format("{:.6f} seconds", seconds); }

// measures execution time for a passed function
template <typename F>
double executionTime(F&& func, int loops = 1, const bool warmup = true) {
    if (warmup)
        func(); // warmup one time
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < loops; i++)
        func();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end - start).count();
}
// Grouped option names keep image/video mode and path unambiguous
struct OptionDefinition {
    std::string_view section;
    std::string_view name;
};

// Supported command-line options and their groups
constexpr std::array cliSettings = {
    OptionDefinition{"global",  "watermark_password"  },
    OptionDefinition{"global",  "p"                   },
    OptionDefinition{"global",  "psnr"                },
    OptionDefinition{"global",  "display_fps"         },
    OptionDefinition{"compute", "gpu_device_id"       },
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
string settingKey(const std::string_view section, const std::string_view name) { return string(section) + "." + string(name); }

// Holds parsed command-line flags and option values
struct CommandLineOptions {
    std::map<string, string> settings;
    bool benchmark = false;
    bool benchmarkSave = false;
    bool help = false;
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

// Parses command-line arguments into flags and option values
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
            continue;
        }
        if (argument == "--bench-save") {
            result.benchmark = true;
            result.benchmarkSave = true;
            continue;
        }
        if (argument == "--no-pause") {
            // Keep accepting the legacy flag; CLI commands never pause
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

        // Match the option to its definition and store its value
        const auto definition = resolveOption(option);
        result.settings[settingKey(definition.section, definition.name)] = std::move(value);
    }
    return result;
}

// Displays command-line usage and available options
void printHelp() {
    cout << R"(
Usage: Watermarking-CLI [options]

Configure the application with command-line options. With no options, show help.

Control options:
  --bench                    Benchmark ME embed/detect for p=3,5,7,9 and 480p..4K.
                             Writes readme_pictures/{cuda,opencl,eigen}.csv.
                             Uses a fixed password unless one is supplied.
  --bench-save               Also save one embedded image per benchmark case.
  --no-pause                 Legacy flag; CLI commands never pause.
  -h, --help                 Show this help.

Global settings:
  --watermark_password TEXT  Required password for image/video operations.
  --p N                      Prediction window size: 3, 5, 7, or 9 (default: 3).
  --psnr DB                  Positive embed PSNR in dB (default: 40).
  --display_fps BOOL         Show FPS for video operations (default: true).

Compute settings:
  --gpu_device_id N          GPU device index (CUDA and OpenCL builds, defaults to 0).
  --cuda_hw_decoder BOOL     Use NVDEC with CPU fallback (CUDA only; default: true).
  --cuda_hw_encoder BOOL     Use NVENC for video (all builds; default: false).

Image settings:
  --image.mode MODE          single, batch_embed, or batch_detect (default: single).
  --image.path PATH          Input image, or input directory for batch mode.
  --output_path FILE         Required destination file for single mode.
  --benchmark_loops N        Positive measured loop count for --bench only.

Video settings:
  --video.mode MODE          embed or detect (default: embed).
  --video.path FILE          Input video; selects video mode when provided.
  --encode_output_path FILE  Destination video file for embed mode.
  --encode_codec_options STR Software encoder and options, e.g. -c:v libx265.
  --hw_encode_options STR    NVENC encoder and options, e.g. -c:v hevc_nvenc.
  --watermark_interval N     Embed or detect every Nth frame (N >= 1; default: 1).

Every option also accepts its group-qualified spelling, for example
--global.p=5 or --compute.gpu_device_id=1. Values may use '--key value' or
'--key=value'. The duplicated mode/path names must be group-qualified.

Example: Watermarking-CLI --image.path input.png --output_path output.png
         --watermark_password "your-password"
)";
}

// Typed access to command-line values with defaults for omitted options
class Settings {
  public:
    explicit Settings(const std::map<string, string>& values) : values_(values) {}

    // Use the caller's default when the option was not supplied
    string Get(const string& section, const string& name, const string& defaultValue) const {
        const auto value = values_.find(settingKey(section, name));
        return value == values_.end() ? defaultValue : value->second;
    }

    // Reject overflow and trailing text in integer options
    long GetInteger(const string& section, const string& name, const long defaultValue) const {
        const string value = Get(section, name, std::to_string(defaultValue));
        errno = 0;
        char* end = nullptr;
        const long parsed = std::strtol(value.c_str(), &end, 0);
        if (errno != 0 || end == value.c_str() || *end != '\0')
            throw std::runtime_error("Invalid integer for '" + section + "." + name + "': " + value);
        return parsed;
    }

    // Reject non-finite, out-of-range and partially parsed numbers
    float GetFloat(const string& section, const string& name, const float defaultValue) const {
        const string value = Get(section, name, std::to_string(defaultValue));
        errno = 0;
        char* end = nullptr;
        const float parsed = std::strtof(value.c_str(), &end);
        if (errno != 0 || end == value.c_str() || *end != '\0' || !std::isfinite(parsed))
            throw std::runtime_error("Invalid number for '" + section + "." + name + "': " + value);
        return parsed;
    }

    // Accept common boolean spellings without case sensitivity
    bool GetBoolean(const string& section, const string& name, const bool defaultValue) const {
        const string value = Get(section, name, defaultValue ? "true" : "false");
        string normalized = value;
        std::transform(normalized.begin(), normalized.end(), normalized.begin(), [](const unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (normalized == "true" || normalized == "yes" || normalized == "on" || normalized == "1")
            return true;
        if (normalized == "false" || normalized == "no" || normalized == "off" || normalized == "0")
            return false;
        throw std::runtime_error("Invalid boolean for '" + section + "." + name + "': " + value);
    }

  private:
    const std::map<string, string>& values_;
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
static int testForImageBatch(const Settings& options, const int p, const float psnr, const bool isEmbed) {
    const string watermarkPassword = options.Get("global", "watermark_password", "");
    checkError(watermarkPassword.empty(), "No valid watermark password specified!");

    const fs::path inputDir(options.Get("image", "path", ""));
    if (!fs::exists(inputDir) || !fs::is_directory(inputDir))
        throw std::runtime_error("Error: Batch path is not a valid directory!");

    // only create the below if we are embedding!
    fs::path outputDir;
    // background saves, one per thread (it waits for its saves before the session is released)
    std::optional<ImageSaver> saver;
    if (isEmbed) {
        outputDir = inputDir / "watermark_output";
        fs::create_directories(outputDir);
        saver.emplace(static_cast<size_t>(omp_get_max_threads()));
    }

    // get the valid images of the directory, if no valid image files are found, throw an error
    const std::vector<fs::path> validFiles = getValidImageFiles(inputDir);
    checkError(validFiles.empty(), "No valid image files found in directory!");
    cout << info(std::format("Found {} images. Starting batch {}...\n\n", validFiles.size(), isEmbed ? "embedding" : "detection"));

    // initialize the watermarking session once and reuse for all images in the batch
    auto session = createImageSession(watermarkPassword, p, psnr);
    int successCount = 0;
    float corr = 0.0f;

    // reports the finished saves (in the order they finished)
    const auto reportSaves = [&](const std::vector<SaveResult>& results) {
        for (const auto& [index, error] : results) {
            const string fileName = validFiles[index].filename().string();
            if (!error.empty()) {
                cout << err(std::format(" [FAILED] {} - Save error: {}\n", fileName, cleanError(error)));
                continue;
            }
            cout << success(std::format(" [OK] {}\n", fileName));
            ++successCount;
        }
    };

    // start the batch process (begin timer)
    const auto batchStart = std::chrono::high_resolution_clock::now();
    // the next images load in the background while the current one is processed
    ImagePrefetcher images(validFiles, getCurrentDeviceIndex());
    for (size_t i = 0; i < validFiles.size(); i++) {
        try {
            // give the buffer to the watermark engine (may trigger lazy init)
            bindPreloadedImage(session.get(), images.next());
            // embed
            if (isEmbed) {
                embedImage(session.get());
                // waits for a free save slot when all are busy, the export moves (CPU) or copies (GPU) the output into the slot's buffer
                saver->save(session.get(), (outputDir / validFiles[i].filename()).string(), i);
                reportSaves(saver->takeFinished());
            } else { // detect
                corr = detectLoadedImage(session.get());
                cout << success(std::format(" [OK] Correlation: {:.2f}, {}\n", corr, validFiles[i].filename().string()));
                ++successCount;
            }
        } catch (const std::exception& e) { cout << err(std::format(" [FAILED] {} - Error: {}\n", validFiles[i].filename().string(), cleanError(e.what()))); }
    }
    // finish any pending saves before exiting
    if (saver)
        reportSaves(saver->finish());
    // stop batch process (and timer)
    const auto batchEnd = std::chrono::high_resolution_clock::now();
    const double totalBatchTime = std::chrono::duration<double>(batchEnd - batchStart).count();

    // print results summary
    checkError(successCount == 0, "No images were successfully processed. Please check the error messages above.");
    cout << info(std::format("\nBatch complete! Successfully processed {}/{} images.\n", successCount, validFiles.size()));
    cout << info("Total batch time: " + formatExecutionTime(false, totalBatchTime) + "\n");

    return static_cast<size_t>(successCount) == validFiles.size() ? EXIT_SUCCESS : EXIT_FAILURE;
}

// embed one ME watermark and write the requested output image
static int testForImageSingle(const Settings& options, const int p, const float psnr) {
    const string imageFile = options.Get("image", "path", "NO_IMAGE");
    checkError(imageFile == "NO_IMAGE", "No valid image file specified!");
    const string outputPath = options.Get("image", "output_path", "");
    checkError(outputPath.empty(), "Single image mode requires --output_path.");
    std::error_code pathError;
    checkError(fs::equivalent(imageFile, outputPath, pathError), "Input and output image must be different files.");
    const string watermarkPassword = options.Get("global", "watermark_password", "");
    checkError(watermarkPassword.empty(), "No valid watermark password specified! Supply --watermark_password.");
    const auto started = std::chrono::steady_clock::now();
    auto s = createImageSession(watermarkPassword, p, psnr);
    loadImage(s.get(), imageFile);
    const auto dims = getImageDims(s.get());
    cout << info("Image size is: " + std::to_string(dims.first) + "x" + std::to_string(dims.second) + " (HxW)\n\n");
    embedImage(s.get());
    finish();
    saveImageExact(s.get(), outputPath);
    const double totalSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    cout << success(std::format("Embedded ME watermark in '{}' (p = {}, PSNR = {} dB).\n", outputPath, p, psnr));
    cout << std::format("Total image time (load, watermark setup, embed, save): {:.3f} seconds\n", totalSeconds);
    return EXIT_SUCCESS;
}

// Video processing embeds or detects the watermark according to --video.mode,
// and optionally encodes the output video with the embedded watermark (using hardware acceleration if specified and available)
static int testForVideo(const Settings& options, const string& videoFile, const int p, const float psnr) {
    const bool showFps = options.GetBoolean("global", "display_fps", true);
    const string videoMode = options.Get("video", "mode", "embed");
    checkError(videoMode != "embed" && videoMode != "detect", "Invalid video mode. Must be 'embed' or 'detect'.");
    const bool isEmbed = videoMode == "embed";

    // supply the relevant settings to the video session input struct
    VideoSettings settings;
    settings.videoFile = videoFile;
    settings.watermarkPassword = options.Get("global", "watermark_password", "");
    checkError(settings.watermarkPassword.empty(), "No valid watermark password specified!");
    settings.p = p;
    settings.psnr = psnr;
    settings.watermarkInterval = static_cast<int>(options.GetInteger("video", "watermark_interval", 1));
    settings.useHwDecoder = options.GetBoolean("compute", "cuda_hw_decoder", true);
    settings.useHwEncoder = options.GetBoolean("compute", "cuda_hw_encoder", false);
    settings.encodeOptions = settings.useHwEncoder ? options.Get("video", "hw_encode_options", "-c:v hevc_nvenc -preset p6 -tune hq -cq 26 -b:v 0")
                                                   : options.Get("video", "encode_codec_options", "-c:v libx265 -preset fast -crf 23");
    settings.encodeOutputPath = options.Get("video", "encode_output_path", "");

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
    // fixed benchmark password keeps measurements independent of external configuration.
    const string watermarkPassword = settings.Get("global", "watermark_password", "benchmark-watermark-password");
    checkError(watermarkPassword.empty(), "No valid watermark password specified!");

    // Create output folder and initialize benchmark CSV file
    const fs::path outputDir = "readme_pictures";
    fs::create_directories(outputDir);
    const fs::path outputPath = outputDir / (backend + ".csv");
    std::ofstream output(outputPath, std::ios::trunc);
    if (!output)
        throw std::runtime_error("Could not write benchmark CSV: " + outputPath.string());
    output << "backend,device,p,resolution,operation,fps,seconds,loops,image\n" << std::setprecision(17);

    // warm up before each measurement
    const int warmupLoops = std::max(loops / 10, 3);
    constexpr auto warmupMinDuration = std::chrono::milliseconds(250);
    const auto warmup = [warmupLoops, warmupMinDuration](const auto& func) {
        const auto started = std::chrono::steady_clock::now();
        for (int i = 0; i < warmupLoops || std::chrono::steady_clock::now() - started < warmupMinDuration; i++)
            func();
    };

    cout << info(std::format("CLI benchmark: {} on {} ({} loops per measurement, >= {} warmup loops and >= {} ms)\n", backend, device, loops, warmupLoops, warmupMinDuration.count()));
    // Test each prediction order
    for (const int p : predictionOrders) {
        auto session = createImageSession(watermarkPassword, p, psnr);
        // Test each resolution image
        for (const auto& image : images) {
            if (!fs::is_regular_file(image.path))
                throw std::runtime_error("Benchmark image not found: " + string(image.path));
            loadImage(session.get(), image.path);

            // Measure embedding execution time (after warming up clocks, caches and memory pools)
            const auto embedOnce = [&]() {
                embedImage(session.get());
                finish();
            };
            warmup(embedOnce);
            const double embedSeconds = executionTime(embedOnce, loops, false) / loops;
            if (saveImages) {
                const fs::path imagesDir = outputDir / (backend + "_images");
                fs::create_directories(imagesDir);
                saveImageExact(session.get(), (imagesDir / std::format("p{}_{}.png", p, image.resolution)).string());
            }
            // Measure detection execution time and correlation
            prepareDetectionImage(session.get());
            float correlation = 0.0f;
            const auto detectOnce = [&]() { correlation = detectEmbeddedBuffer(session.get()); };
            warmup(detectOnce);
            const double detectSeconds = executionTime(detectOnce, loops, false) / loops;

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
    // the process code page is UTF-8 (utf8.manifest), the console must print the same bytes
    SetConsoleOutputCP(CP_UTF8);
    int exitCode = EXIT_SUCCESS;
    try {
        // Parse command-line options before initializing the backend
        const CommandLineOptions commandLine = parseCommandLine(argc, argv);
        if (commandLine.help || argc == 1) {
            printHelp();
            return EXIT_SUCCESS;
        }
        const Settings options(commandLine.settings);
        // initialize backend data (GPU devices, OpenMP threads, etc.)
        initializeEnvironment(options.GetInteger("compute", "gpu_device_id", 0));
        const float psnr = options.GetFloat("global", "psnr", 40.0f);
        if (!std::isfinite(psnr) || psnr <= 0)
            throw std::runtime_error("PSNR must be a finite number greater than 0");
        // Run standalone benchmark if requested
        if (commandLine.benchmark)
            return runCliBenchmark(options, psnr, commandLine.benchmarkSave);

        const int p = options.GetInteger("global", "p", 3);
        if (p != 3 && p != 5 && p != 7 && p != 9)
            throw std::runtime_error("p must be 3, 5, 7 or 9");
        // test algorithms
        const string videoFile = options.Get("video", "path", "");
        const string imageMode = options.Get("image", "mode", "single");
        if (!videoFile.empty())
            exitCode = testForVideo(options, videoFile, p, psnr);
        else if (imageMode == "batch_embed")
            exitCode = testForImageBatch(options, p, psnr, true);
        else if (imageMode == "batch_detect")
            exitCode = testForImageBatch(options, p, psnr, false);
        else if (imageMode == "single")
            exitCode = testForImageSingle(options, p, psnr);
        else
            throw std::runtime_error("Invalid mode. Use --image.mode single, batch_embed or batch_detect, or supply --video.path.");
    } catch (const std::exception& ex) {
        cout << err(string("Fatal error: ") + ex.what() + "\n");
        exitCode = EXIT_FAILURE;
    }
    // flush first
    cout.flush();
    return exitCode;
}
