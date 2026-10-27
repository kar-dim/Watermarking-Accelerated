#include "common_utils.hpp"
#include "CorrelationFormat.hpp"
#include "WatermarkCore.hpp"
#include "VideoBatch.hpp"
#include "ImageBatch.hpp"
#include <functional>
#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <format>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <omp.h>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <syncstream>
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

template <typename F>
double executionTime(F&& func, int loops = 1, const bool warmup = true) {
    if (warmup)
        func(); // warmup once
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

constexpr std::array cliSettings = {
    OptionDefinition{"global",  "watermark_password"   },
    OptionDefinition{"global",  "p"                    },
    OptionDefinition{"global",  "psnr"                 },
    OptionDefinition{"global",  "display_fps"          },
    OptionDefinition{"compute", "gpu_device_id"        },
    OptionDefinition{"compute", "hw_encoder"           },
    OptionDefinition{"image",   "mode"                 },
    OptionDefinition{"image",   "path"                 },
    OptionDefinition{"image",   "devices"              },
    OptionDefinition{"image",   "output_path"          },
    OptionDefinition{"image",   "benchmark_loops"      },
    OptionDefinition{"image",   "benchmark_output_path"},
    OptionDefinition{"video",   "mode"                 },
    OptionDefinition{"video",   "path"                 },
    OptionDefinition{"video",   "encode_output_path"   },
    OptionDefinition{"video",   "encode_codec_options" },
    OptionDefinition{"video",   "hw_encode_options"    },
    OptionDefinition{"video",   "watermark_interval"   },
    OptionDefinition{"video",   "output_directory"     },
    OptionDefinition{"video",   "output_format"        },
    OptionDefinition{"video",   "devices"              },
    OptionDefinition{"video",   "accelerate_final"     }
};

string settingKey(const std::string_view section, const std::string_view name) { return string(section) + "." + string(name); }

struct CommandLineOptions {
    std::map<string, string> settings;
    bool benchmark = false;
    bool benchmarkSave = false;
    bool listGpus = false;
    bool help = false;
};

OptionDefinition resolveOption(const std::string_view option) {
    const size_t separator = option.find('.');
    if (separator != std::string_view::npos) {
        const std::string_view section = option.substr(0, separator);
        const std::string_view name = option.substr(separator + 1);
        for (const auto& definition : cliSettings)
            if (definition.section == section && definition.name == name)
                return definition;
        throw std::runtime_error("Unknown command-line setting '--" + string(option) + "'");
    }

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
        if (argument == "--list-gpu") {
            result.listGpus = true;
            continue;
        }
        if (argument == "--bench-save") {
            result.benchmark = true;
            result.benchmarkSave = true;
            continue;
        }
        if (argument == "--no-pause") {
            // keep accepting legacy flag, CLI commands never pause
            continue;
        }
        if (!argument.starts_with("--"))
            throw std::runtime_error("Unexpected positional argument '" + argument + "'");

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

        const auto definition = resolveOption(option);
        result.settings[settingKey(definition.section, definition.name)] = std::move(value);
    }
    return result;
}

void printHelp() {
    cout << R"(
Usage: Watermarking-CLI [options]

Configure the application with command-line options. With no options, show help.

Control options:
  --list-gpu                 List GPU indices and names for this backend, then exit.
                             No compute environment is started.
  --bench                    Benchmark embed/detect for p=3,5,7,9 and 480p..4K.
                             Writes thesis_data/{cuda,opencl,eigen}.csv.
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
  --hw_encoder BOOL          Use GPU video encoding (CUDA/OpenCL; default: false).

Image settings:
  --image.mode MODE          single, batch_embed, or batch_detect (default: single).
  --image.path PATH          Input image, or input directory for batch mode.
  --output_path FILE         Required destination file for single mode.
  --image.devices IDS        Batch devices, comma-separated indices, e.g. 0,1.
                             Defaults to --gpu_device_id, Eigen uses one worker.
  --benchmark_loops N        Positive measured loop count for --bench only.
  --benchmark_output_path FILE
                             CSV destination (default: thesis_data/BACKEND.csv).

Video settings:
  --video.mode MODE          embed, detect, batch_embed or batch_detect.
                             Defaults to embed.
  --video.path PATH          Input video, or a folder for a video batch.
  --encode_output_path FILE  Destination .mkv or .mp4 file for embed mode.
  --output_directory DIR     Batch output folder, defaults to watermark_output
                             inside the input folder. Existing files are kept.
  --output_format FORMAT     Batch embed container: mkv (default) or mp4.
  --video.devices IDS        Batch devices, comma-separated indices, e.g. 0,1.
                             Defaults to --gpu_device_id, Eigen uses one worker.
  --accelerate_final BOOL    Restart the final file on an idle selected GPU.
                             First complete output wins (default: true).
  --encode_codec_options STR Software encoder and options, e.g. -c:v libx265.
  --hw_encode_options STR    Hardware encoder/options, default follows selected GPU.
  --watermark_interval N     Embed or detect every Nth frame (N >= 1; default: 1).

Every option also accepts its group-qualified spelling, for example
--global.p=5 or --compute.gpu_device_id=1. Values may use '--key value' or
'--key=value'. The duplicated mode/path names must be group-qualified.

Example: Watermarking-CLI --image.path input.png --output_path output.png
         --watermark_password "your-password"
)";
}

class Settings {
  public:
    explicit Settings(const std::map<string, string>& values) : values_(values) {}

    string Get(const string& section, const string& name, const string& defaultValue) const {
        const auto value = values_.find(settingKey(section, name));
        return value == values_.end() ? defaultValue : value->second;
    }

    long GetInteger(const string& section, const string& name, const long defaultValue) const {
        const string value = Get(section, name, std::to_string(defaultValue));
        errno = 0;
        char* end = nullptr;
        const long parsed = std::strtol(value.c_str(), &end, 0);
        if (errno != 0 || end == value.c_str() || *end != '\0')
            throw std::runtime_error("Invalid integer for '" + section + "." + name + "': " + value);
        return parsed;
    }

    float GetFloat(const string& section, const string& name, const float defaultValue) const {
        const string value = Get(section, name, std::to_string(defaultValue));
        errno = 0;
        char* end = nullptr;
        const float parsed = std::strtof(value.c_str(), &end);
        if (errno != 0 || end == value.c_str() || *end != '\0' || !std::isfinite(parsed))
            throw std::runtime_error("Invalid number for '" + section + "." + name + "': " + value);
        return parsed;
    }

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
    cout << success(std::format("Embedded watermark in '{}' (p = {}, PSNR = {} dB).\n", outputPath, p, psnr));
    cout << std::format("Total image time (load, watermark setup, embed, save): {:.3f} seconds\n", totalSeconds);
    return EXIT_SUCCESS;
}

namespace {
std::mutex batchConsoleMutex;
std::function<void()> batchConsoleCancel;

BOOL WINAPI cancelBatchConsole(const DWORD event) {
    if (event != CTRL_C_EVENT && event != CTRL_BREAK_EVENT)
        return FALSE;
    const std::lock_guard lock(batchConsoleMutex);
    if (!batchConsoleCancel)
        return FALSE;
    batchConsoleCancel();
    return TRUE;
}

class BatchConsoleCancel {
  public:
    template <typename Control>
    explicit BatchConsoleCancel(Control& control) {
        const std::lock_guard lock(batchConsoleMutex);
        batchConsoleCancel = [&control] { control.cancel(); };
        registered_ = SetConsoleCtrlHandler(cancelBatchConsole, TRUE) != 0;
    }
    ~BatchConsoleCancel() {
        {
            // Finish any console callback before its captured controller can be destroyed
            const std::lock_guard lock(batchConsoleMutex);
            batchConsoleCancel = {};
        }
        if (registered_)
            SetConsoleCtrlHandler(cancelBatchConsole, FALSE);
    }

  private:
    bool registered_ = false;
};

std::string pathUtf8(const fs::path& path) {
    const auto bytes = path.u8string();
    return std::string(bytes.begin(), bytes.end());
}

class CliVideoBatchObserver : public VideoBatchObserver {
  public:
    explicit CliVideoBatchObserver(const std::vector<VideoBatchJob>& jobs) : jobs_(jobs) {}
    void onUpdate(const VideoBatchUpdate& update) override {
        const bool changed = states_[update.id] != update.state;
        states_[update.id] = update.state;
        const auto& job = jobs_.at(static_cast<size_t>(update.id - 1));
        const std::string name = pathUtf8(fs::u8path(job.inputPath).filename());
        if (update.state == VideoBatchState::Processing) {
            const int percent = update.progress.fraction >= 0 ? static_cast<int>(update.progress.fraction * 100) : -1;
            const auto now = std::chrono::steady_clock::now();
            if (!changed && (percent >= 0 ? percent < percentages_[update.id] + 10 : now - reported_[update.id] < std::chrono::seconds(5)))
                return;
            percentages_[update.id] = percent;
            reported_[update.id] = now;
            const std::string progress = percent >= 0 ? std::format("{:.1f}%", update.progress.fraction * 100) : "unknown total";
            std::osyncstream(cout) << std::format("[{}/{}] {}: {} | {} frames | {:.1f} FPS | device {}, attempt {}\n", update.id, jobs_.size(), name, progress, update.progress.frames,
                                          update.progress.fps, update.deviceIndex, update.attempt)
                                   << std::flush;
        } else if (changed) {
            std::string line = std::format("[{}/{}] {}: {}", update.id, jobs_.size(), videoBatchStateName(update.state), name);
            if (update.deviceIndex >= 0)
                line += std::format(" | device {}, attempt {}", update.deviceIndex, update.attempt);
            if (!update.detail.empty())
                line += " | " + update.detail;
            if (update.detections)
                line += std::format(" | {} samples, mean {}", update.detections, formatCorrelationPercent(update.meanCorrelation));
            std::osyncstream(cout) << (update.state == VideoBatchState::Failed ? err(line) : line) << '\n' << std::flush;
        }
    }
    void onLog(const std::uint64_t id, const std::string& message) override { std::osyncstream(cout) << std::format("  [{}] {}{}", id, message, message.ends_with('\n') ? "" : "\n") << std::flush; }

  private:
    const std::vector<VideoBatchJob>& jobs_;
    std::map<std::uint64_t, VideoBatchState> states_;
    std::map<std::uint64_t, int> percentages_;
    std::map<std::uint64_t, std::chrono::steady_clock::time_point> reported_;
};

std::vector<int> batchDevices(std::string_view value) {
    std::vector<int> devices;
    while (!value.empty()) {
        const auto comma = value.find(',');
        auto item = value.substr(0, comma);
        while (!item.empty() && std::isspace(static_cast<unsigned char>(item.front())))
            item.remove_prefix(1);
        while (!item.empty() && std::isspace(static_cast<unsigned char>(item.back())))
            item.remove_suffix(1);
        int device = -1;
        const auto parsed = std::from_chars(item.data(), item.data() + item.size(), device);
        if (parsed.ec != std::errc{} || parsed.ptr != item.data() + item.size() || device < 0)
            throw std::invalid_argument("Batch devices must be comma-separated nonnegative indices");
        devices.push_back(device);
        if (comma == std::string_view::npos)
            break;
        value.remove_prefix(comma + 1);
        if (value.empty())
            throw std::invalid_argument("A batch device index is missing after the comma");
    }
    return devices;
}

class CliImageBatchObserver : public ImageBatchObserver {
  public:
    explicit CliImageBatchObserver(const std::vector<ImageBatchJob>& jobs) : jobs_(jobs) {}
    void onUpdate(const ImageBatchUpdate& update) override {
        if (update.state != ImageBatchState::Completed && update.state != ImageBatchState::Failed && update.state != ImageBatchState::Cancelled)
            return;
        const auto& job = jobs_.at(static_cast<std::size_t>(update.id - 1));
        std::string line = std::format("[{}/{}] {}: {}", update.id, jobs_.size(), imageBatchStateName(update.state), pathUtf8(fs::u8path(job.inputPath).filename()));
        if (update.deviceIndex >= 0)
            line += std::format(" | device {}", update.deviceIndex);
        if (update.correlation)
            line += " | Correlation: " + formatCorrelationPercent(*update.correlation);
        else if (!update.detail.empty())
            line += " | " + update.detail;
        std::osyncstream(cout) << (update.state == ImageBatchState::Failed ? err(line) : line) << '\n' << std::flush;
    }

  private:
    const std::vector<ImageBatchJob>& jobs_;
};

int testForImageBatch(const Settings& options, const int p, const float psnr, const bool embed) {
    const auto input = fs::u8path(options.Get("image", "path", ""));
    if (!fs::is_directory(input))
        throw std::invalid_argument("Image batch path must be a directory");
    ImageBatchSettings settings;
    settings.password = options.Get("global", "watermark_password", "");
    settings.p = p;
    settings.psnr = psnr;
    settings.deviceIndex = options.GetInteger("compute", "gpu_device_id", 0);
    settings.detect = !embed;
    const auto devices = options.Get("image", "devices", "");
    if (!devices.empty())
        settings.deviceIndices = batchDevices(devices);
    std::vector<std::string> files;
    for (const auto& file : getValidImageFiles(input))
        files.push_back(pathUtf8(file));
    if (files.empty())
        throw std::invalid_argument("No supported images were found in the batch folder");
    const auto jobs = prepareImageBatch(files, pathUtf8(input / "watermark_output"), !embed, "W_ME");
    ImageBatchControl control;
    BatchConsoleCancel console(control);
    CliImageBatchObserver observer(jobs);
    cout << info(std::format("Found {} images. Starting batch {}...\n", jobs.size(), embed ? "embedding" : "detection"));
    const auto result = runImageBatch(settings, jobs, control, &observer);
    cout << info(std::format("Batch complete! Successfully processed {}/{} images.\n", result.succeeded, jobs.size()));
    cout << std::format("{} failed, {} cancelled | Total batch time: {:.3f} seconds\n", result.failed, result.cancelled, result.elapsedSeconds);
    return control.cancelled() ? 130 : result.failed ? EXIT_FAILURE : EXIT_SUCCESS;
}

int testForVideoBatch(const Settings& options, const std::string& input, const int p, const float psnr, const bool detect) {
    const fs::path path = fs::u8path(input);
    const bool directory = fs::is_directory(path);
    const auto files = directory ? getValidVideoFiles(path) : std::vector<fs::path>{path};
    std::vector<std::string> inputs;
    for (const auto& file : files)
        inputs.push_back(pathUtf8(file));
    const fs::path defaultOutput = (directory ? path : path.parent_path()) / "watermark_output";
    const auto jobs = prepareVideoBatch(inputs, options.Get("video", "output_directory", pathUtf8(defaultOutput)), detect, options.Get("video", "output_format", "mkv"));
    VideoBatchSettings settings;
    settings.deviceIndex = static_cast<int>(options.GetInteger("compute", "gpu_device_id", 0));
    settings.detect = detect;
    settings.deviceIndices = batchDevices(options.Get("video", "devices", ""));
    settings.accelerateFinal = options.GetBoolean("video", "accelerate_final", true);
    settings.video.watermarkPassword = options.Get("global", "watermark_password", "");
    settings.video.p = p;
    settings.video.psnr = psnr;
    settings.video.watermarkInterval = static_cast<int>(options.GetInteger("video", "watermark_interval", 1));
    settings.video.useHwDecoder = true;
    settings.video.useHwEncoder = !detect && options.GetBoolean("compute", "hw_encoder", false);
    settings.video.encodeOptions = settings.video.useHwEncoder ? options.Get("video", "hw_encode_options", "") : options.Get("video", "encode_codec_options", "-c:v libx265 -preset medium -crf 20");
    VideoBatchControl control;
    BatchConsoleCancel cancel(control);
    CliVideoBatchObserver observer(jobs);
    cout << info(std::format("Video batch: {} files on {} selected devices, Ctrl+C cancels the batch\n", jobs.size(), settings.deviceIndices.empty() ? 1 : settings.deviceIndices.size()));
    const auto summary = runVideoBatch(settings, jobs, control, &observer);
    cout << std::format("Batch complete: {} done, {} failed, {} skipped, {} cancelled ({:.1f} s)\n", summary.succeeded, summary.failed, summary.skipped, summary.cancelled, summary.elapsedSeconds);
    return summary.cancelled ? 130 : summary.failed ? EXIT_FAILURE : EXIT_SUCCESS;
}
} // namespace

static int testForVideo(const Settings& options, const string& videoFile, const int p, const float psnr) {
    const bool showFps = options.GetBoolean("global", "display_fps", true);
    const string videoMode = options.Get("video", "mode", "embed");
    if (videoMode == "batch_embed" || videoMode == "batch_detect")
        return testForVideoBatch(options, videoFile, p, psnr, videoMode == "batch_detect");
    checkError(videoMode != "embed" && videoMode != "detect", "Invalid video mode. Must be 'embed' or 'detect'.");
    const bool isEmbed = videoMode == "embed";

    VideoSettings settings;
    settings.videoFile = videoFile;
    settings.watermarkPassword = options.Get("global", "watermark_password", "");
    checkError(settings.watermarkPassword.empty(), "No valid watermark password specified!");
    settings.p = p;
    settings.psnr = psnr;
    settings.watermarkInterval = static_cast<int>(options.GetInteger("video", "watermark_interval", 1));
    settings.useHwDecoder = true;
    settings.useHwEncoder = options.GetBoolean("compute", "hw_encoder", false);
    settings.encodeOptions = settings.useHwEncoder ? options.Get("video", "hw_encode_options", "") : options.Get("video", "encode_codec_options", "-c:v libx265 -preset medium -crf 20");
    settings.encodeOutputPath = options.Get("video", "encode_output_path", "");
    if (isEmbed)
        validateVideoOutputPath(settings.encodeOutputPath);

    // encoder threads share CPU, limit OpenMP to physical cores to avoid oversubscription
    if (isEmbed)
        optimizeThreadsForVideoEmbedding();

    VideoHandle session = initVideo(settings);

    int framesProcessed = 0;
    double totalTime = 0;
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

static int runCliBenchmark(const Settings& settings, const float psnr, const bool saveImages) {
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
    // fewer iterations on CPU to keep bench duration bounded
    const int defaultLoops = backend == "eigen" ? 100 : 1000;
    const int loops = settings.GetInteger("image", "benchmark_loops", defaultLoops);
    checkError(loops <= 0, "benchmark_loops must be positive.");
    // fixed bench password keeps measurements independent of configuration
    const string watermarkPassword = settings.Get("global", "watermark_password", "benchmark-watermark-password");
    checkError(watermarkPassword.empty(), "No valid watermark password specified!");

    const fs::path outputDir = "thesis_data";
    fs::create_directories(outputDir);
    const fs::path outputPath = settings.Get("image", "benchmark_output_path", (outputDir / (backend + ".csv")).string());
    if (outputPath.has_parent_path())
        fs::create_directories(outputPath.parent_path());
    std::ofstream output(outputPath, std::ios::trunc);
    if (!output)
        throw std::runtime_error("Could not write benchmark CSV: " + outputPath.string());
    output << "backend,device,p,resolution,operation,fps,seconds,loops,image\n" << std::setprecision(17);

    const int warmupLoops = std::max(loops / 10, 3);
    constexpr auto warmupMinDuration = std::chrono::milliseconds(250);
    const auto warmup = [warmupLoops, warmupMinDuration](const auto& func) {
        const auto started = std::chrono::steady_clock::now();
        for (int i = 0; i < warmupLoops || std::chrono::steady_clock::now() - started < warmupMinDuration; i++)
            func();
    };

    cout << info(std::format("CLI benchmark: {} on {} ({} loops per measurement, >= {} warmup loops and >= {} ms)\n", backend, device, loops, warmupLoops, warmupMinDuration.count()));
    for (const int p : predictionOrders) {
        auto session = createImageSession(watermarkPassword, p, psnr);
        for (const auto& image : images) {
            if (!fs::is_regular_file(image.path))
                throw std::runtime_error("Benchmark image not found: " + string(image.path));
            loadImage(session.get(), image.path);

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
            prepareDetectionImage(session.get());
            float correlation = 0.0f;
            const auto detectOnce = [&]() { correlation = detectEmbeddedBuffer(session.get()); };
            warmup(detectOnce);
            const double detectSeconds = executionTime(detectOnce, loops, false) / loops;

            const auto writeResult = [&](const char* operation, const double seconds) {
                output << csvString(backend) << ',' << csvString(device) << ',' << p << ',' << csvString(image.resolution) << ',' << csvString(operation) << ',' << 1.0 / seconds << ',' << seconds
                       << ',' << loops << ',' << csvString(image.path) << '\n';
            };
            writeResult("embed", embedSeconds);
            writeResult("detect", detectSeconds);
            output.flush();
            cout << std::format(
                "p={}, {:>5}: embed {:>10.2f} FPS, detect {:>10.2f} FPS, correlation {}\n", p, image.resolution, 1.0 / embedSeconds, 1.0 / detectSeconds, formatCorrelationPercent(correlation));
        }
    }
    cout << success("Benchmark CSV written to " + outputPath.string() + "\n");
    return EXIT_SUCCESS;
}

/*!
 *  \brief  This is a project implementation of my Thesis with title:
 *			EFFICIENT IMPLEMENTATION OF WATERMARKING ALGORITHMS AND
 *			WATERMARK DETECTION IN IMAGE AND VIDEO USING GPU
 *  \author Dimitris Karatzas
 */
int main(const int argc, char* argv[]) {
    // the process code page is UTF-8 (utf8.manifest), the console must print the same bytes
    SetConsoleOutputCP(CP_UTF8);
    int exitCode = EXIT_SUCCESS;
    try {
        const CommandLineOptions commandLine = parseCommandLine(argc, argv);
        if (commandLine.help || argc == 1) {
            printHelp();
            return EXIT_SUCCESS;
        }
        if (commandLine.listGpus) {
            if (getBackendName() == "eigen") {
                cout << "No GPU devices: this is the Eigen/CPU backend.\n";
                return EXIT_SUCCESS;
            }
            const auto devices = getAvailableDevices();
            if (devices.empty())
                cout << "No GPU devices found.\n";
            for (size_t i = 0; i < devices.size(); ++i)
                cout << i << ": " << devices[i] << '\n';
            return EXIT_SUCCESS;
        }
        const Settings options(commandLine.settings);
        const string videoFile = options.Get("video", "path", "");
        const string videoMode = options.Get("video", "mode", "embed");
        const bool videoBatch = !commandLine.benchmark && !videoFile.empty() && (videoMode == "batch_embed" || videoMode == "batch_detect");
        const string imageMode = options.Get("image", "mode", "single");
        const bool imageBatch = !commandLine.benchmark && videoFile.empty() && (imageMode == "batch_embed" || imageMode == "batch_detect");
        if (!imageBatch && !options.Get("image", "devices", "").empty())
            throw std::invalid_argument("--image.devices applies to image batch modes");
        if (!videoBatch && !options.Get("video", "devices", "").empty())
            throw std::invalid_argument("--video.devices applies to video batch modes");
        // Batch workers choose their own devices, avoid an unused main-thread GPU context
        if (!videoBatch && !imageBatch)
            initializeEnvironment(options.GetInteger("compute", "gpu_device_id", 0));
        const float psnr = options.GetFloat("global", "psnr", 40.0f);
        if (!std::isfinite(psnr) || psnr <= 0)
            throw std::runtime_error("PSNR must be a finite number greater than 0");
        if (commandLine.benchmark)
            return runCliBenchmark(options, psnr, commandLine.benchmarkSave);

        const int p = options.GetInteger("global", "p", 3);
        if (p != 3 && p != 5 && p != 7 && p != 9)
            throw std::runtime_error("p must be 3, 5, 7 or 9");
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
    cout.flush();
    return exitCode;
}
