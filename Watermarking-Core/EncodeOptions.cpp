#include "EncodeOptions.hpp"
#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdlib>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

extern "C" {
#include "libavutil/dict.h"
#include "libavutil/mem.h"
}

/*!
 *  \brief  Parsing and sanitization of FFmpeg encoder options and codec tags
 *  \author Dimitris Karatzas
 */

namespace video_utils {
std::string hardwareEncoderDefaults(const std::uint32_t vendor) {
    switch (vendor) {
    case 0x8086: return "-c:v hevc_qsv -global_quality 20";
    case 0x1002: return "-c:v hevc_amf -quality balanced -rc cqp -qp_i 22 -qp_p 22 -g 120 -vbaq 0";
    case 0x10de: return "-c:v hevc_nvenc -preset p2 -tune hq -cq 28 -b:v 0";
    default: throw std::runtime_error("No default hardware encoder for this GPU, enter hardware encoder options or use CPU encoding");
    }
}

const AVCodec* findVideoEncoder(const std::string& name, const bool hardware) {
    if (name.empty())
        throw std::runtime_error("No video encoder in the options string, needs '-c:v <encoder>'");
    const AVCodec* codec = avcodec_find_encoder_by_name(name.c_str());
    if (!codec)
        throw std::runtime_error("Encoder not found: '" + name + "' (check the video encoder options)");
    const bool isHardware = (codec->capabilities & (AV_CODEC_CAP_HARDWARE | AV_CODEC_CAP_HYBRID)) != 0;
    if (hardware != isHardware)
        throw std::runtime_error(hardware ? "Hardware encoding is selected, but '" + name + "' is a software encoder. Choose a hardware encoder or turn hardware encoding off."
                                          : "Software encoding is selected, but '" + name + "' is a hardware encoder. Choose a software encoder or turn hardware encoding on.");
    return codec;
}

namespace {

// ffmpeg options split to tokens, respecting single and double quotes
struct OptionToken {
    std::string text;
    bool quoted = false;
};

std::vector<OptionToken> tokenizeOptions(const std::string& text) {
    std::vector<OptionToken> tokens;
    OptionToken current;
    bool started = false;
    char quote = 0;
    for (const char character : text) {
        if (quote != 0) {
            if (character == quote)
                quote = 0;
            else
                current.text += character;
            continue;
        }
        if (character == '"' || character == '\'') {
            if (!started)
                current.quoted = true;
            quote = character;
            started = true;
        } else if (std::isspace(static_cast<unsigned char>(character))) {
            if (started)
                tokens.push_back(current);
            current = {};
            started = false;
        } else {
            current.text += character;
            started = true;
        }
    }
    if (quote)
        throw std::invalid_argument("Unclosed quote in encoder options");
    if (started)
        tokens.push_back(current);
    return tokens;
}

// distinguishes numeric values (e.g. -1, -0.5) from flags (e.g. -crf, -preset)
bool isOptionValue(const std::string& token) {
    if (token.size() < 2 || token[0] != '-')
        return true;
    for (size_t index = 1; index < token.size(); ++index)
        if (std::isdigit(static_cast<unsigned char>(token[index])) == 0 && token[index] != '.')
            return false;
    return true;
}

// stream selection and other streams' codecs, audio and subtitles follow container rules
bool selectsStreams(const std::string& key) {
    static constexpr std::string_view kStreamKeys[] = {"map", "acodec", "scodec", "dcodec", "an", "sn", "dn", "vn", "pix_fmt", "pixel_format", "max_interleave_delta", "s", "video_size", "time_base"};
    return std::ranges::find(kStreamKeys, key) != std::end(kStreamKeys);
}

bool clashesWithPipeline(const std::string& key) {
    static constexpr std::string_view kOwned[] = {"color_range", "color_primaries", "color_trc", "colorspace", "aspect"};
    return std::ranges::find(kOwned, key) != std::end(kOwned);
}

} // namespace

ParsedEncodeOptions parseEncodeOptions(const std::string& text) {
    ParsedEncodeOptions result;
    // name[:stream specifier], output has one video stream, only v and v:0 apply
    const auto splitSpecifier = [](const std::string& key, bool& otherStream) {
        const size_t colon = key.find(':');
        if (colon == std::string::npos) {
            otherStream = false;
            return key;
        }
        const std::string_view stream = std::string_view(key).substr(colon + 1);
        otherStream = stream != "v" && stream != "v:0";
        return key.substr(0, colon);
    };
    const auto tokens = tokenizeOptions(text);
    for (size_t index = 0; index < tokens.size();) {
        if (isOptionValue(tokens[index].text)) {
            ++index;
            continue;
        }
        bool otherStream = false;
        const std::string key = splitSpecifier(tokens[index].text.substr(1), otherStream);
        const bool hasValue = index + 1 < tokens.size() && (tokens[index + 1].quoted || isOptionValue(tokens[index + 1].text));
        const size_t step = hasValue ? 2 : 1;
        if (otherStream || selectsStreams(key)) {
            result.ignored.push_back(tokens[index].text);
            index += step;
            continue;
        }
        if (!hasValue) {
            result.valueless.push_back(tokens[index].text);
            index += step;
            continue;
        }
        const std::string& value = tokens[index + 1].text;
        if (key == "c" || key == "codec" || key == "vcodec") {
            result.codecName = value;
        } else if (key == "tag" || key == "vtag") {
            result.codecTag = value;
        } else {
            if (clashesWithPipeline(key))
                result.overrides.push_back(tokens[index].text);
            av_dict_set(result.dictionary.ptr(), key.c_str(), value.c_str(), 0);
        }
        index += step;
    }
    return result;
}

void applyX265ThreadDefaults(ParsedEncodeOptions& parsed, const int cpuThreads) {
    if (cpuThreads <= 0 || parsed.codecName != "libx265")
        return;

    // FFmpeg's thread_count configures x265 frame threads but leaves its worker pool on auto
    OptionDict& options = parsed.dictionary;
    const auto* entry = av_dict_get(options.get(), "x265-params", nullptr, 0);
    OptionDict params;
    checkAv(av_dict_parse_string(params.ptr(), entry ? entry->value : nullptr, "=", ":", 0), "Invalid x265-params");

    bool hasPool = false;
    bool hasFrameThreads = av_dict_get(options.get(), "threads", nullptr, 0) != nullptr;
    const AVDictionaryEntry* param = nullptr;
    while ((param = av_dict_iterate(params.get(), param)) != nullptr) {
        std::string_view key = param->key;
        if (key.starts_with("--"))
            key.remove_prefix(2);
        hasPool |= key == "pools" || key == "numa-pools" || key == "numa_pools";
        hasFrameThreads |= key == "frame-threads" || key == "frame_threads";
    }

    const bool setFrameThreads = !hasFrameThreads && cpuThreads <= 2;
    if (hasPool && !setFrameThreads)
        return;
    if (!hasPool)
        checkAv(av_dict_set_int(params.ptr(), "pools", cpuThreads, 0), "Failed to set x265 worker pool size");
    if (setFrameThreads)
        checkAv(av_dict_set(params.ptr(), "frame-threads", "1", 0), "Failed to set x265 frame threads");

    char* rawParams = nullptr;
    checkAv(av_dict_get_string(params.get(), &rawParams, '=', ':'), "Failed to serialize x265-params");
    const std::unique_ptr<char, decltype(&av_free)> serialized(rawParams, av_free);
    checkAv(av_dict_set(options.ptr(), "x265-params", serialized.get(), 0), "Failed to set x265-params");
}

unsigned codecTagFromString(const std::string& tag) {
    char* end = nullptr;
    const unsigned long numeric = std::strtoul(tag.c_str(), &end, 0);
    if (end != tag.c_str() && end != nullptr && *end == '\0')
        return static_cast<unsigned>(numeric);
    unsigned packed = 0;
    for (size_t index = 0; index < 4 && index < tag.size(); ++index)
        packed |= static_cast<unsigned>(static_cast<unsigned char>(tag[index])) << (8 * index);
    return packed;
}

void reportParsedOptions(const ParsedEncodeOptions& parsed, const MediaLog& log) {
    if (!log)
        return;
    const auto report = [&log](const std::vector<std::string>& tokens, const std::string_view why) {
        if (tokens.empty())
            return;
        std::string joined;
        for (const std::string& token : tokens)
            joined += (joined.empty() ? "" : " ") + token;
        log("encode: option(s) '" + joined + "' " + std::string(why));
    };
    report(parsed.ignored, "ignored, application handles stream mapping and pixel format and preserves the default interleave limit");
    report(parsed.valueless, "dropped, no value follows");
    report(parsed.overrides, "overrides color/aspect tags taken from source, output may be mislabelled");
}

void reportUnusedOptions(const AVDictionary* options, const std::string& codecName, const char* containerName, const MediaLog& log) {
    if (!log)
        return;
    std::string joined;
    const AVDictionaryEntry* entry = nullptr;
    while ((entry = av_dict_iterate(options, entry)) != nullptr)
        joined += (joined.empty() ? "" : ", ") + std::string(entry->key) + " " + entry->value;
    if (!joined.empty())
        log("encode: neither '" + codecName + "' nor '" + std::string(containerName != nullptr ? containerName : "?") + "' container accepts '" + joined + "', dropped");
}

} // namespace video_utils
