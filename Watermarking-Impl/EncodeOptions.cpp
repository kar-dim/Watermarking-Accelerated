#include "EncodeOptions.hpp"
#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdlib>
#include <iterator>
#include <string>
#include <string_view>
#include <vector>

extern "C" {
#include "libavutil/dict.h"
}

namespace video_utils {
namespace {

// ffmpeg options split to tokens, respecting single and double quotes
std::vector<std::string> tokenizeOptions(const std::string& text) {
    std::vector<std::string> tokens;
    std::string current;
    bool started = false;
    char quote = 0;
    for (const char character : text) {
        if (quote != 0) {
            if (character == quote)
                quote = 0;
            else
                current += character;
            continue;
        }
        if (character == '"' || character == '\'') {
            quote = character;
            started = true;
        } else if (std::isspace(static_cast<unsigned char>(character))) {
            if (started)
                tokens.push_back(current);
            current.clear();
            started = false;
        } else {
            current += character;
            started = true;
        }
    }
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

bool clashesWithPipeline(const std::string& key) {
    static constexpr std::string_view kOwned[] = {"color_range", "color_primaries", "color_trc", "colorspace", "aspect"};
    return std::ranges::find(kOwned, key) != std::end(kOwned);
}

} // namespace

ParsedEncodeOptions parseEncodeOptions(const std::string& text) {
    ParsedEncodeOptions result;
    const auto splitSpecifier = [](const std::string& key, char& specifier) {
        specifier = 0;
        const size_t position = key.rfind(':');
        if (position != std::string::npos && position + 2 == key.size()) {
            const char value = key[position + 1];
            if (value == 'v' || value == 'a' || value == 's' || value == 'd' || value == 't') {
                specifier = value;
                return key.substr(0, position);
            }
        }
        return key;
    };
    const std::vector<std::string> tokens = tokenizeOptions(text);
    for (size_t index = 0; index < tokens.size();) {
        if (isOptionValue(tokens[index])) {
            ++index;
            continue;
        }
        char specifier = 0;
        const std::string key = splitSpecifier(tokens[index].substr(1), specifier);
        const bool hasValue = index + 1 < tokens.size() && isOptionValue(tokens[index + 1]);
        const size_t step = hasValue ? 2 : 1;
        if (key == "map" || (specifier != 0 && specifier != 'v') || key == "pix_fmt" || key == "pixel_format") {
            result.ignored.push_back(tokens[index]);
            index += step;
            continue;
        }
        if (!hasValue) {
            result.valueless.push_back(tokens[index]);
            index += step;
            continue;
        }
        const std::string& value = tokens[index + 1];
        if (key == "c" || key == "codec") {
            result.codecName = value;
        } else if (key == "tag" || key == "vtag") {
            result.codecTag = value;
        } else {
            if (clashesWithPipeline(key))
                result.overrides.push_back(tokens[index]);
            av_dict_set(result.dictionary.ptr(), key.c_str(), value.c_str(), 0);
        }
        index += step;
    }
    return result;
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
    report(parsed.ignored, "ignored, application handles audio/subtitles and pixel format directly");
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
