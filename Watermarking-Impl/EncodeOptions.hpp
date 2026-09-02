#pragma once

/*!
 *  \brief  FFmpeg encode options tokenizer and parser
 *  \author Dimitris Karatzas
 */

#include "AvUtil.hpp"
#include <string>
#include <vector>

extern "C" {
#include "libavutil/dict.h"
}

namespace video_utils {

struct ParsedEncodeOptions {
    std::string codecName;
    std::string codecTag;
    AVDictionary* dictionary = nullptr;
    std::vector<std::string> ignored;
    std::vector<std::string> valueless;
    std::vector<std::string> overrides;
};

ParsedEncodeOptions parseEncodeOptions(const std::string& text);
unsigned codecTagFromString(const std::string& tag);

// refused options (ignored, valueless, overrides)
void reportParsedOptions(const ParsedEncodeOptions& parsed, const MediaLog& log);

// what is left in the dictionary after the encoder/muxer have parsed their own options
void reportUnusedOptions(const AVDictionary* options, const std::string& codecName, const char* containerName, const MediaLog& log);

} // namespace video_utils
