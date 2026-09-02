#pragma once

/*!
 *  \brief  Audio/subtitle remuxing and text subtitle transcoding
 *  \author Dimitris Karatzas
 */

#include "AvUtil.hpp"
#include "video_defines.hpp"
#include <functional>
#include <string>
#include <vector>

extern "C" {
#include "libavcodec/packet.h"
#include "libavformat/avformat.h"
}

namespace video_utils {

using AuxiliaryPacketSink = std::function<bool(AVPacket*)>;

struct AuxiliaryMuxSetup {
    AVFormatContext* input = nullptr;
    AVFormatContext* output = nullptr;
    std::string outputPath;
    int videoWidth = 0;
    int videoHeight = 0;
    MediaLog log;
    AuxiliaryPacketSink sink; // if empty -> write straight into the output container
};

bool validateAudioStreams(const AVFormatContext* input, const std::string& outputPath, std::string& error);

// Adds audio/subtitle streams before avformat_write_header(), then routes their packets while video is encoded
class AuxiliaryMux {
  public:
    AuxiliaryMux() = default;
    ~AuxiliaryMux() = default;
    AuxiliaryMux(const AuxiliaryMux&) = delete;
    AuxiliaryMux& operator=(const AuxiliaryMux&) = delete;

    bool configure(const AuxiliaryMuxSetup& setup, std::string& error);
    bool routePacket(AVPacket* packet, std::string& error);
    void setSink(AuxiliaryPacketSink sink) { sink_ = std::move(sink); }

    [[nodiscard]] bool configured() const { return input_ != nullptr && output_ != nullptr; }

  private:
    struct SubtitleTranscode {
        int inputStreamIndex = -1;
        int outputStreamIndex = -1;
        AVCodecContextPtr decoder;
        AVCodecContextPtr encoder;
    };

    bool copyStream(const AVStream* inputStream, unsigned inputIndex, std::string& error);
    bool addSubtitle(const AVStream* inputStream, unsigned inputIndex, const std::string& outputPath, int videoWidth, int videoHeight, std::string& error);
    bool transcodeSubtitle(SubtitleTranscode& transcode, AVPacket* packet, std::string& error);
    bool emit(AVPacket* packet, std::string& error);

    AVFormatContext* input_ = nullptr;
    AVFormatContext* output_ = nullptr;
    MediaLog log_;
    AuxiliaryPacketSink sink_;
    std::vector<int> inputToOutput_;
    std::vector<SubtitleTranscode> subtitleTranscodes_;
};

} // namespace video_utils
