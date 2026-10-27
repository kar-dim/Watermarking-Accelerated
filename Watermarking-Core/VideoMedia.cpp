#include "AvUtil.hpp"
#include "include/WatermarkCore.hpp"
#include "video_defines.hpp"
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <stop_token>
#include <string>
#include "VideoObserver.hpp"

extern "C" {
#include "libavcodec/codec_id.h"
#include "libavcodec/codec_par.h"
#include "libavformat/avformat.h"
#include "libavformat/avio.h"
#include "libavutil/avutil.h"
#include "libavutil/pixdesc.h"
#include "libavutil/pixfmt.h"
#include "libavutil/rational.h"
}

/*!
 * \brief Source video details and supported output container validation
 * \author Dimitris Karatzas
 */

namespace WatermarkCore {
namespace {
int interruptProbe(void* stop) { return static_cast<std::stop_token*>(stop)->stop_requested() ? 1 : 0; }

void checkProbeCancelled(const std::stop_token& stop) {
    if (stop.stop_requested())
        throw VideoCancelled();
}
} // namespace

void validateVideoOutputPath(const std::string& path) {
    std::string extension = std::filesystem::u8path(path).extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), [](const unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (extension != ".mkv" && extension != ".mp4")
        throw std::invalid_argument("Video output must be a .mkv or .mp4 file");
}

VideoSourceInfo probeVideoSource(const std::string& path, std::stop_token stop) {
    checkProbeCancelled(stop);
    AVFormatContext* rawInput = avformat_alloc_context();
    if (!rawInput)
        throw std::runtime_error("Could not allocate the video probe");
    rawInput->interrupt_callback = {interruptProbe, &stop};
    const int opened = avformat_open_input(&rawInput, path.c_str(), nullptr, nullptr);
    video_utils::AVFormatContextPtr input(rawInput);
    checkProbeCancelled(stop);
    video_utils::checkAv(opened, "Could not open the video for inspection");
    const int inspected = avformat_find_stream_info(input.get(), nullptr);
    checkProbeCancelled(stop);
    video_utils::checkAv(inspected, "Could not read the video stream details");
    const int index = av_find_best_stream(input.get(), AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    video_utils::checkAv(index, "No video stream found");
    const AVStream* stream = input->streams[index];
    const AVCodecParameters* parameters = stream->codecpar;
    VideoSourceInfo info;
    info.width = parameters->width;
    info.height = parameters->height;
    info.codec = avcodec_get_name(parameters->codec_id);
    if (const auto* descriptor = av_pix_fmt_desc_get(static_cast<AVPixelFormat>(parameters->format))) {
        info.bitDepth = descriptor->comp[0].depth;
        info.pixelFormat = descriptor->name;
    } else {
        info.bitDepth = parameters->bits_per_raw_sample;
        info.pixelFormat = "unknown format";
    }
    info.dynamicRange = parameters->color_trc == AVCOL_TRC_SMPTE2084 ? "PQ HDR" : parameters->color_trc == AVCOL_TRC_ARIB_STD_B67 ? "HLG HDR" : "SDR";
    const AVRational rate = stream->avg_frame_rate.num > 0 && stream->avg_frame_rate.den > 0 ? stream->avg_frame_rate : stream->r_frame_rate;
    if (rate.num > 0 && rate.den > 0)
        info.framesPerSecond = av_q2d(rate);
    if (input->duration > 0)
        info.durationSeconds = input->duration / static_cast<double>(AV_TIME_BASE);
    else if (stream->duration > 0 && stream->time_base.num > 0 && stream->time_base.den > 0)
        info.durationSeconds = stream->duration * av_q2d(stream->time_base);
    if (stream->nb_frames > 0)
        info.frames = stream->nb_frames;
    if (input->pb) {
        const std::int64_t bytes = avio_size(input->pb);
        if (bytes > 0)
            info.bytes = bytes;
    }
    for (unsigned track = 0; track < input->nb_streams; ++track) {
        switch (input->streams[track]->codecpar->codec_type) {
        case AVMEDIA_TYPE_AUDIO: ++info.audioTracks; break;
        case AVMEDIA_TYPE_SUBTITLE: ++info.subtitleTracks; break;
        case AVMEDIA_TYPE_ATTACHMENT: ++info.attachments; break;
        default: break;
        }
    }
    info.chapters = input->nb_chapters;
    checkProbeCancelled(stop);
    return info;
}
} // namespace WatermarkCore
