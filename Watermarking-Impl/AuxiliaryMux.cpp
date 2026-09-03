#include "AuxiliaryMux.hpp"
#include "AvUtil.hpp"
#include "video_defines.hpp"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <format>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

extern "C" {
#include "libavcodec/avcodec.h"
#include "libavcodec/codec.h"
#include "libavcodec/codec_desc.h"
#include "libavcodec/codec_id.h"
#include "libavcodec/codec_par.h"
#include "libavcodec/defs.h"
#include "libavcodec/packet.h"
#include "libavformat/avformat.h"
#include "libavutil/avutil.h"
#include "libavutil/dict.h"
#include "libavutil/mathematics.h"
#include "libavutil/mem.h"
}

namespace video_utils {
namespace {

// mp4 and mov reject TrueHD audio muxing without experimental flags
bool muxerRefusesAsExperimental(const AVOutputFormat* format, const AVCodecID codec) {
    const std::string_view name = (format != nullptr && format->name != nullptr) ? format->name : "";
    return (name == "mp4" || name == "mov") && codec == AV_CODEC_ID_TRUEHD;
}

// readable codec descriptor name
std::string codecName(const AVCodecID codec) {
    const AVCodecDescriptor* descriptor = avcodec_descriptor_get(codec);
    return (descriptor != nullptr && descriptor->name != nullptr) ? descriptor->name : "unknown";
}

// find compatible text subtitle encoder for target container
AVCodecID pickContainerTextSubtitleCodec(const AVOutputFormat* format, const std::string& path) {
    const AVCodecID candidates[]{av_guess_codec(format, nullptr, path.c_str(), nullptr, AVMEDIA_TYPE_SUBTITLE), AV_CODEC_ID_MOV_TEXT, AV_CODEC_ID_SUBRIP, AV_CODEC_ID_ASS, AV_CODEC_ID_WEBVTT};
    for (const AVCodecID codec : candidates) {
        const AVCodecDescriptor* descriptor = (codec != AV_CODEC_ID_NONE) ? avcodec_descriptor_get(codec) : nullptr;
        if (descriptor != nullptr && (descriptor->props & AV_CODEC_PROP_TEXT_SUB) != 0 && avformat_query_codec(format, codec, FF_COMPLIANCE_NORMAL) == 1 && avcodec_find_encoder(codec) != nullptr) {
            return codec;
        }
    }
    return AV_CODEC_ID_NONE;
}

// update PlayRes header to match scaled video canvas so fonts do not blow up
std::string retargetAssPlayResolution(const std::string& header, const int width, const int height) {
    std::string result;
    std::istringstream input(header);
    for (std::string line; std::getline(input, line);) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.rfind("PlayResX:", 0) == 0) {
            line = "PlayResX: " + std::to_string(width);
        } else if (line.rfind("PlayResY:", 0) == 0) {
            line = "PlayResY: " + std::to_string(height);
        }
        result += line + "\r\n";
    }
    return result;
}

// duplicate chapter markers and metadata into output container
bool copyChapters(const AVFormatContext* input, AVFormatContext* output) {
    if (input->nb_chapters == 0) {
        return true;
    }
    if (output->chapters != nullptr || output->nb_chapters != 0) {
        return false;
    }
    auto** chapters = static_cast<AVChapter**>(av_calloc(input->nb_chapters, sizeof(AVChapter*)));
    if (chapters == nullptr) {
        return false;
    }
    unsigned copied = 0;
    for (unsigned index = 0; index < input->nb_chapters; ++index) {
        const AVChapter* source = input->chapters[index];
        auto* destination = static_cast<AVChapter*>(av_mallocz(sizeof(AVChapter)));
        // partial chapter tables silently lose markers, fail instead so the caller can report it
        if (destination == nullptr) {
            for (unsigned done = 0; done < copied; ++done) {
                av_dict_free(&chapters[done]->metadata);
                av_freep(&chapters[done]);
            }
            av_freep(&chapters);
            return false;
        }
        destination->id = source->id;
        destination->time_base = source->time_base;
        destination->start = source->start;
        destination->end = source->end;
        av_dict_copy(&destination->metadata, source->metadata, 0);
        chapters[copied++] = destination;
    }
    output->chapters = chapters;
    output->nb_chapters = copied;
    return true;
}

} // namespace

// verify that target container accepts all input audio codecs
bool validateAudioStreams(const AVFormatContext* input, const std::string& outputPath, std::string& error) {
    const AVOutputFormat* format = av_guess_format(nullptr, outputPath.c_str(), nullptr);
    if (format == nullptr) {
        error = "Cannot determine the output container from '" + outputPath + "'.";
        return false;
    }
    for (unsigned index = 0; index < input->nb_streams; ++index) {
        const AVCodecParameters* parameters = input->streams[index]->codecpar;
        if (parameters->codec_type != AVMEDIA_TYPE_AUDIO) {
            continue;
        }
        if (avformat_query_codec(format, parameters->codec_id, FF_COMPLIANCE_NORMAL) == 1 && !muxerRefusesAsExperimental(format, parameters->codec_id)) {
            continue;
        }
        error = std::format("Audio stream #{} is {}, which the {} container cannot store. Write a .mkv instead, Matroska accepts every audio codec.", index, codecName(parameters->codec_id),
                            (format->name != nullptr) ? format->name : "requested");
        return false;
    }
    return true;
}

// configure audio and subtitle stream mapping for output container
bool AuxiliaryMux::configure(const AuxiliaryMuxSetup& setup, std::string& error) {
    if (setup.input == nullptr || setup.output == nullptr || setup.output->oformat == nullptr) {
        error = "Auxiliary mux configuration received an invalid format context.";
        return false;
    }
    if (!validateAudioStreams(setup.input, setup.outputPath, error)) {
        return false;
    }
    input_ = setup.input;
    output_ = setup.output;
    log_ = setup.log;
    sink_ = setup.sink;
    inputToOutput_.assign(input_->nb_streams, -1);

    for (unsigned index = 0; index < input_->nb_streams; ++index) {
        const AVStream* stream = input_->streams[index];
        if (stream->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
            if (!copyStream(stream, index, error)) {
                return false;
            }
        } else if (stream->codecpar->codec_type == AVMEDIA_TYPE_SUBTITLE) {
            if (!addSubtitle(stream, index, setup.outputPath, setup.videoWidth, setup.videoHeight, error)) {
                return false;
            }
        }
    }

    // copy global metadata and chapter marks
    av_dict_copy(&output_->metadata, input_->metadata, 0);
    if (!copyChapters(input_, output_)) {
        error = "Could not allocate the output chapter table.";
        return false;
    }
    output_->max_interleave_delta = 0;
    return true;
}

// clone stream parameters, metadata and disposition into output
bool AuxiliaryMux::copyStream(const AVStream* inputStream, const unsigned inputIndex, std::string& error) {
    AVStream* outputStream = avformat_new_stream(output_, nullptr);
    if (outputStream == nullptr || avcodec_parameters_copy(outputStream->codecpar, inputStream->codecpar) < 0) {
        error = "Could not create output stream for input stream #" + std::to_string(inputIndex) + ".";
        return false;
    }
    outputStream->codecpar->codec_tag = 0;
    av_dict_copy(&outputStream->metadata, inputStream->metadata, 0);
    outputStream->disposition = inputStream->disposition;
    outputStream->time_base = inputStream->time_base;
    inputToOutput_[inputIndex] = outputStream->index;
    return true;
}

// add or transcode subtitle stream for target container
bool AuxiliaryMux::addSubtitle(const AVStream* inputStream, const unsigned inputIndex, const std::string& outputPath, const int videoWidth, const int videoHeight, std::string& error) {
    const AVCodecID sourceCodec = inputStream->codecpar->codec_id;
    // copy directly if output container natively supports the subtitle codec
    if (avformat_query_codec(output_->oformat, sourceCodec, FF_COMPLIANCE_NORMAL) == 1) {
        return copyStream(inputStream, inputIndex, error);
    }

    // drop incompatible non-text streams
    const AVCodecDescriptor* sourceDescriptor = avcodec_descriptor_get(sourceCodec);
    const AVCodecID destinationCodec = pickContainerTextSubtitleCodec(output_->oformat, outputPath);
    const AVCodec* decoder = avcodec_find_decoder(sourceCodec);
    const AVCodec* encoder = (destinationCodec != AV_CODEC_ID_NONE) ? avcodec_find_encoder(destinationCodec) : nullptr;
    const auto drop = [&](const std::string& why) {
        if (log_) {
            log_(std::format("subtitle stream #{} ({}) {}, dropping it", inputIndex, codecName(sourceCodec), why));
        }
        return true;
    };
    if (sourceDescriptor == nullptr || (sourceDescriptor->props & AV_CODEC_PROP_TEXT_SUB) == 0 || decoder == nullptr || encoder == nullptr) {
        return drop("is not storable in this container and cannot be converted");
    }

    // setup decoder and encoder contexts
    AVCodecContextPtr decoderContext(avcodec_alloc_context3(decoder));
    AVCodecContextPtr encoderContext(avcodec_alloc_context3(encoder));
    if (!decoderContext || !encoderContext || avcodec_parameters_to_context(decoderContext.get(), inputStream->codecpar) < 0) {
        error = "Could not allocate subtitle conversion for input stream #" + std::to_string(inputIndex) + ".";
        return false;
    }
    decoderContext->pkt_timebase = inputStream->time_base;
    if (avcodec_open2(decoderContext.get(), decoder, nullptr) < 0) {
        return drop("decoder open failed");
    }
    if (decoderContext->subtitle_header != nullptr && decoderContext->subtitle_header_size > 0) {
        std::string header(reinterpret_cast<const char*>(decoderContext->subtitle_header), decoderContext->subtitle_header_size);
        if (sourceCodec == AV_CODEC_ID_MOV_TEXT && destinationCodec == AV_CODEC_ID_ASS) {
            header = retargetAssPlayResolution(header, videoWidth, videoHeight);
        }
        encoderContext->subtitle_header = static_cast<std::uint8_t*>(av_mallocz(header.size() + 1));
        if (encoderContext->subtitle_header == nullptr) {
            error = "Could not allocate converted subtitle header for input stream #" + std::to_string(inputIndex) + ".";
            return false;
        }
        std::memcpy(encoderContext->subtitle_header, header.data(), header.size());
        encoderContext->subtitle_header_size = static_cast<int>(header.size());
    }
    encoderContext->time_base = AVRational{1, 1000};
    if ((output_->oformat->flags & AVFMT_GLOBALHEADER) != 0) {
        encoderContext->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }
    if (avcodec_open2(encoderContext.get(), encoder, nullptr) < 0) {
        return drop("encoder open failed");
    }
    AVStream* outputStream = avformat_new_stream(output_, nullptr);
    if (outputStream == nullptr || avcodec_parameters_from_context(outputStream->codecpar, encoderContext.get()) < 0) {
        error = "Could not create the converted subtitle output stream for input stream #" + std::to_string(inputIndex) + ".";
        return false;
    }
    outputStream->codecpar->codec_tag = 0;
    av_dict_copy(&outputStream->metadata, inputStream->metadata, 0);
    outputStream->disposition = inputStream->disposition;
    outputStream->time_base = encoderContext->time_base;
    inputToOutput_[inputIndex] = outputStream->index;
    subtitleTranscodes_.push_back(SubtitleTranscode{static_cast<int>(inputIndex), outputStream->index, std::move(decoderContext), std::move(encoderContext)});
    if (log_) {
        log_(std::format("subtitle stream #{} converted from {} to {}", inputIndex, codecName(sourceCodec), codecName(destinationCodec)));
    }
    return true;
}

// route auxiliary packet to output stream with timestamp rescaling
bool AuxiliaryMux::routePacket(AVPacket* packet, std::string& error) {
    if (!configured() || packet == nullptr || packet->stream_index < 0 || packet->stream_index >= static_cast<int>(inputToOutput_.size())) {
        error = "Auxiliary mux received a packet with an invalid stream index.";
        return false;
    }
    const int inputIndex = packet->stream_index;
    const AVMediaType type = input_->streams[inputIndex]->codecpar->codec_type;
    if (type != AVMEDIA_TYPE_AUDIO && type != AVMEDIA_TYPE_SUBTITLE) {
        return true;
    }
    const int outputIndex = inputToOutput_[inputIndex];
    if (outputIndex < 0) {
        return true;
    }
    const AVRational inputTimeBase = input_->streams[inputIndex]->time_base;
    const auto transcode = std::find_if(subtitleTranscodes_.begin(), subtitleTranscodes_.end(), [inputIndex](const SubtitleTranscode& state) { return state.inputStreamIndex == inputIndex; });
    if (transcode != subtitleTranscodes_.end()) {
        return transcodeSubtitle(*transcode, packet, error);
    }
    av_packet_rescale_ts(packet, inputTimeBase, output_->streams[outputIndex]->time_base);
    packet->stream_index = outputIndex;
    packet->pos = -1;
    return emit(packet, error);
}

// decode, re-encode and timestamp (align) subtitle packet
bool AuxiliaryMux::transcodeSubtitle(SubtitleTranscode& transcode, AVPacket* packet, std::string& error) {
    AVSubtitle subtitle{};
    int decoded = 0;
    if (avcodec_decode_subtitle2(transcode.decoder.get(), &subtitle, &decoded, packet) < 0 || decoded == 0) {
        return true;
    }
    bool success = true;
    if (subtitle.pts != AV_NOPTS_VALUE) {
        subtitle.pts += av_rescale_q(subtitle.start_display_time, AVRational{1, 1000}, AV_TIME_BASE_Q);
        subtitle.end_display_time -= subtitle.start_display_time;
        subtitle.start_display_time = 0;
        constexpr size_t kSubtitleScratchCapacity = 1 << 20;
        if (transcode.encodeScratch.empty()) {
            transcode.encodeScratch.resize(kSubtitleScratchCapacity);
        }
        // encode into the reused scratch first, then allocate a packet of exactly the produced size
        const int bytes = avcodec_encode_subtitle(transcode.encoder.get(), transcode.encodeScratch.data(), static_cast<int>(transcode.encodeScratch.size()), &subtitle);
        AVPacketPtr outputPacket(av_packet_alloc());
        if (bytes > 0 && outputPacket && av_new_packet(outputPacket.get(), bytes) == 0) {
            std::memcpy(outputPacket->data, transcode.encodeScratch.data(), static_cast<size_t>(bytes));
            const AVRational outputTimeBase = output_->streams[transcode.outputStreamIndex]->time_base;
            outputPacket->stream_index = transcode.outputStreamIndex;
            outputPacket->pts = av_rescale_q(subtitle.pts, AV_TIME_BASE_Q, outputTimeBase);
            outputPacket->dts = outputPacket->pts;
            outputPacket->duration = av_rescale_q(subtitle.end_display_time, AVRational{1, 1000}, outputTimeBase);
            outputPacket->pos = -1;
            success = emit(outputPacket.get(), error);
        }
    }
    avsubtitle_free(&subtitle);
    return success;
}

// give the packet to sink (or write directly if no sink specified)
bool AuxiliaryMux::emit(AVPacket* packet, std::string& error) {
    if (sink_) {
        if (!sink_(packet)) {
            error = "Writing an auxiliary stream packet failed.";
            return false;
        }
        return true;
    }
    const int status = av_interleaved_write_frame(output_, packet);
    if (status < 0) {
        error = "Writing an auxiliary stream packet failed: " + avErrorText(status);
        return false;
    }
    return true;
}

} // namespace video_utils
