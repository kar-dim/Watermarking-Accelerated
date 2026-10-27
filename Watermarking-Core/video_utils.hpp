#pragma once

#include "video_defines.hpp"
#include "VideoProcessingContext.hpp"
#include <string>

extern "C" {
#include "libavcodec/avcodec.h"
#include "libavcodec/codec_par.h"
#include "libavformat/avformat.h"
#include "libavutil/pixdesc.h"
#include "libavutil/pixfmt.h"
}

/*!
 *  \brief  Video decoding, watermark processing and encoding utilities
 *  \author Dimitris Karatzas
 */
namespace video_utils {
// rejects formats whose components exceed 8 bits
inline bool is10bit(const AVCodecContext* codecCtx, const AVStream* st) {
    const auto exceeds8Bits = [](const AVPixelFormat format) {
        if (format == AV_PIX_FMT_NONE)
            return false;
        const AVPixFmtDescriptor* descriptor = av_pix_fmt_desc_get(format);
        return descriptor && descriptor->nb_components > 0 && descriptor->comp[0].depth > 8;
    };
    return exceeds8Bits(codecCtx->pix_fmt) || exceeds8Bits(codecCtx->sw_pix_fmt) || exceeds8Bits(static_cast<AVPixelFormat>(st->codecpar->format)) || st->codecpar->bits_per_raw_sample > 8;
}
// PQ HDR10 or HLG HDR
inline bool isHDR(const AVCodecContext* codecCtx) { return codecCtx->color_trc == AVCOL_TRC_SMPTE2084 || codecCtx->color_trc == AVCOL_TRC_ARIB_STD_B67; }

enum class VideoMode { EMBED, DETECT };

AVCodecContextPtr openDecoder(WatermarkCore::VideoSession* session);
int findVideoStream(const AVFormatContext* inputFormatCtx);
bool initFilterGraph(WatermarkCore::VideoSession* s);
int videoDispatcher(WatermarkCore::VideoSession* s, VideoMode op, bool needsFilter = false);
void initOutputEncoder(WatermarkCore::VideoSession* s);
void flushAndFinalize(WatermarkCore::VideoSession* s, bool replaceExistingOutput = true);
// failed embed closes and deletes temporary output file
void discardOutput(WatermarkCore::VideoSession* s) noexcept;
void checkCancelled(const WatermarkCore::VideoSession* s);
int interruptVideo(void* session);
void reportProgress(WatermarkCore::VideoSession* s, WatermarkCore::VideoState state, bool force = true, const AVFrame* frame = nullptr);
void previewVideoFrame(WatermarkCore::VideoSession* s, const AVFrame* frame, bool encoded);
void videoLog(const WatermarkCore::VideoSession* s, const std::string& text);
} // namespace video_utils
