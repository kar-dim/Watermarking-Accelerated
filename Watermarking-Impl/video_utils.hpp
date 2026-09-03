#pragma once

#include "include/WatermarkTypes.hpp"
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
 *  \brief  Utility functions for video, including decoding, frames processing and writing.
 *  \author Dimitris Karatzas
 */
namespace video_utils {
// High-bit-depth and HDR helpers. Rejects every known format whose components exceed 8 bits
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

// public API
AVCodecContextPtr openDecoder(const AVCodecParameters* inputCodecParams, bool useHwDecoderRequested, bool& useHwDecoder, AVRational pktTimebase);
int findVideoStream(const AVFormatContext* inputFormatCtx);
bool initFilterGraph(WatermarkCore::VideoSession* s);
int videoDispatcher(WatermarkCore::VideoSession* s, VideoMode op, bool needsFilter = false);
void initOutputEncoder(WatermarkCore::VideoSession* s);
void flushAndFinalize(WatermarkCore::VideoSession* s);
} // namespace video_utils
