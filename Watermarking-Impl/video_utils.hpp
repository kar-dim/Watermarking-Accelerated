#pragma once

#include "include/WatermarkTypes.hpp"
#include "video_defines.hpp"
#include "VideoProcessingContext.hpp"
#include <string>

extern "C" {
#include "libavcodec/avcodec.h"
#include "libavcodec/codec_par.h"
#include "libavformat/avformat.h"
#include "libavutil/pixfmt.h"
}

/*!
 *  \brief  Utility functions for video, including decoding, frames processing and writing.
 *  \author Dimitris Karatzas
 */
namespace video_utils {
// 10-bit and HDR helpers
inline bool is10bit(const AVCodecContext* codecCtx, const AVStream* st) {
    const bool is10bitCtx = codecCtx->pix_fmt == AV_PIX_FMT_YUV420P10LE || codecCtx->pix_fmt == AV_PIX_FMT_YUV420P16LE;
    return is10bitCtx || (st->codecpar->format == AV_PIX_FMT_YUV420P10LE || st->codecpar->format == AV_PIX_FMT_YUV420P16LE || st->codecpar->bits_per_raw_sample == 10);
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
