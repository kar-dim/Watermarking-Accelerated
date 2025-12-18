#pragma once

#include <memory>

extern "C" {
#include "libavcodec/avcodec.h"
#include "libavcodec/packet.h"
#include "libavfilter/avfilter.h"
#include "libavfilter/buffersrc.h"
#include "libavformat/avformat.h"
#include "libavutil/mem.h"
#include "libavutil/pixfmt.h"
}

namespace video_utils::detail
{
	template <auto FreeFunc>
	struct AVDeleter
	{
		template <typename T>
		void operator()(T* p) const noexcept { if (p) FreeFunc(&p); }
	};
	struct AVGenericDeleter { void operator()(void* p) const noexcept { if (p) av_free(p); } };
}

/*!
 *  \brief  Utility functions for video, including decoding, frames processing and writing.
 *  \author Dimitris Karatzas
 */
namespace video_utils
{
	enum VideoOp { EMBED, DETECT };

	using AVPacketPtr = std::unique_ptr<AVPacket, detail::AVDeleter<av_packet_free>>;
	using AVFramePtr = std::unique_ptr<AVFrame, detail::AVDeleter<av_frame_free>>;
	using AVBufferRefPtr = std::unique_ptr<AVBufferRef, detail::AVDeleter<av_buffer_unref>>;
	using AVFormatContextPtr = std::unique_ptr<AVFormatContext, detail::AVDeleter<avformat_close_input>>;
	using AVCodecContextPtr = std::unique_ptr<AVCodecContext, detail::AVDeleter<avcodec_free_context>>;
	using AVFilterInOutPtr = std::unique_ptr<AVFilterInOut, detail::AVDeleter<avfilter_inout_free>>;
	using AVFilterGraphPtr = std::unique_ptr<AVFilterGraph, detail::AVDeleter<avfilter_graph_free>>;
	using AVBufferSrcParametersPtr = std::unique_ptr<AVBufferSrcParameters, detail::AVGenericDeleter>;

	static constexpr AVPixelFormat supportedFormats[] = { AV_PIX_FMT_YUV420P, AV_PIX_FMT_YUVJ420P, AV_PIX_FMT_YUV420P10LE, AV_PIX_FMT_CUDA };
#if defined(_USE_CUDA_)
	static constexpr AVPixelFormat supportedHwFormats[] = { AV_PIX_FMT_NV12, AV_PIX_FMT_P010LE, AV_PIX_FMT_P016LE };
#endif
}