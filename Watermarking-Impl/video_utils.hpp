#pragma once

#include "buffer.hpp"
#include "VideoProcessingContext.hpp"
#include <cerrno>
#include <cstdio>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>

extern "C" {
#include <libavcodec/avcodec.h>
#include "libavcodec/packet.h"
#include <libavformat/avformat.h>
#include "libavfilter/avfilter.h"
#include "libavcodec/codec_par.h"
#include "libavutil/pixfmt.h"
#include "libavutil/error.h"
#include "libavutil/frame.h"
#include "libavutil/buffer.h"
#include "libavutil/rational.h"
}

namespace video_utils::detail 
{
	template <auto FreeFunc>
	struct AVDeleter
	{
		template <typename T>
		void operator()(T* p) const noexcept { if (p) FreeFunc(&p); }
	};
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

	static constexpr AVPixelFormat supportedFormats[] = { AV_PIX_FMT_YUV420P, AV_PIX_FMT_YUVJ420P, AV_PIX_FMT_YUV420P10LE, AV_PIX_FMT_CUDA };

	//10-bit and HDR helpers
	inline bool is10bit(const AVCodecContext* codecCtx, const AVStream* st)
	{ 
		const bool is10bitCtx = codecCtx->pix_fmt == AV_PIX_FMT_YUV420P10LE || codecCtx->pix_fmt == AV_PIX_FMT_YUV420P16LE;
		return is10bitCtx || (st->codecpar->format == AV_PIX_FMT_YUV420P10LE || st->codecpar->format == AV_PIX_FMT_YUV420P16LE || st->codecpar->bits_per_raw_sample == 10);
	}
	//PQ HDR10 or HLG HDR
	inline bool isHDR(const AVCodecContext* codecCtx) 
	{ return codecCtx->color_trc == AVCOL_TRC_SMPTE2084 || codecCtx->color_trc == AVCOL_TRC_ARIB_STD_B67; }

#if defined(_USE_CUDA_)
	static constexpr AVPixelFormat supportedHwFormats[] = { AV_PIX_FMT_NV12, AV_PIX_FMT_P010LE, AV_PIX_FMT_P016LE };
	AVCodecContextPtr openDecoderHWAccel(const AVCodecParameters* inputCodecParams, const std::string& userHwDecoder, bool& useHwDecoder);
	void embedWatermarkHWAccel(VideoProcessingContext& data, int& framesCount, const AVFrame* frame, FILE* ffmpegPipe);
	void detectWatermarkHWAccel(VideoProcessingContext& data, int& framesCount, const AVFrame* frame);
#endif
	std::string getFilterGraphString(const AVCodecContext* codecCtx, const AVStream* st, const bool useHwDecoder);
	bool initFilterGraph(const AVCodecContext* inputDecoderCtx, const AVStream* st, const bool useHwDecoder, FilterGraphContext& filterCtx);
	void filterFrame(AVFramePtr& frame, const FilterGraphContext& ctx);

	AVCodecContextPtr openDecoder(const AVCodecParameters* inputCodecParams, const std::string& userHwDecoder, bool& useHwDecoder);
	AVCodecContextPtr openSoftwareDecoder(const AVCodecParameters* inputCodecParams);
	bool checkPixelFormatSupport(const std::span<const AVPixelFormat> supportedFormats, const AVPixelFormat format);
	std::string getFrameRate(const AVStream *st);
	AVRational getTimeBase(const AVStream* st);
	inline std::string getPixFmt(const AVStream* st)
	{
		return st->codecpar->color_range == AVCOL_RANGE_JPEG ? "-pix_fmt yuvj420p " : "-pix_fmt yuv420p ";
	}
	inline std::string getColorRange(const AVStream* st)
	{
		return st->codecpar->color_range == AVCOL_RANGE_JPEG ? "-color_range:v:0 pc " : "-color_range:v:0 tv ";
	}
	std::string getStreamRotation(const AVStream* st);
	int findVideoStream(const AVFormatContext* inputFormatCtx);
	void embedWatermark(VideoProcessingContext& data, int& framesCount, const AVFrame* frame, FILE* ffmpegPipe);
	void detectWatermark(VideoProcessingContext& data, int& framesCount, const AVFrame* frame);
	void embedAndWriteFrame(VideoProcessingContext& data, const ImageBuffer& buffer, const int elements, FILE* ffmpegPipe);
	void processAndWriteYPlane(const bool embedWatermark, const AVFrame* frame, VideoProcessingContext& data, FILE* ffmpegPipe);
	void writeChromaPlanes(const AVFrame* frame, VideoProcessingContext& data, FILE* ffmpegPipe);
	int videoDispatcher(VideoProcessingContext& data, const bool useHwDecoder, const VideoOp op, const bool needsFiltert = false, FILE* ffmpegPipe = nullptr);
	void filterFrame(AVFramePtr& frame, const FilterGraphContext& filterGraphContext);

	//main frames loop logic for video watermark embedding and detection
	template<bool needsFilter, typename Func>
	int processFrames(const VideoProcessingContext& data, Func&& processFrame)
	{
		const AVPacketPtr packet(av_packet_alloc());
		AVFramePtr frame(av_frame_alloc());
		int framesCount = 0;

		//read video frames loop
		while (av_read_frame(data.inputFormatCtx, packet.get()) >= 0)
		{
			if (packet->stream_index != data.videoStreamIndex || avcodec_send_packet(data.inputDecoderCtx, packet.get()) < 0)
			{
				av_packet_unref(packet.get());
				continue;
			}
			while (true)
			{
				const int ret = avcodec_receive_frame(data.inputDecoderCtx, frame.get());
				if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
					break;
				if (ret < 0)
				{
					char errbuf[256];
					av_strerror(ret, errbuf, sizeof(errbuf));
					av_packet_unref(packet.get());
					throw std::runtime_error(std::string("FFmpeg decoding error: ") + errbuf);
				}
				//optionally filter frame (10-bit to 8-bit conversion, HDR to SDR tonemapping)
				if constexpr (needsFilter)
					filterFrame(frame, data.filterGraphContext);
				std::forward<Func>(processFrame)(frame.get(), framesCount);
			}
			av_packet_unref(packet.get());
		}
		//ensure all remaining frames are flushed
		avcodec_send_packet(data.inputDecoderCtx, nullptr);
		while (avcodec_receive_frame(data.inputDecoderCtx, frame.get()) == 0)
		{
			if constexpr (needsFilter)
				filterFrame(frame, data.filterGraphContext);
			std::forward<Func>(processFrame)(frame.get(), framesCount);
		}
		return framesCount;
	}

	template<typename TYPE, typename T>
	void loadInputFrame(VideoProcessingContext& data, T* hostPtr)
	{
#if defined(_USE_GPU_)
		data.inputFrame = TYPE(data.width, data.height, hostPtr, afHost).T().as(f32);
#else
		data.inputFrame = Eigen::Map<TYPE>(hostPtr, data.width, data.height).transpose().template cast<float>();
#endif
	}
}