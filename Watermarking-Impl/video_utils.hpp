#pragma once

#include "videoprocessingcontext.hpp"
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include "libavcodec/codec_par.h"
#include "libavutil/pixfmt.h"
#include "libavutil/error.h"
#include "libavutil/frame.h"
#include "libavcodec/packet.h"
#include "libavutil/buffer.h"
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

	static constexpr AVPixelFormat supportedFormats[] = { AV_PIX_FMT_YUV420P, AV_PIX_FMT_YUVJ420P, AV_PIX_FMT_YUV420P10LE, AV_PIX_FMT_CUDA };
#if defined(_USE_CUDA_)
	static constexpr AVPixelFormat supportedHwFormats[] = { AV_PIX_FMT_NV12, AV_PIX_FMT_P010LE, AV_PIX_FMT_P016LE };
	AVCodecContextPtr openDecoderHWAccel(const AVCodecParameters* inputCodecParams, const std::string& userHwDecoder, bool& useHwDecoder);
	void embedWatermarkHWAccel(VideoProcessingContext& data, int& framesCount, const AVFrame* frame, FILE* ffmpegPipe);
	void detectWatermarkHWAccel(VideoProcessingContext& data, int& framesCount, const AVFrame* frame);
#endif
	inline uint8_t scale10to8(uint16_t value) { return static_cast<uint8_t>(value * 255 / 1023); }
	AVCodecContextPtr openDecoder(const AVCodecParameters* inputCodecParams, const std::string& userHwDecoder, bool& useHwDecoder);
	AVCodecContextPtr openSoftwareDecoder(const AVCodecParameters* inputCodecParams);
	bool checkPixelFormatSupport(const std::span<const AVPixelFormat> supportedFormats, const AVPixelFormat format);
	std::string getFrameRate(const AVStream *st);
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
	void embedAndWriteFrame(VideoProcessingContext& data, const BufferType& buffer, const int elements, FILE* ffmpegPipe);
	void processAndWriteYPlane(const bool embedWatermark, const AVFrame* frame, VideoProcessingContext& data, FILE* ffmpegPipe);
	void writeChromaPlanes(const AVFrame* frame, VideoProcessingContext& data, FILE* ffmpegPipe);
	int videoDispatcher(VideoProcessingContext& data, const bool useHwDecoder, const VideoOp op, FILE* ffmpegPipe = nullptr);

	//main frames loop logic for video watermark embedding and detection
	template<typename Func>
	int processFrames(const VideoProcessingContext& data, Func&& processFrame)
	{
		const AVPacketPtr packet(av_packet_alloc());
		const AVFramePtr frame(av_frame_alloc());
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
				std::forward<Func>(processFrame)(frame.get(), framesCount);
			}
			av_packet_unref(packet.get());
		}
		//ensure all remaining frames are flushed
		avcodec_send_packet(data.inputDecoderCtx, nullptr);
		while (avcodec_receive_frame(data.inputDecoderCtx, frame.get()) == 0)
			std::forward<Func>(processFrame)(frame.get(), framesCount);
		return framesCount;
	}

	template<typename BufferType, typename T>
	void loadInputFrame(VideoProcessingContext& data, T* hostPtr)
	{
#if defined(_USE_GPU_)
		data.inputFrame = BufferType(data.width, data.height, hostPtr, afHost).T().as(f32);
#else
		data.inputFrame = Eigen::Map<BufferType>(hostPtr, data.width, data.height).transpose().template cast<float>();
#endif
	}
}