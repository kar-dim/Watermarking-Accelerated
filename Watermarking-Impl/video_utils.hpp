#pragma once

#include "utils.hpp"
#include "videoprocessingcontext.hpp"
#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>

#if defined(_USE_CUDA_)
#include "libavutil/hwcontext.h"
#endif

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

using AVPacketPtr = std::unique_ptr<AVPacket, std::function<void(AVPacket*)>>;
using AVFramePtr = std::unique_ptr<AVFrame, std::function<void(AVFrame*)>>;
using AVBufferRefPtr = std::unique_ptr<AVBufferRef, std::function<void(AVBufferRef*)>>;
using AVFormatContextPtr = std::unique_ptr<AVFormatContext, std::function<void(AVFormatContext*)>>;
using AVCodecContextPtr = std::unique_ptr<AVCodecContext, std::function<void(AVCodecContext*)>>;
using FILEPtr = std::unique_ptr<FILE, decltype(&_pclose)>;

/*!
 *  \brief  Utility functions for video, including decoding, frames processing and writing.
 *  \author Dimitris Karatzas
 */
namespace video_utils
{
#if defined(_USE_CUDA_)
	AVCodecContextPtr openDecoderHWAccel(const AVCodecParameters* inputCodecParams, const std::string& userHwDecoder, bool& useHwDecoder);
	void embedWatermarkHWAccel(VideoProcessingContext& data, int& framesCount, const AVFrame* frame, FILE* ffmpegPipe);
	void detectWatermarkHWAccel(VideoProcessingContext& data, int& framesCount, const AVFrame* frame);
#endif
	AVCodecContextPtr openDecoder(const AVCodecParameters* inputCodecParams);
	std::string getFrameRate(const AVFormatContext* inputFormatCtx, const int videoStreamIndex);
	void embedWatermark(VideoProcessingContext& data, int& framesCount, const AVFrame* frame, FILE* ffmpegPipe);
	int findVideoStream(const AVFormatContext* inputFormatCtx);
	void detectWatermark(VideoProcessingContext& data, int& framesCount, const AVFrame* frame);
	void processAndWriteYPlane(const bool embedWatermark, const AVFrame* frame, VideoProcessingContext& data, FILE* ffmpegPipe);
	void writeChromaPlanes(const bool rowPadding, const AVFrame* frame, VideoProcessingContext& data, FILE* ffmpegPipe);
	void loadInputFrame(VideoProcessingContext& data, uint8_t* hostPtr);

	//main frames loop logic for video watermark embedding and detection
	template<bool HW_ACCEL = false, typename Func>
	int processFrames(const VideoProcessingContext& data, Func&& processFrame)
	{
		const AVPacketPtr packet(av_packet_alloc(), [](AVPacket* pkt) { av_packet_free(&pkt); });
		const AVFramePtr frame(av_frame_alloc(), [](AVFrame* frame) { av_frame_free(&frame); });
		int framesCount = 0;
		auto processValidFrame = [&]()
		{
			static constexpr AVPixelFormat supportedFormats[] = { AV_PIX_FMT_YUV420P, AV_PIX_FMT_YUVJ420P, AV_PIX_FMT_CUDA };
			auto frameFormat = static_cast<AVPixelFormat>(frame->format);
			bool isValidFormat = std::ranges::any_of(supportedFormats, [frameFormat](auto f) { return f == frameFormat; });
#if defined(_USE_CUDA_)
			if constexpr (HW_ACCEL)
				isValidFormat = isValidFormat && ((AVHWFramesContext*)(frame->hw_frames_ctx->data))->sw_format == AV_PIX_FMT_NV12;
#endif
			Utils::checkError(!isValidFormat, "Error: Video frame format not supported, aborting");
			std::forward<Func>(processFrame)(frame.get(), framesCount);
		};
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
				processValidFrame();
			}
			av_packet_unref(packet.get());
		}
		//ensure all remaining frames are flushed
		avcodec_send_packet(data.inputDecoderCtx, nullptr);
		while (avcodec_receive_frame(data.inputDecoderCtx, frame.get()) == 0)
			processValidFrame();
		return framesCount;
	}
}