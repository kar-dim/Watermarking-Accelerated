#pragma once

#include "utils.hpp"
#include "videoprocessingcontext.hpp"
#include <cerrno>
#include <cstdio>
#include <functional>
#include <memory>
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
}

using AVPacketPtr = std::unique_ptr<AVPacket, std::function<void(AVPacket*)>>;
using AVFramePtr = std::unique_ptr<AVFrame, std::function<void(AVFrame*)>>;
using AVFormatContextPtr = std::unique_ptr<AVFormatContext, std::function<void(AVFormatContext*)>>;
using AVCodecContextPtr = std::unique_ptr<AVCodecContext, std::function<void(AVCodecContext*)>>;
using FILEPtr = std::unique_ptr<FILE, decltype(&_pclose)>;

/*!
 *  \brief  Utility functions for video, including decoding, frames processing and writing.
 *  \author Dimitris Karatzas
 */
namespace video_utils
{
	int findVideoStream(const AVFormatContext* inputFormatCtx);
	AVCodecContext* openDecoder(const AVCodecParameters* params);
	std::string getFrameRate(const AVFormatContext* inputFormatCtx, const int videoStreamIndex);
	void embedWatermark(VideoProcessingContext& data, int& framesCount, const AVFrame* frame, FILE* ffmpegPipe);
	void detectWatermark(VideoProcessingContext& data, int& framesCount, const AVFrame* frame);
	void writeWatermarkeFrame(VideoProcessingContext& data, const AVFrame* frame, FILE* ffmpegPipe);
	void writeConditionalWatermarkFrame(const bool embedWatermark, VideoProcessingContext& data, const AVFrame* frame, FILE* ffmpegPipe);

	//main frames loop logic for video watermark embedding and detection
	template<typename Func>
	int processFrames(const VideoProcessingContext& data, Func&& processFrame)
	{
		const AVPacketPtr packet(av_packet_alloc(), [](AVPacket* pkt) { av_packet_free(&pkt); });
		const AVFramePtr frame(av_frame_alloc(), [](AVFrame* frame) { av_frame_free(&frame); });
		int framesCount = 0;
		auto processValidFrame = [&](const AVFrame* f)
		{
			Utils::checkError(f->format != AV_PIX_FMT_YUV420P && f->format != AV_PIX_FMT_YUVJ420P, "Error: Video frame format not supported, aborting");
			std::forward<Func>(processFrame)(f, framesCount);
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
				int ret = avcodec_receive_frame(data.inputDecoderCtx, frame.get());
				if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
					break;
				if (ret < 0)
				{
					char errbuf[256];
					av_strerror(ret, errbuf, sizeof(errbuf));
					av_packet_unref(packet.get());
					throw std::runtime_error(std::string("FFmpeg decoding error: ") + errbuf);
				}
				processValidFrame(frame.get());
			}
			av_packet_unref(packet.get());
		}
		//ensure all remaining frames are flushed
		avcodec_send_packet(data.inputDecoderCtx, nullptr);
		while (avcodec_receive_frame(data.inputDecoderCtx, frame.get()) == 0)
		{
			processValidFrame(frame.get());
		}
		return framesCount;
	}
}