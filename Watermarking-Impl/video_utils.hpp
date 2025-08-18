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
	std::string getFrameRate(const AVFormatContext* inputFormatCtx, const int videoStreamIndex);
	void embedWatermark(VideoProcessingContext& data, int& framesCount, const AVFrame* frame, FILE* ffmpegPipe);
	int findVideoStream(const AVFormatContext* inputFormatCtx);
	void detectWatermark(VideoProcessingContext& data, int& framesCount, const AVFrame* frame);
	void embedAndWriteFrame(VideoProcessingContext& data, const BufferType& buffer, const int elements, FILE* ffmpegPipe);
	void processAndWriteYPlane(const bool embedWatermark, const AVFrame* frame, VideoProcessingContext& data, FILE* ffmpegPipe);
	void writeChromaPlanes(const AVFrame* frame, VideoProcessingContext& data, FILE* ffmpegPipe);
	void embedDispatcher(VideoProcessingContext& data, const bool useHwDecoder, FILE* ffmpegPipe);
	int detectDispatcher(VideoProcessingContext& data, const bool useHwDecoder);

	//main frames loop logic for video watermark embedding and detection
	template<bool HW_ACCEL = false, typename Func>
	int processFrames(const VideoProcessingContext& data, Func&& processFrame)
	{
		const AVPacketPtr packet(av_packet_alloc(), [](AVPacket* pkt) { av_packet_free(&pkt); });
		const AVFramePtr frame(av_frame_alloc(), [](AVFrame* frame) { av_frame_free(&frame); });
		int framesCount = 0;
		auto processValidFrame = [&]()
		{
			bool isValidFormat = std::ranges::any_of(supportedFormats, [&frame](auto f) { return f == frame->format; });
#if defined(_USE_CUDA_)
			if constexpr (HW_ACCEL) 
			{
				const auto hwFormat = ((AVHWFramesContext*)(frame->hw_frames_ctx->data))->sw_format;
				isValidFormat &= std::ranges::any_of(supportedHwFormats, [&hwFormat](auto f) { return f == hwFormat; });
			}
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