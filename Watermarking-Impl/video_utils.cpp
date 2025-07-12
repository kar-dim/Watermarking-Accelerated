#include "buffer.hpp"
#include "utils.hpp"
#include "video_utils.hpp"
#include "videoprocessingcontext.hpp"
#include "WatermarkBase.hpp"
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <format>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include "libavcodec/codec_par.h"
#include "libavutil/frame.h"
#include "libavutil/pixfmt.h"
#include "libavutil/error.h"
#include "libavutil/avutil.h"
#include "libavcodec/codec.h"
#include "libavutil/rational.h"
}

#if defined(_USE_EIGEN_)
using namespace Eigen;
#endif

using std::string;
using std::cout;

namespace video_utils
{
	//main frames loop logic for video watermark embedding and detection
	int processFrames(const VideoProcessingContext& data, std::function<void(const AVFrame*, int&)> processFrame)
	{
		const AVPacketPtr packet(av_packet_alloc(), [](AVPacket* pkt) { av_packet_free(&pkt); });
		const AVFramePtr frame(av_frame_alloc(), [](AVFrame* frame) { av_frame_free(&frame); });
		int framesCount = 0;
		auto processValidFrame = [&](const AVFrame* f)
		{
			Utils::checkError(f->format != AV_PIX_FMT_YUV420P && f->format != AV_PIX_FMT_YUVJ420P,"Error: Video frame format not supported, aborting");
			processFrame(f, framesCount);
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
					throw std::runtime_error(string("FFmpeg decoding error: ") + errbuf);
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

	//embed watermark in a video frame
	void embedWatermark(const VideoProcessingContext& data, BufferType& inputFrame, GrayBuffer& watermarkedFrame, int& framesCount, const AVFrame* frame, FILE* ffmpegPipe)
	{
		const bool embedWatermark = framesCount % data.watermarkInterval == 0;
		//if there is row padding (for alignment), we must copy the data to a contiguous block!
		if (frame->linesize[0] != data.width)
		{
			if (embedWatermark)
			{
				for (int y = 0; y < data.height; y++)
					memcpy(data.inputFramePtr + y * data.width, frame->data[0] + y * frame->linesize[0], data.width);
				//embed the watermark, receive the watermarked data back to host and write the watermarked image data to ffmpeg pipe
				writeWatermarkeFrame(data, inputFrame, watermarkedFrame, frame, ffmpegPipe);
			}
			else
			{
				//write from frame buffer row-by-row the the valid image data (and not the alignment bytes)
				for (int y = 0; y < data.height; y++)
					fwrite(frame->data[0] + y * frame->linesize[0], 1, data.width, ffmpegPipe);
			}
			//always write UI planes as-is
			for (int y = 0; y < data.height / 2; y++)
				fwrite(frame->data[1] + y * frame->linesize[1], 1, data.width / 2, ffmpegPipe);
			for (int y = 0; y < data.height / 2; y++)
				fwrite(frame->data[2] + y * frame->linesize[2], 1, data.width / 2, ffmpegPipe);
		}
		//no row padding, read and write data directly
		else
		{
			writeConditionalWatermarkFrame(embedWatermark, data, inputFrame, watermarkedFrame, frame, ffmpegPipe);
			fwrite(frame->data[1], 1, data.width * frame->height / 4, ffmpegPipe);
			fwrite(frame->data[2], 1, data.width * frame->height / 4, ffmpegPipe);
		}
		framesCount++;
	}

	//detect the watermark for a video frame
	void detectWatermark(const VideoProcessingContext& data, BufferType& inputFrame, int& framesCount, const AVFrame* frame)
	{
		//detect watermark after X frames
		if (framesCount % data.watermarkInterval == 0)
		{
			//if there is row padding (for alignment), we must copy the data to a contiguous block!
			const bool rowPadding = frame->linesize[0] != data.width;
			if (rowPadding)
			{
				for (int y = 0; y < data.height; y++)
					memcpy(data.inputFramePtr + y * data.width, frame->data[0] + y * frame->linesize[0], data.width);
			}
			//supply the input frame to the GPU and run the detection of the watermark
#if defined(_USE_GPU_)
			inputFrame = GrayBuffer(data.width, data.height, rowPadding ? data.inputFramePtr : frame->data[0], afHost).T().as(f32);
#elif defined(_USE_EIGEN_)
			inputFrame = Map<GrayBuffer>(rowPadding ? data.inputFramePtr : frame->data[0], data.width, data.height).transpose().cast<float>();
#endif
			float correlation = data.watermarkObj->detectWatermark(inputFrame, ME);
			cout << "Correlation for frame: " << (framesCount + 1) << ": " << correlation << "\n";
		}
		framesCount++;
	}

	//find the first video stream index
	int findVideoStream(const AVFormatContext* inputFormatCtx)
	{
		for (unsigned int i = 0; i < inputFormatCtx->nb_streams; i++)
			if (inputFormatCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO)
				return i;
		return -1;
	}

	//open decoder context for video
	AVCodecContext* openDecoder(const AVCodecParameters* inputCodecParams)
	{
		const AVCodec* inputDecoder = avcodec_find_decoder(inputCodecParams->codec_id);
		AVCodecContext* inputDecoderCtx = avcodec_alloc_context3(inputDecoder);
		avcodec_parameters_to_context(inputDecoderCtx, inputCodecParams);
		//multithreading decode
		inputDecoderCtx->thread_count = 0;
		if (inputDecoder->capabilities & AV_CODEC_CAP_FRAME_THREADS)
			inputDecoderCtx->thread_type = FF_THREAD_FRAME;
		else if (inputDecoder->capabilities & AV_CODEC_CAP_SLICE_THREADS)
			inputDecoderCtx->thread_type = FF_THREAD_SLICE;
		else
			inputDecoderCtx->thread_count = 1; //don't use multithreading
		avcodec_open2(inputDecoderCtx, inputDecoder, nullptr);
		return inputDecoderCtx;
	}

	//get the input video FPS (average)
	string getFrameRate(const AVFormatContext* inputFormatCtx, const int videoStreamIndex)
	{
		const AVRational frameRate = inputFormatCtx->streams[videoStreamIndex]->avg_frame_rate;
		return std::format("{:.3f}", static_cast<float>(frameRate.num) / frameRate.den);
	}

	//runs the watermark creation for a video frame and writes the watermarked frame to the ffmpeg pipe
	void writeWatermarkeFrame(const VideoProcessingContext& data, BufferType& inputFrame, GrayBuffer& watermarkedFrame, const AVFrame* frame, FILE* ffmpegPipe)
	{
		float watermarkStrength;
#if defined(_USE_GPU_)
		inputFrame = BufferType(data.width, data.height, data.inputFramePtr, afHost).T().as(f32);
		watermarkedFrame = data.watermarkObj->makeWatermark(inputFrame, inputFrame, watermarkStrength, ME).as(u8).T();
		watermarkedFrame.host(data.inputFramePtr);
		fwrite(data.inputFramePtr, 1, data.width * frame->height, ffmpegPipe);
#elif defined(_USE_EIGEN_)
		inputFrame = Map<GrayBuffer>(data.inputFramePtr, data.width, data.height).transpose().cast<float>();
		watermarkedFrame = data.watermarkObj->makeWatermark(inputFrame, inputFrame, watermarkStrength, ME).getGray().transpose().cast<uint8_t>();
		fwrite(watermarkedFrame.data(), 1, data.width * frame->height, ffmpegPipe);
#endif
	}

	//runs the watermark creation for a video frame and writes the watermarked frame to the ffmpeg pipe, if the watermark is embedded, or writes the original frame data otherwise
	void writeConditionalWatermarkFrame(const bool embedWatermark, const VideoProcessingContext& data, BufferType& inputFrame, GrayBuffer& watermarkedFrame, const AVFrame* frame, FILE* ffmpegPipe)
	{
		if (embedWatermark)
		{
			float watermarkStrength;
#if defined(_USE_GPU_)
			inputFrame = BufferType(data.width, data.height, frame->data[0], afHost).T().as(f32);
			watermarkedFrame = data.watermarkObj->makeWatermark(inputFrame, inputFrame, watermarkStrength, ME).as(u8).T();
			watermarkedFrame.host(data.inputFramePtr);
		}
		fwrite(embedWatermark ? data.inputFramePtr : frame->data[0], 1, data.width * frame->height, ffmpegPipe);
#elif defined(_USE_EIGEN_)
			inputFrame = BufferType(Map<GrayBuffer>(frame->data[0], data.width, data.height).transpose().cast<float>());
			watermarkedFrame = data.watermarkObj->makeWatermark(inputFrame, inputFrame, watermarkStrength, ME).getGray().transpose().cast<uint8_t>();
	}
		fwrite(embedWatermark ? watermarkedFrame.data() : frame->data[0], 1, data.width* frame->height, ffmpegPipe);
#endif
	}
}