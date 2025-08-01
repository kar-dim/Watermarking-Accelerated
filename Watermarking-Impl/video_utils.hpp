#pragma once

#include "buffer.hpp"
#include "videoprocessingcontext.hpp"
#include <cstdio>
#include <functional>
#include <memory>
#include <string>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include "libavcodec/codec_par.h"
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
	void embedWatermark(const VideoProcessingContext& data, BufferType& inputFrame, GrayBuffer& watermarkedFrame, int& framesCount, const AVFrame* frame, FILE* ffmpegPipe);
	void detectWatermark(const VideoProcessingContext& data, BufferType& inputFrame, int& framesCount, const AVFrame* frame);
	int processFrames(const VideoProcessingContext& data, std::function<void(const AVFrame*, int&)> processFrame);
	void writeWatermarkeFrame(const VideoProcessingContext& data, BufferType& inputFrame, GrayBuffer& watermarkedFrame, const AVFrame* frame, FILE* ffmpegPipe);
	void writeConditionalWatermarkFrame(const bool embedWatermark, const VideoProcessingContext& data, BufferType& inputFrame, GrayBuffer& watermarkedFrame, const AVFrame* frame, FILE* ffmpegPipe);
}