#pragma once

#if defined(_USE_EIGEN_)
#include <cstdint>
#endif

#include "buffer.hpp"
#include "videoprocessingcontext.hpp"
#include "WatermarkBase.hpp"
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

int findVideoStreamIndex(const AVFormatContext* inputFormatCtx);
AVCodecContext* openDecoderContext(const AVCodecParameters* params);
std::string getVideoFrameRate(const AVFormatContext* inputFormatCtx, const int videoStreamIndex);
void embedWatermarkFrame(const VideoProcessingContext& data, BufferType& inputFrame, GrayBuffer& watermarkedFrame, int& framesCount, AVFrame* frame, FILE* ffmpegPipe);
void detectFrameWatermark(const VideoProcessingContext& data, BufferType& inputFrame, int& framesCount, AVFrame* frame);
int processFrames(const VideoProcessingContext& data, std::function<void(AVFrame*, int&)> processFrame);
void makeRgbWatermarkBuffer(const std::unique_ptr<WatermarkBase>& watermarkObj, const BufferType& image, const BufferType& rgbImage, BufferType& output, float& watermarkStrength, MASK_TYPE maskType);
void writeWatermarkeFrameToPipe(const VideoProcessingContext& data, BufferType& inputFrame, GrayBuffer& watermarkedFrame, AVFrame* frame, FILE* ffmpegPipe);
void writeConditionallyWatermarkeFrameToPipe(const bool embedWatermark, const VideoProcessingContext& data, BufferType& inputFrame, GrayBuffer& watermarkedFrame, AVFrame* frame, FILE* ffmpegPipe);