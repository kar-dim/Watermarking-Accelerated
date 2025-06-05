#pragma once
#include "videoprocessingcontext.hpp"
#include <cuda_runtime.h>
#include <functional>
#include <INIReader.h>
#include <cstdio>
#include <string>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavcodec/packet.h>
#include <libavutil/frame.h>
#include <libavcodec/codec_par.h>
}

/*!
 *  \brief  Helper methods for testing the watermark algorithms
 *  \author Dimitris Karatzas
 */
void exitProgram(const int exitCode);
std::string executionTime(const bool showFps, const double seconds);
int testForImage(const INIReader& inir, const cudaDeviceProp& properties, const int p, const float psnr);
int testForVideo(const INIReader& inir, const std::string& videoFile, const cudaDeviceProp& properties, const int p, const float psnr);
int findVideoStreamIndex(const AVFormatContext* inputFormatCtx);
AVCodecContext* openDecoderContext(const AVCodecParameters* params);
bool receivedValidVideoFrame(AVCodecContext* inputDecoderCtx, AVPacket* packet, AVFrame* frame, const int videoStreamIndex);
std::string getVideoFrameRate(const AVFormatContext* inputFormatCtx, const int videoStreamIndex);
void embedWatermarkFrame(const VideoProcessingContext& data, int& framesCount, AVFrame* frame, FILE* ffmpegPipe);
void detectFrameWatermark(const VideoProcessingContext& data, int& framesCount, AVFrame* frame);
int processFrames(const VideoProcessingContext& data, std::function<void(AVFrame*, int&)> processFrame);