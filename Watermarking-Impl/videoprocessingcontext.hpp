#pragma once
#include "buffer.hpp"
#include "WatermarkBase.hpp"
#include <cstdint>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
}

/*!
 *  \brief  Struct to hold common data for video watermarking and detection
 *  \author Dimitris Karatzas
 */
struct VideoProcessingContext 
{
    //common libav contexts and constants
    AVFormatContext* inputFormatCtx;
    AVCodecContext* inputDecoderCtx;
    const int videoStreamIndex;
    WatermarkBase* watermarkObj;
    const int height;
    const int width;
    const int watermarkInterval;
    //data host pointer and device/host buffers which are overwritten per frame
    uint8_t* hostFramePtr;
    ImageBuffer inputFrame;
    ImageBuffer watermarkedFrame;
    Gray8Buffer grayFrame;

    VideoProcessingContext(AVFormatContext* inputCtx, AVCodecContext* decoderCtx, const int streamIdx,
        WatermarkBase* watermark, const int h, const int w, const int interval, uint8_t* inputFrame)
        : inputFormatCtx(inputCtx), inputDecoderCtx(decoderCtx), videoStreamIndex(streamIdx), watermarkObj(watermark),
		height(h), width(w), watermarkInterval(interval), hostFramePtr(inputFrame), inputFrame({ h, w }), watermarkedFrame({ h, w }), grayFrame({ h, w })
    { }
};