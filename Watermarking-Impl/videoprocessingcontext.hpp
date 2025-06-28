#pragma once
#include "WatermarkBase.hpp"
#include <cstdint>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
}

/*!
 *  \brief  Struct to hold common data for video watermarking and detection. Hholds pointers and references, does not own any resources.
 *  \author Dimitris Karatzas
 */
struct VideoProcessingContext 
{
    AVFormatContext* inputFormatCtx;
    AVCodecContext* inputDecoderCtx;
    const int videoStreamIndex;
    WatermarkBase* watermarkObj;
    const int height;
    const int width;
    const int watermarkInterval;
    uint8_t* inputFramePtr;

    VideoProcessingContext(AVFormatContext* inputCtx, AVCodecContext* decoderCtx, const int streamIdx,
        WatermarkBase* watermark, const int h, const int w, const int interval, uint8_t* inputFrame)
        : inputFormatCtx(inputCtx), inputDecoderCtx(decoderCtx), videoStreamIndex(streamIdx), watermarkObj(watermark),
        height(h), width(w), watermarkInterval(interval), inputFramePtr(inputFrame)
    { }
};