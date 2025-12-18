#pragma once
#include "buffer.hpp"
#include "video_defines.hpp"
#include "WatermarkBase.hpp"
#include <cstdint>

extern "C" {
#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
#include "libavfilter/avfilter.h"
}

using video_utils::AVFilterGraphPtr;

/*!
 *  \brief  Struct to hold Input Filter Graph and its source and sink contexts
 *          used only when needed (10-bit or HDR video embedding only)
 *  \author Dimitris Karatzas
 */
struct FilterGraphContext
{
    AVFilterGraphPtr filterGraph;
    AVFilterContext* buffersrcCtx;
    AVFilterContext* buffersinkCtx;

    FilterGraphContext() : filterGraph(nullptr), buffersrcCtx(nullptr), buffersinkCtx(nullptr)
    { }

    FilterGraphContext(AVFilterGraph* graph, AVFilterContext* srcCtx, AVFilterContext* sinkCtx)
        : filterGraph(graph), buffersrcCtx(srcCtx), buffersinkCtx(sinkCtx)
    { }
};

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
    const AVStream* videoStream;
    WatermarkBase* watermarkObj;
    const int height;
    const int width;
    const int watermarkInterval;
    FilterGraphContext filterGraphContext;
    //data host pointer and device/host buffers which are overwritten per frame
    uint8_t* hostFramePtr;
    ImageBuffer inputFrame;
    ImageBuffer watermarkedFrame;
    Gray8Buffer grayFrame;

    VideoProcessingContext(AVFormatContext* inputCtx, AVCodecContext* decoderCtx, const int streamIdx,
        const AVStream* videoStream, WatermarkBase* watermark, const int interval, uint8_t* inputFrame)
        : inputFormatCtx(inputCtx), inputDecoderCtx(decoderCtx), videoStreamIndex(streamIdx), videoStream(videoStream), watermarkObj(watermark), height(videoStream->codecpar->height), width(videoStream->codecpar->width),
        watermarkInterval(interval), hostFramePtr(inputFrame), inputFrame({ height, width }), watermarkedFrame({ height, width }), grayFrame({ height, width })
    { }
};

