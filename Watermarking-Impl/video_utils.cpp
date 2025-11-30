#include "buffer.hpp"
#include "utils.hpp"
#include "video_utils.hpp"
#include "VideoProcessingContext.hpp"
#include "WatermarkBase.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <format>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>

#if defined(_USE_CUDA_)
#include "cuda_utils.hpp"
#include "CudaStreamManager.hpp"
extern "C" {
#include "libavutil/buffer.h"
#include "libavutil/hwcontext.h"
#include <libavutil/hwcontext_cuda.h>
}
#endif

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include "libavcodec/codec_par.h"
#include "libavcodec/codec.h"
#include "libavcodec/packet.h"
#include "libavutil/frame.h"
#include "libavutil/avutil.h"
#include "libavfilter/avfilter.h"
#include "libavfilter/buffersrc.h"
#include "libavfilter/buffersink.h"
#include "libavutil/display.h"
#include "libavutil/rational.h"
#include <libavutil/pixdesc.h>
#include "libavutil/pixfmt.h"
#include "libavutil/mem.h"
}

#if defined(_USE_EIGEN_)
using namespace Eigen;
#endif

using std::string;
using std::cout;

namespace video_utils
{
#if defined(_USE_CUDA_)
	//try to open a CUDA hardware accelerated decoder, if the user specified one, if it fails , fallback to a software decoder
	AVCodecContextPtr openDecoderHWAccel(const AVCodecParameters* inputCodecParams, const string& userHwDecoder, bool& useHwDecoder)
	{
		if (userHwDecoder.empty())
			return openSoftwareDecoder(inputCodecParams);
		const AVCodec* inputDecoder = avcodec_find_decoder_by_name(userHwDecoder.c_str());
		if (!inputDecoder)
			return openSoftwareDecoder(inputCodecParams);
		AVCodecContextPtr ctx(avcodec_alloc_context3(inputDecoder));
		if (!ctx)
			return openSoftwareDecoder(inputCodecParams);
		if (avcodec_parameters_to_context(ctx.get(), inputCodecParams) < 0)
			return openSoftwareDecoder(inputCodecParams);
		AVBufferRef* raw_hw_device_ctx = nullptr;
		if (av_hwdevice_ctx_create(&raw_hw_device_ctx, AV_HWDEVICE_TYPE_CUDA, NULL, NULL, AV_CUDA_USE_CURRENT_CONTEXT) < 0)
			return openSoftwareDecoder(inputCodecParams);
		AVBufferRefPtr hw_device_ctx(raw_hw_device_ctx);
		if (av_hwdevice_ctx_init(hw_device_ctx.get()) < 0)
			return openSoftwareDecoder(inputCodecParams);
		ctx->hw_device_ctx = av_buffer_ref(hw_device_ctx.get());
		ctx->get_format = [](AVCodecContext* ctx, const enum AVPixelFormat* pix_fmts) { return AV_PIX_FMT_CUDA; };
		if (avcodec_open2(ctx.get(), inputDecoder, nullptr) < 0)
			return openSoftwareDecoder(inputCodecParams);
		useHwDecoder = true;
		checkPixelFormatSupport(supportedHwFormats, ctx->sw_pix_fmt);
		return ctx;
	}

	//embed watermark in a video frame by using CUDA hardware acceleration
	void embedWatermarkHWAccel(VideoProcessingContext& data, int& framesCount, const AVFrame* frame, FILE* ffmpegPipe)
	{
		const auto afStream = CudaStreamManager::getInstance().getAfStream();
		const auto videoStream = CudaStreamManager::getInstance().getCustomStream();
		const ImageBuffer lumaBuffer(data.height, data.width, f32);
		const ImageBuffer lumaBufferPacked(data.width, data.height, u8); //used only when we don't embed
		const ImageBuffer chromaBuffer(data.width, data.height / 2, u8);
		//launch NV12 to YUV420 kernel (for UV planes)
		cuda_utils::launchNV12ToYUV420pKernel(frame->data[1], frame->linesize[1], chromaBuffer.device<uint8_t>(), data.width / 2, data.height / 2, afStream);
		chromaBuffer.unlock();
		
		if (framesCount % data.watermarkInterval == 0)
		{
			//overlap kernel and host copy
			cuda_utils::launchPitchedToFloatKernel(frame->data[0], lumaBuffer.device<float>(), data.width, data.height, frame->linesize[0], videoStream);
			chromaBuffer.host(data.hostFramePtr + (data.width * data.height));
			cudaStreamSynchronize(videoStream);
			//write Y + UV packed
			embedAndWriteFrame(data, lumaBuffer, data.width * data.height * 3 / 2, ffmpegPipe);
		}
		else
		{
			//try to overlap the two D2H copies
			cudaMemcpy2DAsync(data.hostFramePtr, data.width, frame->data[0], frame->linesize[0], data.width, data.height, cudaMemcpyDeviceToHost, videoStream);
			chromaBuffer.host(data.hostFramePtr + (data.width * data.height));
			cudaStreamSynchronize(videoStream);
			//write Y + UV packed
			fwrite(data.hostFramePtr, 1, data.width * data.height * 3 / 2, ffmpegPipe);
		}
		framesCount++;
	}

	//detect a watermark in a video frame using hardware acceleration
	//directly use the GPU memory from the cuda decoder, no need to copy the data to host and back to GPU
	void detectWatermarkHWAccel(VideoProcessingContext& data, int& framesCount, const AVFrame* frame)
	{
		const auto afStream = CudaStreamManager::getInstance().getAfStream();
		const ImageBuffer lumaBuffer(data.height, data.width, f32);
		//detect watermark after watermarkInterval frames
		if (framesCount % data.watermarkInterval == 0)
		{
			cuda_utils::launchPitchedToFloatKernel(frame->data[0], lumaBuffer.device<float>(), data.width, data.height, frame->linesize[0], afStream);
			lumaBuffer.unlock();
			float correlation = data.watermarkObj->detectWatermark(lumaBuffer, ME);
			cout << "Correlation for frame: " << (framesCount + 1) << ": " << correlation << "\n";
		}
		framesCount++;
	}
#endif

	//get HDR info from the codec context in order to pass it to the filter graph for correct SDR tonemapping
	//if 10-bit HDR video -> tonemap to SDR with ffmpeg CPU filters and convert to 8-bit
	//if 10-bit SDR video -> convert to 8-bit fast
    string getFilterGraphString(const AVCodecContext* codecCtx, const AVStream *st, const bool useHwDecoder)
	{
		if (!is10bit(codecCtx, st))
			return "";  //8-bit SDR, no filtering (save processing time)
		if (!isHDR(codecCtx))
			return useHwDecoder ? "scale_cuda=format=nv12" : "format=yuv420p";  //10-bit SDR, fast downscale to 8-bit
		//HDR10 / 10-bit HDR GPU case -> unfortunately no way to tonemap in GPU with cuda filters yet! Should use CPU decoder instead 
		if (useHwDecoder)
			throw std::runtime_error("Cannot tonemap HDR input to SDR with Hardware Accelerated Decoder yet. Use CPU decoder instead.");
		//HDR10 / 10-bit HDR CPU case -> scaler needs more input info
		const char* primaries = av_color_primaries_name(codecCtx->color_primaries);
		const char* matrix = av_color_space_name(codecCtx->colorspace);
		//fallback to safe HDR10 defaults if any field is unspecified
		if (!primaries || codecCtx->color_primaries == AVCOL_PRI_UNSPECIFIED)
			primaries = "bt2020";
		if (!matrix || codecCtx->colorspace == AVCOL_SPC_UNSPECIFIED)
			matrix = "bt2020nc";
		return std::format("zscale=primaries={}:transfer=linear:matrix={}:npl=100,tonemap=mobius,zscale=transfer=bt709:primaries=bt709:matrix=bt709,format=yuv420p", primaries, matrix);
	}

	//filter a single frame
	void filterFrame(AVFramePtr& frame, const FilterGraphContext& filterGraphContext)
	{
		AVFrame* filteredFrame = av_frame_alloc();
		if (av_buffersrc_add_frame_flags(filterGraphContext.buffersrcCtx, frame.get(), AV_BUFFERSRC_FLAG_KEEP_REF) < 0)
		{
			av_frame_free(&filteredFrame);
			throw std::runtime_error("Failed to add frame to filter graph");
		}
		int ret = av_buffersink_get_frame(filterGraphContext.buffersinkCtx, filteredFrame);
		if (ret < 0)
		{
			av_frame_free(&filteredFrame);
			throw std::runtime_error("Failed to get filtered frame");
		}
		frame.reset(filteredFrame);

	}

	//initialize the filter graph for 10-bit to 8-bit conversion and HDR to SDR tonemapping
	bool initFilterGraph(const AVCodecContext* inputDecoderCtx, const AVStream* st, const bool useHwDecoder, FilterGraphContext& filterCtx)
	{
		const string filterDesc = getFilterGraphString(inputDecoderCtx, st, useHwDecoder);
		if (filterDesc.empty())
			return false;

		char pixFmtName[32];
		snprintf(pixFmtName, sizeof(pixFmtName), "%s", av_get_pix_fmt_name((AVPixelFormat)inputDecoderCtx->pix_fmt));
		const AVRational timeBase = getTimeBase(st);
		char args[512];
		snprintf(args, sizeof(args), "video_size=%dx%d:pix_fmt=%s:time_base=%d/%d:pixel_aspect=%d/%d",
			inputDecoderCtx->width, inputDecoderCtx->height, pixFmtName, timeBase.num, timeBase.den,
			inputDecoderCtx->sample_aspect_ratio.num, inputDecoderCtx->sample_aspect_ratio.den);

		filterCtx.filterGraph = avfilter_graph_alloc();
		if (!filterCtx.filterGraph)
			return false;

		const AVFilter* buffersrc = avfilter_get_by_name("buffer");
		const AVFilter* buffersink = avfilter_get_by_name("buffersink");
		if (!buffersrc || !buffersink)
			return false;

		if (avfilter_graph_create_filter(&filterCtx.buffersrcCtx, buffersrc, "in", args, nullptr, filterCtx.filterGraph) < 0)
			return false;

		if (avfilter_graph_create_filter(&filterCtx.buffersinkCtx, buffersink, "out", nullptr, nullptr, filterCtx.filterGraph) < 0)
			return false;

		AVFilterInOut* outputs = avfilter_inout_alloc();
		AVFilterInOut* inputs = avfilter_inout_alloc();

		outputs->name = av_strdup("in");
		outputs->filter_ctx = filterCtx.buffersrcCtx;
		outputs->pad_idx = 0;
		outputs->next = nullptr;

		inputs->name = av_strdup("out");
		inputs->filter_ctx = filterCtx.buffersinkCtx;
		inputs->pad_idx = 0;
		inputs->next = nullptr;

		if (avfilter_graph_parse_ptr(filterCtx.filterGraph, filterDesc.c_str(), &inputs, &outputs, nullptr) < 0)
			return false;

		if (avfilter_graph_config(filterCtx.filterGraph, nullptr) < 0)
			return false;

		avfilter_inout_free(&inputs);
		avfilter_inout_free(&outputs);

		return true;
	}

	//helper method to embed the watermark in the video frame and write it to the ffmpeg pipe
	void embedAndWriteFrame(VideoProcessingContext& data, const ImageBuffer& buffer, const int elements, FILE* ffmpegPipe)
	{
		float watermarkStrength;
		data.watermarkObj->makeWatermark(buffer, buffer, data.watermarkedFrame, watermarkStrength, ME);
#if defined(_USE_GPU_)
		data.watermarkedFrame.as(u8).T().host(data.hostFramePtr);
		fwrite(data.hostFramePtr, 1, elements, ffmpegPipe);
#elif defined(_USE_EIGEN_)
		data.grayFrame = data.watermarkedFrame.getGray().transpose().cast<uint8_t>();
		fwrite(data.grayFrame.data(), 1, elements, ffmpegPipe);
#endif
	}

	//helper method to dispatch the correct watermarking or detection method
	int videoDispatcher(VideoProcessingContext& data, const bool useHwDecoder, const VideoOp op, const bool needsFilter, FILE* ffmpegPipe)
	{
#if defined(_USE_CUDA_)
		if (useHwDecoder)
			return needsFilter ? processFrames<true>(data, [&](const AVFrame* frame, int& framesCount) { op == EMBED ? embedWatermarkHWAccel(data, framesCount, frame, ffmpegPipe) : detectWatermarkHWAccel(data, framesCount, frame); })
			                   : processFrames<false>(data, [&](const AVFrame* frame, int& framesCount) { op == EMBED ? embedWatermarkHWAccel(data, framesCount, frame, ffmpegPipe) : detectWatermarkHWAccel(data, framesCount, frame); });
#endif
		return needsFilter ? processFrames<true>(data, [&](const AVFrame* frame, int& framesCount) { op == EMBED ? embedWatermark(data, framesCount, frame, ffmpegPipe) : detectWatermark(data, framesCount, frame); })
			               : processFrames<false>(data, [&](const AVFrame* frame, int& framesCount) { op == EMBED ? embedWatermark(data, framesCount, frame, ffmpegPipe) : detectWatermark(data, framesCount, frame); });
	}

	//embed watermark in a video frame
	void embedWatermark(VideoProcessingContext& data, int& framesCount, const AVFrame* frame, FILE* ffmpegPipe)
	{
		const bool embedWatermark = framesCount % data.watermarkInterval == 0;
		processAndWriteYPlane(embedWatermark, frame, data, ffmpegPipe);
		writeChromaPlanes(frame, data, ffmpegPipe);
		framesCount++;
	}

	//detect the watermark for a video frame
	//NOTE: supports 10-bit decoding. Experimental because we don't encode 10-bit yet
	void detectWatermark(VideoProcessingContext& data, int& framesCount, const AVFrame* frame)
	{
		//detect watermark after watermarkInterval frames
		if (framesCount % data.watermarkInterval == 0)
		{
			uint8_t* srcY = frame->data[0];
			if (frame->linesize[0] != data.width)
			{
				for (int y = 0; y < data.height; y++)
					memcpy(data.hostFramePtr + y * data.width, frame->data[0] + y * frame->linesize[0], data.width);
				srcY = data.hostFramePtr;
			}
			loadInputFrame<Gray8Buffer>(data, srcY);
			float correlation = data.watermarkObj->detectWatermark(data.inputFrame, ME);
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

	//get the rotation angle of the video stream (if any) to apply as a filter in the encoder
	string getStreamRotation(const AVStream* st)
	{
		if (!st || !st->codecpar)
			return "";
		for (int i = 0; i < st->codecpar->nb_coded_side_data; i++)
		{
			const AVPacketSideData& sd = st->codecpar->coded_side_data[i];
			if (sd.type == AV_PKT_DATA_DISPLAYMATRIX && sd.data && sd.size >= 9 * sizeof(int32_t)) 
			{
				const double rotation = -av_display_rotation_get(reinterpret_cast<const int32_t*>(sd.data));
				const int angle = (static_cast<int>(std::lrint(rotation)) + 360) % 360; //normalize to [0, 360)
				switch (angle)
				{
					case 90:  return "-vf \"transpose=1\" ";
					case 180: return "-vf \"hflip,vflip\" ";
					case 270: return "-vf \"transpose=2\" ";
					default:  return "";
				}
			}
		}
		return "";
	}

	//if CUDA, try to open hw decoder (if requested), else fallback to open software decoder context for video
	//otherwise, just open software decoder
	AVCodecContextPtr openDecoder(const AVCodecParameters* inputCodecParams, const string& userHwDecoder, bool& useHwDecoder)
	{
#if defined(_USE_CUDA_)
		return openDecoderHWAccel(inputCodecParams, userHwDecoder, useHwDecoder);
#else
		return openSoftwareDecoder(inputCodecParams);
#endif
	}

	//open software decoder context for video
	AVCodecContextPtr openSoftwareDecoder(const AVCodecParameters* inputCodecParams)
	{
		const AVCodec* inputDecoder = avcodec_find_decoder(inputCodecParams->codec_id);
		if (!inputDecoder)
			return nullptr;
		AVCodecContextPtr ctx(avcodec_alloc_context3(inputDecoder));
		if (!ctx)
			return nullptr;
		if (avcodec_parameters_to_context(ctx.get(), inputCodecParams) < 0)
			return nullptr;
		//multithreading decode
		ctx->thread_count = 0;
		if (inputDecoder->capabilities & AV_CODEC_CAP_FRAME_THREADS)
			ctx->thread_type = FF_THREAD_FRAME;
		else if (inputDecoder->capabilities & AV_CODEC_CAP_SLICE_THREADS)
			ctx->thread_type = FF_THREAD_SLICE;
		else
			ctx->thread_count = 1; //don't use multithreading
		if (avcodec_open2(ctx.get(), inputDecoder, nullptr) < 0)
			return nullptr;
		checkPixelFormatSupport(supportedFormats, ctx->pix_fmt);
		return ctx;
	}

	//check if the pixel format provided is in the list of provided supported formats
	bool checkPixelFormatSupport(const std::span<const AVPixelFormat> supportedFormats, const AVPixelFormat format)
	{
		const bool isValidFormat = std::ranges::any_of(supportedFormats, [&](auto f) { return f == format; });
		Utils::checkError(!isValidFormat, "Error: Video frame format not supported, aborting");
		return isValidFormat;
	}

	//get the input video FPS (average)
	string getFrameRate(const AVStream* st)
	{
		const AVRational frameRate = st->avg_frame_rate;
		return std::format("{:.3f}", static_cast<float>(frameRate.num) / frameRate.den);
	}

	//get the input video time base (should be 1/average(fps))
	AVRational getTimeBase(const AVStream* st)
	{
		AVRational fps = st->avg_frame_rate;
		AVRational tb;
		tb.num = fps.den;
		tb.den = fps.num; //time_base = 1/fps
		return tb;
	}

	//runs the watermark creation for a video frame and writes the watermarked frame to the ffmpeg pipe
	void processAndWriteYPlane(const bool embedWatermark, const AVFrame* frame, VideoProcessingContext& data, FILE* ffmpegPipe)
	{
		uint8_t* srcY = frame->data[0];
		//if there is row padding (for alignment), we must copy the data to a contiguous block!
		if (frame->linesize[0] != data.width)
		{
			for (int y = 0; y < data.height; y++)
				memcpy(data.hostFramePtr + y * data.width, srcY + y * frame->linesize[0], data.width);
			srcY = data.hostFramePtr;
		}
		if (embedWatermark)
		{
			loadInputFrame<Gray8Buffer>(data, srcY);
			embedAndWriteFrame(data, data.inputFrame, data.width * data.height, ffmpegPipe);
		}
		else
			fwrite(srcY, 1, data.width * data.height, ffmpegPipe);
	}

	//writes the chroma planes (U and V) to the ffmpeg pipe, either assuming aligned pointers or not
	void writeChromaPlanes(const AVFrame* frame, VideoProcessingContext& data, FILE* ffmpegPipe)
	{
		//lambda to write a single chroma plane
		auto writePlane = [&](const uint8_t* src, const int linesize)
		{
			if (linesize != data.width / 2)
			{
				for (int y = 0; y < data.height / 2; y++)
					fwrite(src + y * linesize, 1, data.width / 2, ffmpegPipe);
			}
			else
				fwrite(src, 1, data.width * data.height / 4, ffmpegPipe);
		};

		//write U
		writePlane(frame->data[1], frame->linesize[1]);
		//write V
		writePlane(frame->data[2], frame->linesize[2]);
	}
}