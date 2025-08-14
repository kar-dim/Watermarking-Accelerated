#include "buffer.hpp"
#include "video_utils.hpp"
#include "videoprocessingcontext.hpp"
#include "WatermarkBase.hpp"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <format>
#include <iostream>
#include <string>

#if defined(_USE_CUDA_)
#include "cuda_utils.hpp"
#include "cuda_stream_manager.hpp"
extern "C" {
#include "libavutil/buffer.h"
#include "libavutil/pixfmt.h"
#include "libavutil/hwcontext.h"
#include <libavutil/hwcontext_cuda.h>
}
#endif

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include "libavcodec/codec_par.h"
#include "libavutil/frame.h"
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
#if defined(_USE_CUDA_)
	//try to open a CUDA hardware accelerated decoder, if the user specified one, if it fails , fallback to a software decoder
	AVCodecContext* openDecoderHWAccel(const AVCodecParameters* inputCodecParams, const std::string& userHwDecoder, bool& useHwDecoder)
	{
		if (userHwDecoder.empty())
			return openDecoder(inputCodecParams);
		const AVCodec* inputDecoder = avcodec_find_decoder_by_name(userHwDecoder.c_str());
		if (!inputDecoder)
			return openDecoder(inputCodecParams);
		AVCodecContextPtr ctx(avcodec_alloc_context3(inputDecoder), [](AVCodecContext* c) { avcodec_free_context(&c); });
		if (!ctx)
			return openDecoder(inputCodecParams);
		if (avcodec_parameters_to_context(ctx.get(), inputCodecParams) < 0)
			return openDecoder(inputCodecParams);
		AVBufferRef* raw_hw_device_ctx = nullptr;
		if (av_hwdevice_ctx_create(&raw_hw_device_ctx, AV_HWDEVICE_TYPE_CUDA, NULL, NULL, AV_CUDA_USE_CURRENT_CONTEXT) < 0)
			return openDecoder(inputCodecParams);
		AVBufferRefPtr hw_device_ctx(raw_hw_device_ctx, [](AVBufferRef* ref) { av_buffer_unref(&ref); });
		if (av_hwdevice_ctx_init(hw_device_ctx.get()) < 0)
			return openDecoder(inputCodecParams);
		ctx->hw_device_ctx = av_buffer_ref(hw_device_ctx.get());
		ctx->get_format = [](AVCodecContext* ctx, const enum AVPixelFormat* pix_fmts) { return AV_PIX_FMT_CUDA; };
		if (avcodec_open2(ctx.get(), inputDecoder, nullptr) < 0)
			return openDecoder(inputCodecParams);
		useHwDecoder = true;
		return ctx.release();
	}

	//embed watermark in a video frame by using CUDA hardware acceleration
	void embedWatermarkHWAccel(VideoProcessingContext& data, int& framesCount, const AVFrame* frame, FILE* ffmpegPipe)
	{
		const auto afStream = CudaStreamManager::getInstance().getAfStream();
		const auto videoStream = CudaStreamManager::getInstance().getCustomStream();
		const BufferType lumaBuffer(data.height, data.width, f32);
		const BufferType chromaBuffer(data.width, data.height / 2, u8);

		//launch NV12 to YUV420 kernel
		cuda_utils::launchNV12ToYUV420pKernel(frame->data[1], frame->linesize[1], chromaBuffer.device<uint8_t>(), data.width / 2, data.height / 2, afStream);
		chromaBuffer.unlock();

		if (framesCount % data.watermarkInterval == 0)
		{
			//try to overlap kernel and host copy
			cuda_utils::launchU8PitchedToFloatKernel(frame->data[0], lumaBuffer.device<float>(), data.width, data.height, frame->linesize[0], videoStream);
			chromaBuffer.host(data.hostFramePtr + (data.width * data.height));
			cudaStreamSynchronize(videoStream);
			float watermarkStrength;
			data.watermarkObj->makeWatermark(lumaBuffer, lumaBuffer, data.watermarkedFrame, watermarkStrength, ME);
			data.watermarkedFrame.as(u8).T().host(data.hostFramePtr);
			//write Y + UV packed
			cudaStreamSynchronize(afStream);
			fwrite(data.hostFramePtr, 1, data.width * data.height * 3 / 2, ffmpegPipe);
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
		const BufferType lumaBuffer(data.height, data.width, f32);
		if (framesCount % data.watermarkInterval == 0)
		{
			cuda_utils::launchU8PitchedToFloatKernel(frame->data[0], lumaBuffer.device<float>(), data.width, data.height, frame->linesize[0], afStream);
			lumaBuffer.unlock();
			float correlation = data.watermarkObj->detectWatermark(lumaBuffer, ME);
			std::cout << "Correlation for frame: " << (framesCount + 1) << ": " << correlation << "\n";
		}
		framesCount++;
	}
#endif

	//embed watermark in a video frame
	void embedWatermark(VideoProcessingContext& data, int& framesCount, const AVFrame* frame, FILE* ffmpegPipe)
	{
		const bool embedWatermark = framesCount % data.watermarkInterval == 0;
		const bool rowPadding = frame->linesize[0] != data.width;
		processAndWriteYPlane(embedWatermark, frame, data, ffmpegPipe);
		writeChromaPlanes(rowPadding, frame, data, ffmpegPipe);
		framesCount++;
	}

	//detect the watermark for a video frame
	void detectWatermark(VideoProcessingContext& data, int& framesCount, const AVFrame* frame)
	{
		//detect watermark after X frames
		if (framesCount % data.watermarkInterval == 0)
		{
			//if there is row padding (for alignment), we must copy the data to a contiguous block!
			const bool rowPadding = frame->linesize[0] != data.width;
			if (rowPadding)
			{
				for (int y = 0; y < data.height; y++)
					memcpy(data.hostFramePtr + y * data.width, frame->data[0] + y * frame->linesize[0], data.width);
			}
			//supply the input frame to the GPU and run the detection of the watermark
#if defined(_USE_GPU_)
			data.inputFrame = GrayBuffer(data.width, data.height, rowPadding ? data.hostFramePtr : frame->data[0], afHost).T().as(f32);
#else
			data.inputFrame = Map<GrayBuffer>(rowPadding ? data.hostFramePtr : frame->data[0], data.width, data.height).transpose().cast<float>();
#endif
			float correlation = data.watermarkObj->detectWatermark(data.inputFrame, ME);
			std::cout << "Correlation for frame: " << (framesCount + 1) << ": " << correlation << "\n";
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
		if (!inputDecoder)
			return nullptr;
		AVCodecContextPtr ctx(avcodec_alloc_context3(inputDecoder), [](AVCodecContext* ctx) { avcodec_free_context(&ctx); });
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
		return ctx.release();
	}

	//get the input video FPS (average)
	string getFrameRate(const AVFormatContext* inputFormatCtx, const int videoStreamIndex)
	{
		const AVRational frameRate = inputFormatCtx->streams[videoStreamIndex]->avg_frame_rate;
		return std::format("{:.3f}", static_cast<float>(frameRate.num) / frameRate.den);
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
			float watermarkStrength;
#if defined(_USE_GPU_)
			data.inputFrame = BufferType(data.width, data.height, srcY, afHost).T().as(f32);
			data.watermarkObj->makeWatermark(data.inputFrame, data.inputFrame, data.watermarkedFrame, watermarkStrength, ME);
			data.watermarkedFrame.as(u8).T().host(data.hostFramePtr);
			fwrite(data.hostFramePtr, 1, data.width * data.height, ffmpegPipe);
#elif defined(_USE_EIGEN_)
			data.inputFrame = Map<GrayBuffer>(srcY, data.width, data.height).transpose().cast<float>();
			data.watermarkObj->makeWatermark(data.inputFrame, data.inputFrame, data.watermarkedFrame, watermarkStrength, ME);
			data.grayFrame = data.watermarkedFrame.getGray().transpose().cast<uint8_t>();
			fwrite(data.grayFrame.data(), 1, data.width * data.height, ffmpegPipe);
#endif
		}
		else
		    fwrite(srcY, 1, data.width * data.height, ffmpegPipe);
	}

	//writes the chroma planes (U and V) to the ffmpeg pipe, either assuming aligned pointers or not
	void writeChromaPlanes(const bool rowPadding, const AVFrame* frame, VideoProcessingContext& data, FILE* ffmpegPipe)
	{
		if (rowPadding) 
		{
			for (int y = 0; y < data.height / 2; y++)
				fwrite(frame->data[1] + y * frame->linesize[1], 1, data.width / 2, ffmpegPipe);
			for (int y = 0; y < data.height / 2; y++)
				fwrite(frame->data[2] + y * frame->linesize[2], 1, data.width / 2, ffmpegPipe);
			return;
		}
		fwrite(frame->data[1], 1, data.width * data.height / 4, ffmpegPipe);
		fwrite(frame->data[2], 1, data.width * data.height / 4, ffmpegPipe);
	}
}