#include "buffer.hpp"
#include "video_utils.hpp"
#include "videoprocessingcontext.hpp"
#include "WatermarkBase.hpp"
#include <cstdio>
#include <cstring>
#include <format>
#include <iostream>
#include <string>

#if defined(_USE_CUDA_)
#include <arrayfire.h>
#include "cuda_utils.hpp"
#include "WatermarkGpu.hpp" //DEBUG REMOVE IT LATER
#include <cuda.h>
#include <cstdint>
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
		AVCodecContext* inputDecoderCtx = avcodec_alloc_context3(inputDecoder);
		if (!inputDecoderCtx)
			return openDecoder(inputCodecParams);
		if (avcodec_parameters_to_context(inputDecoderCtx, inputCodecParams) < 0)
		{
			avcodec_free_context(&inputDecoderCtx);
			return openDecoder(inputCodecParams);
		}
		AVBufferRef* hw_device_ctx = nullptr;
		if (av_hwdevice_ctx_create(&hw_device_ctx, AV_HWDEVICE_TYPE_CUDA, NULL, NULL, AV_CUDA_USE_CURRENT_CONTEXT) < 0)
		{
			avcodec_free_context(&inputDecoderCtx);
			return openDecoder(inputCodecParams);
		}
		if (av_hwdevice_ctx_init(hw_device_ctx) < 0)
		{
			av_buffer_unref(&hw_device_ctx);
			avcodec_free_context(&inputDecoderCtx);
			return openDecoder(inputCodecParams);
		}
		inputDecoderCtx->hw_device_ctx = av_buffer_ref(hw_device_ctx);
		av_buffer_unref(&hw_device_ctx);
		inputDecoderCtx->get_format = [](AVCodecContext* ctx, const enum AVPixelFormat* pix_fmts) { return AV_PIX_FMT_CUDA; };
		if (avcodec_open2(inputDecoderCtx, inputDecoder, nullptr) < 0)
		{
			avcodec_free_context(&inputDecoderCtx);
			return openDecoder(inputCodecParams);
		}
		useHwDecoder = true;
		return inputDecoderCtx;
	}

	//embed watermark in a video frame by using CUDA hardware acceleration
	void embedWatermarkHWAccel(VideoProcessingContext& data, int& framesCount, const AVFrame* frame, FILE* ffmpegPipe)
	{
		//TODO MAY FIX PERFORMANCE HMM
		const af::array frameT(data.width, data.height, u8);
		const af::array gpuOutputUV(data.width, data.height / 2, u8);
		const uint8_t* gpuFramePtr = reinterpret_cast<const uint8_t*>(reinterpret_cast<CUdeviceptr>(frame->data[0]));
		const uint8_t* gpuUV = reinterpret_cast<const uint8_t*>(reinterpret_cast<CUdeviceptr>(frame->data[1]));
		//TODO: if it works -> try to OVERLAP kernels/memory copies etc
		const int uvPitch = frame->linesize[1];
		cuda_utils::launchNV12ToYUV420pKernel(gpuUV, uvPitch, gpuOutputUV.device<uint8_t>(), data.width / 2, data.height / 2);
		gpuOutputUV.unlock();
		gpuOutputUV.host(data.hostFramePtr + (data.width * data.height));
		const bool embedWatermark = framesCount % data.watermarkInterval == 0;
		//if there is row padding (for alignment), we must copy the data to a contiguous block!
		if (frame->linesize[0] != data.width)
		{
			cudaMemcpy2D(frameT.device<void>(), data.width * sizeof(uint8_t), gpuFramePtr, frame->linesize[0], data.width * sizeof(uint8_t), data.height, cudaMemcpyDeviceToDevice);
			frameT.unlock();
			//WatermarkGPU::displayArray(data.inputFrame.as(u8));
			if (embedWatermark)
			{
				float watermarkStrength;
				data.inputFrame = frameT.T().as(f32);
				data.watermarkObj->makeWatermark(data.inputFrame, data.inputFrame, data.watermarkedFrame, watermarkStrength, ME);
				data.watermarkedFrame.as(u8).T().host(data.hostFramePtr);
				//write Y
				fwrite(data.hostFramePtr, 1, data.width * data.height, ffmpegPipe);
			}
			else
			{
				frameT.host(data.hostFramePtr);
				//write Y
				fwrite(data.hostFramePtr, 1, data.width * data.height, ffmpegPipe);
			}
			//write UV packed
			fwrite(data.hostFramePtr + (data.width * data.height), 1, gpuOutputUV.elements(), ffmpegPipe);
		}

		//no row padding, read and write data directly
		//TODO, FIX THIS LATER, UNTESTED (easier though)
		else
		{
			if (embedWatermark)
			{
				float watermarkStrength;
				data.inputFrame = BufferType(data.width, data.height, gpuFramePtr, afDevice).T().as(f32);
				data.watermarkObj->makeWatermark(data.inputFrame, data.inputFrame, data.watermarkedFrame, watermarkStrength, ME);
				data.watermarkedFrame.as(u8).T().host(data.hostFramePtr);
				//write Y
				fwrite(data.hostFramePtr, 1, data.width * data.height, ffmpegPipe);
			}
			//write UV packed
			fwrite(data.hostFramePtr + (data.width * data.height), 1, gpuOutputUV.elements(), ffmpegPipe);
		}
		framesCount++;
	}

	//detect a watermark in a video frame using hardware acceleration
	//directly use the GPU memory from the cuda decoder, no need to copy the data to host and back to GPU
	void detectWatermarkHWAccel(VideoProcessingContext& data, int& framesCount, const AVFrame* frame)
	{
		if (framesCount % data.watermarkInterval == 0)
		{
			const uint8_t* gpuFramePtr = reinterpret_cast<const uint8_t*>(reinterpret_cast<CUdeviceptr>(frame->data[0]));
			const int pitch = frame->linesize[0];
			//if there is row padding (for alignment), we must copy the data to a contiguous block!
			if (pitch != data.width)
			{
				af::array frameT = af::array(data.width, data.height, u8);
				cudaMemcpy2D(frameT.device<void>(), data.width * sizeof(uint8_t), gpuFramePtr, pitch, data.width * sizeof(uint8_t), data.height, cudaMemcpyDeviceToDevice);
				frameT.unlock();
				data.inputFrame = frameT.T().as(f32);
			}
			//read from GPU memory directly (needs transposing, since arrayfire uses column-major order)
			else
				data.inputFrame = af::array(data.width, data.height, gpuFramePtr, afDevice).T().as(f32);
			//WatermarkGPU::displayArray(data.inputFrame.as(u8));

			float correlation = data.watermarkObj->detectWatermark(data.inputFrame, ME);
			std::cout << "Correlation for frame: " << (framesCount + 1) << ": " << correlation << "\n";
			framesCount++;
			return;
		}
	}
#endif

	//embed watermark in a video frame
	void embedWatermark(VideoProcessingContext& data, int& framesCount, const AVFrame* frame, FILE* ffmpegPipe)
	{
		const bool embedWatermark = framesCount % data.watermarkInterval == 0;
		//if there is row padding (for alignment), we must copy the data to a contiguous block!
		if (frame->linesize[0] != data.width)
		{
			if (embedWatermark)
			{
				for (int y = 0; y < data.height; y++)
					memcpy(data.hostFramePtr + y * data.width, frame->data[0] + y * frame->linesize[0], data.width);
				//embed the watermark, receive the watermarked data back to host and write the watermarked image data to ffmpeg pipe
				writeWatermarkeFrame(data, frame, ffmpegPipe);
			}
			else
			{
				//write from frame buffer row-by-row the the valid image data (and not the alignment bytes)
				for (int y = 0; y < data.height; y++)
					fwrite(frame->data[0] + y * frame->linesize[0], 1, data.width, ffmpegPipe);
			}
			//always write UV planes as-is
			for (int y = 0; y < data.height / 2; y++)
				fwrite(frame->data[1] + y * frame->linesize[1], 1, data.width / 2, ffmpegPipe);
			for (int y = 0; y < data.height / 2; y++)
				fwrite(frame->data[2] + y * frame->linesize[2], 1, data.width / 2, ffmpegPipe);
		}
		//no row padding, read and write data directly
		else
		{
			writeConditionalWatermarkFrame(embedWatermark, data, frame, ffmpegPipe);
			fwrite(frame->data[1], 1, data.width * data.height / 4, ffmpegPipe);
			fwrite(frame->data[2], 1, data.width * data.height / 4, ffmpegPipe);
		}
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
#if defined(_USE_CUDA_)
			data.inputFrame = GrayBuffer(data.width, data.height, rowPadding ? data.hostFramePtr : frame->data[0], afHost).T().as(f32);
#elif defined(_USE_OPENCL_)
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
	void writeWatermarkeFrame(VideoProcessingContext& data, const AVFrame* frame, FILE* ffmpegPipe)
	{
		float watermarkStrength;
#if defined(_USE_GPU_)
		data.inputFrame = BufferType(data.width, data.height, data.hostFramePtr, afHost).T().as(f32);
		data.watermarkObj->makeWatermark(data.inputFrame, data.inputFrame, data.watermarkedFrame, watermarkStrength, ME);
		data.watermarkedFrame.as(u8).T().host(data.hostFramePtr);
		fwrite(data.hostFramePtr, 1, data.width * data.height, ffmpegPipe);
#elif defined(_USE_EIGEN_)
		data.inputFrame = Map<GrayBuffer>(data.hostFramePtr, data.width, data.height).transpose().cast<float>();
		data.watermarkObj->makeWatermark(data.inputFrame, data.inputFrame, data.watermarkedFrame, watermarkStrength, ME);
		data.grayFrame = data.watermarkedFrame.getGray().transpose().cast<uint8_t>();
		fwrite(data.grayFrame.data(), 1, data.width * data.height, ffmpegPipe);
#endif
	}

	//runs the watermark creation for a video frame and writes the watermarked frame to the ffmpeg pipe, if the watermark is embedded, or writes the original frame data otherwise
	void writeConditionalWatermarkFrame(const bool embedWatermark, VideoProcessingContext& data, const AVFrame* frame, FILE* ffmpegPipe)
	{
		if (embedWatermark)
		{
			float watermarkStrength;
#if defined(_USE_GPU_)
			data.inputFrame = BufferType(data.width, data.height, frame->data[0], afHost).T().as(f32);
			data.watermarkObj->makeWatermark(data.inputFrame, data.inputFrame, data.watermarkedFrame, watermarkStrength, ME);
			data.watermarkedFrame.as(u8).T().host(data.hostFramePtr);
		}
		fwrite(embedWatermark ? data.hostFramePtr : frame->data[0], 1, data.width * data.height, ffmpegPipe);
#elif defined(_USE_EIGEN_)
			data.inputFrame = BufferType(Map<GrayBuffer>(frame->data[0], data.width, data.height).transpose().cast<float>());
			data.watermarkObj->makeWatermark(data.inputFrame, data.inputFrame, data.watermarkedFrame, watermarkStrength, ME);
			data.grayFrame = data.watermarkedFrame.getGray().transpose().cast<uint8_t>();
	}
		fwrite(embedWatermark ? data.grayFrame.data() : frame->data[0], 1, data.width * data.height, ffmpegPipe);
#endif
	}
}