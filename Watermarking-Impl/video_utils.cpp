#include "buffer.hpp"
#include "utils.hpp"
#include "video_utils.hpp"
#include "videoprocessingcontext.hpp"
#include "WatermarkBase.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <format>
#include <iostream>
#include <span>
#include <string>

#if defined(_USE_CUDA_)
#include "cuda_utils.hpp"
#include "cuda_stream_manager.hpp"
#include <cuda_runtime.h>
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
#include "libavcodec/packet.h"
#include "libavutil/frame.h"
#include "libavutil/avutil.h"
#include "libavcodec/codec.h"
#include "libavutil/display.h"
#include "libavutil/rational.h"
#include "libavutil/pixfmt.h"
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
		const int bitDepth = ((AVHWFramesContext*)(frame->hw_frames_ctx->data))->sw_format == AV_PIX_FMT_NV12 ? 8 : 10;
		const BufferType lumaBuffer(data.height, data.width, f32);
		const BufferType lumaBufferPacked(data.width, data.height, u8); //used only when we don't embed
		const BufferType chromaBuffer(data.width, data.height / 2, u8);
		//launch NV12 to YUV420 kernel (for UV planes)
		cuda_utils::launchNV12ToYUV420pKernel(frame->data[1], frame->linesize[1], chromaBuffer.device<uint8_t>(), data.width / 2, data.height / 2, bitDepth, afStream);
		chromaBuffer.unlock();

		if (framesCount % data.watermarkInterval == 0)
		{
			//overlap kernel and host copy
			cuda_utils::launchPitchedToFloatKernel(frame->data[0], lumaBuffer.device<float>(), data.width, data.height, frame->linesize[0], bitDepth, videoStream);
			chromaBuffer.host(data.hostFramePtr + (data.width * data.height));
			cudaStreamSynchronize(videoStream);
			//write Y + UV packed
			embedAndWriteFrame(data, lumaBuffer, data.width * data.height * 3 / 2, ffmpegPipe);
		}
		else
		{
			if (bitDepth == 10)
			{
				//overlap kernel and host copy
				cuda_utils::launchPitched10BitTo8BitKernel(reinterpret_cast<const uint16_t*>(frame->data[0]), lumaBufferPacked.device<uint8_t>(), data.width, data.height, frame->linesize[0], videoStream);
				chromaBuffer.host(data.hostFramePtr + (data.width * data.height));
				cudaStreamSynchronize(videoStream);
				lumaBufferPacked.unlock();
				lumaBufferPacked.host(data.hostFramePtr);
			}
			else 
			{
				//try to overlap the two D2H copies
				cudaMemcpy2DAsync(data.hostFramePtr, data.width, frame->data[0], frame->linesize[0], data.width, data.height, cudaMemcpyDeviceToHost, videoStream);
				chromaBuffer.host(data.hostFramePtr + (data.width * data.height));
				cudaStreamSynchronize(videoStream);
			}
			//write Y + UV packed
			fwrite(data.hostFramePtr, 1, data.width * data.height * 3 / 2, ffmpegPipe);
		}
		framesCount++;
	}

	//detect a watermark in a video frame using hardware acceleration
	//directly use the GPU memory from the cuda decoder, no need to copy the data to host and back to GPU
	//NOTE: supports 10-bit decoding. Experimental because we don't encode 10-bit yet
	void detectWatermarkHWAccel(VideoProcessingContext& data, int& framesCount, const AVFrame* frame)
	{
		const auto afStream = CudaStreamManager::getInstance().getAfStream();
		const int bitDepth = ((AVHWFramesContext*)(frame->hw_frames_ctx->data))->sw_format == AV_PIX_FMT_NV12 ? 8 : 10;
		const BufferType lumaBuffer(data.height, data.width, f32);
		//detect watermark after watermarkInterval frames
		if (framesCount % data.watermarkInterval == 0)
		{
			cuda_utils::launchPitchedToFloatKernel(frame->data[0], lumaBuffer.device<float>(), data.width, data.height, frame->linesize[0], bitDepth, afStream);
			lumaBuffer.unlock();
			float correlation = data.watermarkObj->detectWatermark(lumaBuffer, ME);
			cout << "Correlation for frame: " << (framesCount + 1) << ": " << correlation << "\n";
		}
		framesCount++;
	}
#endif

	//helper method to embed the watermark in the video frame and write it to the ffmpeg pipe
	void embedAndWriteFrame(VideoProcessingContext& data, const BufferType& buffer, const int elements, FILE* ffmpegPipe)
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
	int videoDispatcher(VideoProcessingContext& data, const bool useHwDecoder, const VideoOp op, FILE* ffmpegPipe)
	{
#if defined(_USE_CUDA_)
		if (useHwDecoder) 
			return processFrames(data, [&](const AVFrame* frame, int& framesCount) { 
				op == EMBED ? embedWatermarkHWAccel(data, framesCount, frame, ffmpegPipe) : detectWatermarkHWAccel(data, framesCount, frame);
			});
#endif
		return processFrames(data, [&](const AVFrame* frame, int& framesCount) { 
			op == EMBED ? embedWatermark(data, framesCount, frame, ffmpegPipe) : detectWatermark(data, framesCount, frame);
		});
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
		const int bitDepth = frame->format == AV_PIX_FMT_YUV420P10LE ? 10 : 8;
		//detect watermark after watermarkInterval frames
		if (framesCount % data.watermarkInterval == 0)
		{
			if (bitDepth == 8)
			{
				uint8_t* srcY = frame->data[0];
				if (frame->linesize[0] != data.width)
				{
					for (int y = 0; y < data.height; y++)
						memcpy(data.hostFramePtr + y * data.width, frame->data[0] + y * frame->linesize[0], data.width);
					srcY = data.hostFramePtr;
				}
				loadInputFrame<GrayBuffer>(data, srcY);
			}
			else
			{
				uint16_t* srcY = reinterpret_cast<uint16_t*>(frame->data[0]);
				if (frame->linesize[0] / 2 != data.width)
				{
					for (int y = 0; y < data.height; y++)
						for (int x = 0; x < data.width; x++)
							data.hostFramePtr[y * data.width + x] = scale10to8(srcY[y * (frame->linesize[0] / 2) + x]);
					loadInputFrame<GrayBuffer>(data, data.hostFramePtr);
				}
				else
				    loadInputFrame<GrayExtBuffer>(data, srcY);
			}
			
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

	//runs the watermark creation for a video frame and writes the watermarked frame to the ffmpeg pipe
	void processAndWriteYPlane(const bool embedWatermark, const AVFrame* frame, VideoProcessingContext& data, FILE* ffmpegPipe)
	{
		const int bitDepth = frame->format == AV_PIX_FMT_YUV420P10LE ? 10 : 8;
		uint8_t* srcY = frame->data[0];
		if (bitDepth == 8)
		{
			//if there is row padding (for alignment), we must copy the data to a contiguous block!
			if (frame->linesize[0] != data.width)
			{
				for (int y = 0; y < data.height; y++)
					memcpy(data.hostFramePtr + y * data.width, srcY + y * frame->linesize[0], data.width);
				srcY = data.hostFramePtr;
			}
		}
		else
		{
			//convert (optionally aligned) 16-bit to packed 8-bit
			uint16_t* srcYu16 = reinterpret_cast<uint16_t*>(frame->data[0]);
			for (int y = 0; y < data.height; y++)
				for (int x = 0; x < data.width; x++)
					data.hostFramePtr[y * data.width + x] = scale10to8(srcYu16[y * (frame->linesize[0] / 2) + x]);
			srcY = data.hostFramePtr;
		}
		if (embedWatermark)
		{
			loadInputFrame<GrayBuffer>(data, srcY);
			embedAndWriteFrame(data, data.inputFrame, data.width * data.height, ffmpegPipe);
		}
		else
			fwrite(srcY, 1, data.width * data.height, ffmpegPipe);
	}

	//writes the chroma planes (U and V) to the ffmpeg pipe, either assuming aligned pointers or not
	void writeChromaPlanes(const AVFrame* frame, VideoProcessingContext& data, FILE* ffmpegPipe)
	{
		const int bitDepth = (frame->format == AV_PIX_FMT_YUV420P10LE) ? 10 : 8;
		const int expectedChromaPitch = (bitDepth == 8) ? data.width / 2 : data.width;
		uint8_t* hostPtr = data.hostFramePtr;

		//lambda to write a single chroma plane
		auto writePlane = [&](const uint8_t* src, const int linesize)
		{
			if (bitDepth == 8)
			{
				if (linesize != expectedChromaPitch)
				{
					for (int y = 0; y < data.height / 2; y++)
						fwrite(src + y * linesize, 1, data.width / 2, ffmpegPipe);
				}
				else
					fwrite(src, 1, data.width * data.height / 4, ffmpegPipe);
			}
			else
			{
				if (linesize != expectedChromaPitch)
				{
					for (int y = 0; y < data.height / 2; y++)
					{
						const uint16_t* row = reinterpret_cast<const uint16_t*>(src + y * linesize);
						for (int x = 0; x < data.width / 2; x++)
							hostPtr[x] = scale10to8(row[x]);
						fwrite(hostPtr, 1, data.width / 2, ffmpegPipe);
					}
				}
				else
				{
					const uint16_t* row = reinterpret_cast<const uint16_t*>(src);
					for (int i = 0; i < data.width * data.height / 4; i++)
						hostPtr[i] = scale10to8(row[i]);
					fwrite(hostPtr, 1, data.width * data.height / 4, ffmpegPipe);
				}
			}
		};

		//write U
		writePlane(frame->data[1], frame->linesize[1]);
		hostPtr += data.width * data.height / 4;
		//write V
		writePlane(frame->data[2], frame->linesize[2]);
	}
}