#if defined(_USE_CUDA_)
#include "cuda_utils.hpp"
#include "WatermarkCuda.cuh"
#include <cuda_runtime.h>
#include <omp.h>

#elif defined(_USE_OPENCL_)
#include "opencl_init.h"
#include "WatermarkOCL.hpp"
#include <af/opencl.h>

#elif defined(_USE_EIGEN_)
#include "cimg_init.h"
#include "eigen_utils.hpp"
#include "WatermarkEigen.hpp"
#include <CImg.h>
#include <cstdlib>
#include <Eigen/Dense>
#include <omp.h>
#include <thread>
#endif

#include "buffer.hpp"
#include "utils.hpp"
#include "videoprocessingcontext.hpp"
#include <cstdint>
#include <cstring>
#include <exception>
#include <format>
#include "host_memory.h"
#include <functional>
#include <INIReader.h>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include "WatermarkBase.hpp"
#include <utility>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavcodec/packet.h>
#include <libavutil/frame.h>
#include <libavutil/log.h>
#include <libavutil/avutil.h>
#include <libavcodec/codec.h>
#include <libavutil/pixfmt.h>
#include "libavcodec/codec_par.h"
#include "libavutil/rational.h"
}

#if defined(_USE_OPENCL_) || defined(_USE_CUDA_)
#define _USE_GPU_
#endif

#if defined(_USE_EIGEN_)
using namespace cimg_library;
using namespace Eigen;
using GrayBuffer = Array<uint8_t, Dynamic, Dynamic>;
#elif defined(_USE_GPU_) 
using GrayBuffer = af::array;
#endif

using std::cout;
using std::string;
using AVPacketPtr = std::unique_ptr<AVPacket, std::function<void(AVPacket*)>>;
using AVFramePtr = std::unique_ptr<AVFrame, std::function<void(AVFrame*)>>;
using AVFormatContextPtr = std::unique_ptr<AVFormatContext, std::function<void(AVFormatContext*)>>;
using AVCodecContextPtr = std::unique_ptr<AVCodecContext, std::function<void(AVCodecContext*)>>;
using FILEPtr = std::unique_ptr<FILE, decltype(&_pclose)>;

/*!
 *  \brief  Helper methods for testing the watermark algorithms
 *  \author Dimitris Karatzas
 */
void exitProgram(const int exitCode);
std::string executionTime(const bool showFps, const double seconds);
int testForImage(const INIReader& inir, const int p, const float psnr);
int testForVideo(const INIReader& inir, const string& videoFile, const int p, const float psnr);
int findVideoStreamIndex(const AVFormatContext* inputFormatCtx);
AVCodecContext* openDecoderContext(const AVCodecParameters* params);
bool receivedValidVideoFrame(AVCodecContext* inputDecoderCtx, AVPacket* packet, AVFrame* frame, const int videoStreamIndex);
std::string getVideoFrameRate(const AVFormatContext* inputFormatCtx, const int videoStreamIndex);
void embedWatermarkFrame(const VideoProcessingContext& data, BufferType& inputFrame, GrayBuffer& watermarkedFrame, int& framesCount, AVFrame* frame, FILE* ffmpegPipe);
void detectFrameWatermark(const VideoProcessingContext& data, BufferType& inputFrame, int& framesCount, AVFrame* frame);
int processFrames(const VideoProcessingContext& data, std::function<void(AVFrame*, int&)> processFrame);
void makeRgbWatermarkBuffer(const std::unique_ptr<WatermarkBase>& watermarkObj, const BufferType& image, const BufferType& rgbImage, BufferType& output, float& watermarkStrength, MASK_TYPE maskType);
void writeWatermarkeFrameToPipe(const VideoProcessingContext& data, BufferType& inputFrame, GrayBuffer& watermarkedFrame, AVFrame* frame, FILE* ffmpegPipe);
void writeConditionallyWatermarkeFrameToPipe(const bool embedWatermark, const VideoProcessingContext& data, BufferType& inputFrame, GrayBuffer& watermarkedFrame, AVFrame* frame, FILE* ffmpegPipe);
void checkError(const bool criticalErrorCondition, const string& errorMessage);

/*!
 *  \brief  This is a project implementation of my Thesis with title:
 *			EFFICIENT IMPLEMENTATION OF WATERMARKING ALGORITHMS AND
 *			WATERMARK DETECTION IN IMAGE AND VIDEO USING GPU, CUDA version
 *  \author Dimitris Karatzas
 */
int main(void)
{
	//open parameters file
	const INIReader inir("settings.ini");
	checkError(inir.ParseError() < 0, "Could not load settings.ini file");

#if defined(_USE_OPENCL_)
	try {
		af::setDevice(inir.GetInteger("options", "opencl_device", 0));
	}
	catch (const std::exception&) {
		cout << "NOTE: Invalid OpenCL device specified, using default 0" << "\n";
		af::setDevice(0);
	}

#elif defined(_USE_CUDA_)
	omp_set_num_threads(omp_get_max_threads());
#pragma omp parallel for
	for (int i = 0; i < 24; i++) { }
#endif
#if defined(_USE_GPU_)
	af::info();
	cout << "\n";
#endif
	const int p = inir.GetInteger("parameters", "p", -1);
	const float psnr = inir.GetFloat("parameters", "psnr", -1.0f);

#if defined(_USE_EIGEN_)
	int numThreads = inir.GetInteger("parameters", "threads", 0);
	if (numThreads <= 0)
	{
		auto threadsSupported = std::thread::hardware_concurrency();
		numThreads = threadsSupported == 0 ? 2 : threadsSupported;
	}

	//openmp initialization
	omp_set_num_threads(numThreads);
#pragma omp parallel for
	for (int i = 0; i < 24; i++) {}
	//check valid parameter values
	checkError(p <= 1 || p % 2 != 1 || p > 9, "p parameter must be a positive odd number greater than or equal to 3 and less than or equal to 9");
	cout << "Using " << numThreads << " parallel threads.\n";
#else
	//TODO GPU: for p>3 we have problems with ME masking buffers
	checkError(p != 3, "For now, only p=3 is allowed");
#endif
	checkError(psnr <= 0, "PSNR must be a positive number");

	//test algorithms
	try {
		const string videoFile = inir.Get("paths", "video", "");
		const int code = videoFile != "" ?
			testForVideo(inir, videoFile, p, psnr) :
			testForImage(inir, p, psnr);
		exitProgram(code);
	}
	catch (const std::exception& ex) {
		cout << ex.what() << "\n";
		exitProgram(EXIT_FAILURE);
	}
	exitProgram(EXIT_SUCCESS);
}

//embed watermark for static images
int testForImage(const INIReader& inir, const int p, const float psnr)
{
	constexpr float rPercent = 0.299f;
	constexpr float gPercent = 0.587f;
	constexpr float bPercent = 0.114f;
	const string imageFile = inir.Get("paths", "image", "NO_IMAGE");
	const bool showFps = inir.GetBoolean("options", "execution_time_in_fps", false);
	int loops = inir.GetInteger("parameters", "loops_for_test", 5);
	loops = loops <= 0 ? 5 : loops;
	cout << "Each test will be executed " << loops << " times. Average time will be shown below\n";

#if defined(_USE_GPU_)
	const auto maxImageDims = Utilities::getMaxImageSize();
	//load image from disk into an arrayfire array
	BufferType rgbImage, image;
	double secs = Utilities::executionTime([&] {
		rgbImage = af::loadImage(imageFile.c_str(), true);
		image = af::rgb2gray(rgbImage, rPercent, gPercent, bPercent);
		af::sync();
	});
	const auto rows = static_cast<unsigned int>(image.dims(0));
	const auto cols = static_cast<unsigned int>(image.dims(1));
	cout << "Time to load and transfer RGB image from disk to VRAM: " << secs << "\n\n";
#elif defined(_USE_EIGEN_)
	constexpr auto maxImageDims = std::pair<unsigned int, unsigned int>(65536, 65536);
	BufferType rgbImage, image;
	//load image from disk into CImg and copy from CImg object to Eigen arrays
	double secs = Utilities::executionTime([&] {
		rgbImage = BufferType(cimgToEigen3dArray(CImg<float>(imageFile.c_str())));
		image = BufferType(eigen3dArrayToGrayscaleArray(rgbImage.getRGB(), rPercent, gPercent, bPercent));
	});
	const auto rows = image.getGray().rows();
	const auto cols = image.getGray().cols();
	cout << "Time to load image from disk and initialize CImg and Eigen memory objects: " << secs << " seconds\n\n";
#endif
	checkError(cols < 64 || rows < 64, "Image dimensions too low");
	checkError(cols > maxImageDims.first || rows > maxImageDims.second, "Image dimensions too high");

	float watermarkStrength;
	//initialize watermark functions class, including parameters, ME and custom (NVF in this example) kernels
	std::unique_ptr<WatermarkBase> watermarkObj = Utilities::createWatermarkObject(rows, cols, inir.Get("paths", "watermark", ""), p, psnr);

#if defined(_USE_GPU_)
	//warmup for arrayfire
	watermarkObj->makeWatermark(image, rgbImage, watermarkStrength, MASK_TYPE::NVF);
	watermarkObj->makeWatermark(image, rgbImage, watermarkStrength, MASK_TYPE::ME);
#endif

	BufferType watermarkNVF, watermarkME;
	//make NVF watermark
	secs = Utilities::executionTime([&]() { makeRgbWatermarkBuffer(watermarkObj, image, rgbImage, watermarkNVF, watermarkStrength, MASK_TYPE::NVF); }, loops);
	cout << std::format("Watermark strength (parameter a): {}\nCalculation of NVF mask with {} rows and {} columns and parameters:\np = {}  PSNR(dB) = {}\n{}\n\n", watermarkStrength, rows, cols, p, psnr, executionTime(showFps, secs / loops));
	//make ME watermark
	secs = Utilities::executionTime([&]() { makeRgbWatermarkBuffer(watermarkObj, image, rgbImage, watermarkME, watermarkStrength, MASK_TYPE::ME); }, loops);
	cout << std::format("Watermark strength (parameter a): {}\nCalculation of ME mask with {} rows and {} columns and parameters:\np = {}  PSNR(dB) = {}\n{}\n\n", watermarkStrength, rows, cols, p, psnr, executionTime(showFps, secs / loops));

#if defined(_USE_GPU_)
	const BufferType watermarkedNVFgray = af::rgb2gray(watermarkNVF, rPercent, gPercent, bPercent);
	const BufferType watermarkedMEgray = af::rgb2gray(watermarkME, rPercent, gPercent, bPercent);
	//warmup for arrayfire
	watermarkObj->detectWatermark(watermarkedNVFgray, MASK_TYPE::NVF);
	watermarkObj->detectWatermark(watermarkedMEgray, MASK_TYPE::ME);
#elif defined(_USE_EIGEN_)
	const BufferType watermarkedNVFgray(eigen3dArrayToGrayscaleArray(watermarkNVF.getRGB(), rPercent, gPercent, bPercent));
	const BufferType watermarkedMEgray(eigen3dArrayToGrayscaleArray(watermarkME.getRGB(), rPercent, gPercent, bPercent));
#endif

	float correlationNvf, correlationMe;
	//NVF and ME mask detection
	secs = Utilities::executionTime([&]() { correlationNvf = watermarkObj->detectWatermark(watermarkedNVFgray, MASK_TYPE::NVF); }, loops);
	cout << std::format("Calculation of the watermark correlation (NVF) of an image with {} rows and {} columns and parameters:\np = {}  PSNR(dB) = {}\n{}\n\n", rows, cols, p, psnr, executionTime(showFps, secs / loops));
	secs = Utilities::executionTime([&]() { correlationMe = watermarkObj->detectWatermark(watermarkedMEgray, MASK_TYPE::ME); }, loops);
	cout << std::format("Calculation of the watermark correlation (ME) of an image with {} rows and {} columns and parameters:\np = {}  PSNR(dB) = {}\n{}\n\n", rows, cols, p, psnr, executionTime(showFps, secs / loops));
	//print the correlation values
	cout << std::format("Correlation [NVF]: {:.16f}\nCorrelation [ME]: {:.16f}\n", correlationNvf, correlationMe);

	//save watermarked images to disk
	if (inir.GetBoolean("options", "save_watermarked_files_to_disk", false)) 
	{
		cout << "\nSaving watermarked files to disk...\n";
#if defined(_USE_OPENCL_)
		Utilities::saveImage(imageFile, "W_NVF", watermarkNVF);
		Utilities::saveImage(imageFile, "W_ME", watermarkME);
#elif defined(_USE_CUDA_) || defined(_USE_EIGEN_)
#pragma omp parallel sections
		{
#pragma omp section
			Utilities::saveImage(imageFile, "W_NVF", watermarkNVF);
#pragma omp section
			Utilities::saveImage(imageFile, "W_ME", watermarkME);
		}
#endif
		cout << "Successully saved to disk\n";
	}
	return EXIT_SUCCESS;
}

//embed watermark for a video or try to detect watermark in a video
int testForVideo(const INIReader& inir, const string& videoFile, const int p, const float psnr)
{
	const bool showFps = inir.GetBoolean("options", "execution_time_in_fps", false);
	const int watermarkInterval = inir.GetInteger("parameters_video", "watermark_interval", 30);

	//Set ffmpeg log level
	av_log_set_level(AV_LOG_INFO);

	//Load input video
	AVFormatContext* rawInputCtx = nullptr;
	checkError(avformat_open_input(&rawInputCtx, videoFile.c_str(), nullptr, nullptr) < 0, "ERROR: Failed to open input video file");
	AVFormatContextPtr inputFormatCtx(rawInputCtx, [](AVFormatContext* ctx) { if (ctx) { avformat_close_input(&ctx); } });
	avformat_find_stream_info(inputFormatCtx.get(), nullptr);
	av_dump_format(inputFormatCtx.get(), 0, videoFile.c_str(), 0);

	//Find video stream and open video decoder
	const int videoStreamIndex = findVideoStreamIndex(inputFormatCtx.get());
	checkError(videoStreamIndex == -1, "ERROR: No video stream found");
	const AVCodecContextPtr inputDecoderCtx(openDecoderContext(inputFormatCtx->streams[videoStreamIndex]->codecpar), [](AVCodecContext* ctx) { avcodec_free_context(&ctx); });

	//initialize watermark functions class
	const int height = inputFormatCtx->streams[videoStreamIndex]->codecpar->height;
	const int width = inputFormatCtx->streams[videoStreamIndex]->codecpar->width;
	//initialize watermark functions class, including parameters, ME and custom (NVF in this example) kernels
	std::unique_ptr<WatermarkBase> watermarkObj = Utilities::createWatermarkObject(height, width, inir.Get("paths", "watermark", ""), p, psnr);

	//initialize host pinned memory for fast GPU<->CPU transfers, or simple Eigen memory for CPU implementation
	HostMemory<uint8_t> framePinned(width * height);

	//group common video data for both embedding and detection
	const VideoProcessingContext videoData(inputFormatCtx.get(), inputDecoderCtx.get(), videoStreamIndex, watermarkObj.get(), height, width, watermarkInterval, framePinned.get());

	//realtime watermarking of raw video
	const string makeWatermarkVideoPath = inir.Get("parameters_video", "encode_watermark_file_path", "");
	if (makeWatermarkVideoPath != "")
	{
		const string ffmpegOptions = inir.Get("parameters_video", "encode_options", "-c:v libx265 -preset fast -crf 23");
		// Build the FFmpeg command
		std::ostringstream ffmpegCmd;
		ffmpegCmd << "ffmpeg -y -f rawvideo -pix_fmt yuv420p " << "-s " << width << "x" << height
			<< " -r " << getVideoFrameRate(inputFormatCtx.get(), videoStreamIndex) << " -i - -i " << videoFile << " " << ffmpegOptions
			<< " -c:s copy -c:a copy -map 1:s? -map 0:v -map 1:a? -max_interleave_delta 0 " << makeWatermarkVideoPath;
		cout << "\nFFmpeg encode command: " << ffmpegCmd.str() << "\n\n";

		// Open FFmpeg process (with pipe) for writing
		FILEPtr ffmpegPipe(_popen(ffmpegCmd.str().c_str(), "wb"), _pclose);
		checkError(!ffmpegPipe.get(), "Error: Could not open FFmpeg pipe");

		BufferType inputFrame;
		GrayBuffer watermarkedFrame;
		//embed watermark on the video frames
		double secs = Utilities::executionTime([&] { processFrames(videoData, [&](AVFrame* frame, int& framesCount) { embedWatermarkFrame(videoData, inputFrame, watermarkedFrame, framesCount, frame, ffmpegPipe.get()); }); });
		processFrames(videoData, [&](AVFrame* frame, int& framesCount) { embedWatermarkFrame(videoData, inputFrame, watermarkedFrame, framesCount, frame, ffmpegPipe.get()); });
		cout << "\nWatermark embedding total execution time: " << executionTime(false, secs) << "\n";
	}

	//realtime watermarked video detection
	else if (inir.GetBoolean("parameters_video", "watermark_detection", false))
	{
		BufferType inputFrame;
		//detect watermark on the video frames
		int framesCount;
		double secs = Utilities::executionTime([&] { framesCount = processFrames(videoData, [&](AVFrame* frame, int& framesCount) { detectFrameWatermark(videoData, inputFrame, framesCount, frame); }); });
		cout << "\nWatermark detection total execution time: " << executionTime(false, secs) << "\n";
		cout << "\nWatermark detection average execution time per frame: " << executionTime(showFps, secs / framesCount) << "\n";
	}
	return EXIT_SUCCESS;
}

//Main frames loop logic for video watermark embedding and detection
int processFrames(const VideoProcessingContext& data, std::function<void(AVFrame*, int&)> processFrame)
{
	const AVPacketPtr packet(av_packet_alloc(), [](AVPacket* pkt) { av_packet_free(&pkt); });
	const AVFramePtr frame(av_frame_alloc(), [](AVFrame* frame) { av_frame_free(&frame); });
	int framesCount = 0;

	// Read video frames loop
	while (av_read_frame(data.inputFormatCtx, packet.get()) >= 0)
	{
		if (!receivedValidVideoFrame(data.inputDecoderCtx, packet.get(), frame.get(), data.videoStreamIndex))
			continue;
		processFrame(frame.get(), framesCount);
	}
	// Ensure all remaining frames are flushed
	avcodec_send_packet(data.inputDecoderCtx, nullptr);
	while (avcodec_receive_frame(data.inputDecoderCtx, frame.get()) == 0)
	{
		if (frame->format == data.inputDecoderCtx->pix_fmt)
			processFrame(frame.get(), framesCount);
	}
	return framesCount;
}

// Embed watermark in a video frame
void embedWatermarkFrame(const VideoProcessingContext& data, BufferType& inputFrame, GrayBuffer& watermarkedFrame, int& framesCount, AVFrame* frame, FILE* ffmpegPipe)
{
	const bool embedWatermark = framesCount % data.watermarkInterval == 0;
	//if there is row padding (for alignment), we must copy the data to a contiguous block!
	if (frame->linesize[0] != data.width)
	{
		if (embedWatermark)
		{
			for (int y = 0; y < data.height; y++)
				memcpy(data.inputFramePtr + y * data.width, frame->data[0] + y * frame->linesize[0], data.width);
			//embed the watermark, receive the watermarked data back to host and write the watermarked image data to ffmpeg pipe
			writeWatermarkeFrameToPipe(data, inputFrame, watermarkedFrame, frame, ffmpegPipe);
		}
		else
		{
			//write from frame buffer row-by-row the the valid image data (and not the alignment bytes)
			for (int y = 0; y < data.height; y++)
				fwrite(frame->data[0] + y * frame->linesize[0], 1, data.width, ffmpegPipe);
		}
		//always write UI planes as-is
		for (int y = 0; y < data.height / 2; y++)
			fwrite(frame->data[1] + y * frame->linesize[1], 1, data.width / 2, ffmpegPipe);
		for (int y = 0; y < data.height / 2; y++)
			fwrite(frame->data[2] + y * frame->linesize[2], 1, data.width / 2, ffmpegPipe);

	}
	//no row padding, read and write data directly
	else
	{
		writeConditionallyWatermarkeFrameToPipe(embedWatermark, data, inputFrame, watermarkedFrame, frame, ffmpegPipe);
		fwrite(frame->data[1], 1, data.width * frame->height / 4, ffmpegPipe);
		fwrite(frame->data[2], 1, data.width * frame->height / 4, ffmpegPipe);
	}
	framesCount++;
}

// Detect the watermark for a video frame
void detectFrameWatermark(const VideoProcessingContext& data, BufferType& inputFrame, int& framesCount, AVFrame* frame)
{
	//detect watermark after X frames
	if (framesCount % data.watermarkInterval == 0)
	{
		//if there is row padding (for alignment), we must copy the data to a contiguous block!
		const bool rowPadding = frame->linesize[0] != data.width;
		if (rowPadding)
		{
			for (int y = 0; y < data.height; y++)
				memcpy(data.inputFramePtr + y * data.width, frame->data[0] + y * frame->linesize[0], data.width);
		}
		//supply the input frame to the GPU and run the detection of the watermark
#if defined(_USE_GPU_)
		inputFrame = GrayBuffer(data.width, data.height, rowPadding ? data.inputFramePtr : frame->data[0], afHost).T().as(f32);
#elif defined(_USE_EIGEN_)
		inputFrame = BufferType(Map<GrayBuffer>(rowPadding ? data.inputFramePtr : frame->data[0], data.width, data.height).transpose().cast<float>());
#endif
		float correlation = data.watermarkObj->detectWatermark(inputFrame, MASK_TYPE::ME);
		cout << "Correlation for frame: " << framesCount << ": " << correlation << "\n";
	}
	framesCount++;
}

// find the first video stream index
int findVideoStreamIndex(const AVFormatContext* inputFormatCtx)
{
	for (unsigned int i = 0; i < inputFormatCtx->nb_streams; i++)
		if (inputFormatCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO)
			return i;
	return -1;
}

//open decoder context for video
AVCodecContext* openDecoderContext(const AVCodecParameters* inputCodecParams)
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

// Get the input video FPS (average)
string getVideoFrameRate(const AVFormatContext* inputFormatCtx, const int videoStreamIndex)
{
	const AVRational frameRate = inputFormatCtx->streams[videoStreamIndex]->avg_frame_rate;
	return std::format("{:.3f}", static_cast<float>(frameRate.num) / frameRate.den);
}

//supply a packet to the decoder and check if the received frame is valid by checking its format
bool receivedValidVideoFrame(AVCodecContext* inputDecoderCtx, AVPacket* packet, AVFrame* frame, const int videoStreamIndex)
{
	if (packet->stream_index != videoStreamIndex)
	{
		av_packet_unref(packet);
		return false;
	}
	int sendPacketResult = avcodec_send_packet(inputDecoderCtx, packet);
	av_packet_unref(packet);
	if (sendPacketResult != 0 || avcodec_receive_frame(inputDecoderCtx, frame) != 0)
		return false;
	const bool validFormat = frame->format == AV_PIX_FMT_YUV420P || frame->format == AV_PIX_FMT_YUVJ420P;
	checkError(!validFormat, "Error: Video frame format not supported, aborting");
	return validFormat;
}

//helper method to calculate execution time in FPS or in seconds
string executionTime(const bool showFps, const double seconds) 
{
	return showFps ? std::format("FPS: {:.2f} FPS", 1.0 / seconds) : std::format("{:.6f} seconds", seconds);
}

//terminates the program
void exitProgram(const int exitCode) 
{
	std::system("pause");
	std::exit(exitCode);
}

//creates a watermarked RGB BufferType
void makeRgbWatermarkBuffer(const std::unique_ptr<WatermarkBase>& watermarkObj, const BufferType& image, const BufferType& rgbImage, BufferType& output, float& watermarkStrength, MASK_TYPE maskType)
{
#if defined(_USE_GPU_)
	output = watermarkObj->makeWatermark(image, rgbImage, watermarkStrength, maskType);
#elif defined(_USE_EIGEN_)
	output = std::move(watermarkObj->makeWatermark(image, rgbImage, watermarkStrength, maskType).getRGB());
#endif
}
// runs the watermark creation for a video frame and writes the watermarked frame to the ffmpeg pipe
void writeWatermarkeFrameToPipe(const VideoProcessingContext& data, BufferType& inputFrame, GrayBuffer& watermarkedFrame, AVFrame* frame, FILE* ffmpegPipe)
{
	float watermarkStrength;
#if defined(_USE_GPU_)
	inputFrame = BufferType(data.width, data.height, data.inputFramePtr, afHost).T().as(f32);
	watermarkedFrame = data.watermarkObj->makeWatermark(inputFrame, inputFrame, watermarkStrength, MASK_TYPE::ME).as(u8).T();
	watermarkedFrame.host(data.inputFramePtr);
	fwrite(data.inputFramePtr, 1, data.width * frame->height, ffmpegPipe);
#elif defined(_USE_EIGEN_)
	inputFrame = BufferType(Map<GrayBuffer>(data.inputFramePtr, data.width, data.height).transpose().cast<float>().eval());
	watermarkedFrame = data.watermarkObj->makeWatermark(inputFrame, inputFrame, watermarkStrength, MASK_TYPE::ME).getGray().transpose().cast<uint8_t>();
	fwrite(watermarkedFrame.data(), 1, data.width * frame->height, ffmpegPipe);
#endif
}

// runs the watermark creation for a video frame and writes the watermarked frame to the ffmpeg pipe, if the watermark is embedded, or writes the original frame data otherwise
void writeConditionallyWatermarkeFrameToPipe(const bool embedWatermark, const VideoProcessingContext& data, BufferType& inputFrame, GrayBuffer& watermarkedFrame, AVFrame* frame, FILE* ffmpegPipe)
{
	if (embedWatermark)
	{
		float watermarkStrength;
#if defined(_USE_GPU_)
		inputFrame = BufferType(data.width, data.height, frame->data[0], afHost).T().as(f32);
		watermarkedFrame = data.watermarkObj->makeWatermark(inputFrame, inputFrame, watermarkStrength, MASK_TYPE::ME).as(u8).T();
		watermarkedFrame.host(data.inputFramePtr);
	}
	fwrite(embedWatermark ? data.inputFramePtr : frame->data[0], 1, data.width * frame->height, ffmpegPipe);
#elif defined(_USE_EIGEN_)
		inputFrame = BufferType(Map<GrayBuffer>(frame->data[0], data.width, data.height).transpose().cast<float>().eval());
		watermarkedFrame = data.watermarkObj->makeWatermark(inputFrame, inputFrame, watermarkStrength, MASK_TYPE::ME).getGray().transpose().cast<uint8_t>();
	}
	fwrite(embedWatermark ? watermarkedFrame.data() : frame->data[0], 1, data.width* frame->height, ffmpegPipe);
#endif
}

//prints an error message and terminates the program if an error condition is true
void checkError(const bool criticalErrorCondition, const string& errorMessage)
{
	if (criticalErrorCondition)
	{
		cout << errorMessage << "\n";
		exitProgram(EXIT_FAILURE);
	}
};