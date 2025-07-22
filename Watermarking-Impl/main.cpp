#if defined(_USE_GPU_)
#include <arrayfire.h>
#elif defined(_USE_EIGEN_)
#include "cimg_init.h"
#include "eigen_utils.hpp"
#include <Eigen/Dense>
#include <omp.h>
#include <thread>
#endif

#include "buffer.hpp"
#include "constants.h"
#include "host_memory.h"
#include "utils.hpp"
#include "videoprocessingcontext.hpp"
#include "video_utils.hpp"
#include "WatermarkBase.hpp"
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <format>
#include <INIReader.h>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/frame.h>
#include <libavutil/log.h>
}

#if defined(_USE_EIGEN_)
using namespace cimg_library;
using namespace Eigen;
#endif

using std::cout;
using std::string;

/*!
 *  \brief  Helper functions for testing the watermark algorithms
 *  \author Dimitris Karatzas
 */
void makeRgbWatermark(const std::unique_ptr<WatermarkBase>& watermarkObj, const BufferType& image, const BufferType& rgbImage, BufferType& output, float& watermarkStrength, MASK_TYPE maskType);
int testForImage(const INIReader& inir, const int p, const float psnr);
int testForVideo(const INIReader& inir, const string& videoFile, const int p, const float psnr);

/*!
 *  \brief  This is a project implementation of my Thesis with title:
 *			EFFICIENT IMPLEMENTATION OF WATERMARKING ALGORITHMS AND
 *			WATERMARK DETECTION IN IMAGE AND VIDEO USING GPU.
 *  \author Dimitris Karatzas
 */
int main(void)
{
	try {
		//open parameters file
		const INIReader inir("settings.ini");
		Utils::checkError(inir.ParseError() < 0, "Could not load settings.ini file");

#if defined(_USE_OPENCL_)
		try {
			af::setDevice(inir.GetInteger("options", "opencl_device", 0));
		}
		catch (const std::exception&) {
			cout << "NOTE: Invalid OpenCL device specified, using default 0" << "\n";
			af::setDevice(0);
		}
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
			numThreads = std::max(omp_get_max_threads(), static_cast<int>(std::thread::hardware_concurrency()));
		omp_set_num_threads(numThreads);
		//check valid parameter values
		Utils::checkError(p <= 1 || p % 2 != 1 || p > 9, "p parameter must be a positive odd number greater than or equal to 3 and less than or equal to 9");
		cout << "Using " << numThreads << " parallel threads.\n";
#else
		//TODO GPU: for p>3 we have problems with ME masking buffers
		Utils::checkError(p != 3, "For now, only p=3 is allowed");
#endif

		//initialize openmp
#pragma omp parallel
		{ }

		Utils::checkError(psnr <= 0, "PSNR must be a positive number");

		//test algorithms
		const string videoFile = inir.Get("paths", "video", "");
		const int code = videoFile != "" ? 
			testForVideo(inir, videoFile, p, psnr) : testForImage(inir, p, psnr);
		return code;
	}
	catch (const std::exception& ex) {
		cout << "Fatal error: " << ex.what() << "\n";
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}

//embed watermark for static images
int testForImage(const INIReader& inir, const int p, const float psnr)
{
	//not hardware specific, but a reasonable limit for images
	constexpr auto maxImageDims = std::pair<unsigned int, unsigned int>(65536, 65536);

	const string imageFile = inir.Get("paths", "image", "NO_IMAGE");
	const bool showFps = inir.GetBoolean("options", "execution_time_in_fps", false);
	int loops = inir.GetInteger("parameters", "loops_for_test", 5);
	loops = loops <= 0 ? 5 : loops;
	cout << "Each test will be executed " << loops << " times. Average time will be shown below\n";
	
	BufferType rgbImage, image;
	std::optional<BufferAlphaType> alphaChannel;
	
	//load image from disk into arrayfire (GPU), or CImg and copy from CImg object to Eigen arrays (CPU)
	double secs = Utils::executionTime([&] { Utils::loadImage(rgbImage, image, imageFile, alphaChannel); });
#if defined(_USE_GPU_)
	const auto rows = static_cast<unsigned int>(image.dims(0));
	const auto cols = static_cast<unsigned int>(image.dims(1));
	cout << "Time to load and transfer RGB image from disk to VRAM: " << secs << "\n\n";
#elif defined(_USE_EIGEN_)
	const auto rows = image.getGray().rows();
	const auto cols = image.getGray().cols();
	cout << "Time to load image from disk and initialize CImg and Eigen memory objects: " << secs << " seconds\n\n";
#endif
	Utils::checkError(cols > maxImageDims.first || rows > maxImageDims.second, "Image dimensions too high");

	float watermarkStrength;
	//initialize watermark functions class, including parameters, ME and custom (NVF in this example) kernels
	const auto watermarkObj = Utils::createWatermarkObject(rows, cols, inir.Get("paths", "watermark", ""), p, psnr);

#if defined(_USE_GPU_)
	//warmup for arrayfire
	watermarkObj->makeWatermark(image, rgbImage, watermarkStrength, NVF);
	watermarkObj->makeWatermark(image, rgbImage, watermarkStrength, ME);
#endif

	BufferType watermarkNVF, watermarkME;
	//make NVF watermark
	secs = Utils::executionTime([&]() { makeRgbWatermark(watermarkObj, image, rgbImage, watermarkNVF, watermarkStrength, NVF); }, loops);
	cout << std::format("Watermark strength (parameter a): {}\nCalculation of NVF mask with {} rows and {} columns and parameters:\np = {}  PSNR(dB) = {}\n{}\n\n", watermarkStrength, rows, cols, p, psnr, Utils::formatExecutionTime(showFps, secs / loops));
	//make ME watermark
	secs = Utils::executionTime([&]() { makeRgbWatermark(watermarkObj, image, rgbImage, watermarkME, watermarkStrength, ME); }, loops);
	cout << std::format("Watermark strength (parameter a): {}\nCalculation of ME mask with {} rows and {} columns and parameters:\np = {}  PSNR(dB) = {}\n{}\n\n", watermarkStrength, rows, cols, p, psnr, Utils::formatExecutionTime(showFps, secs / loops));

#if defined(_USE_GPU_)
	const BufferType watermarkedNVFgray = af::rgb2gray(watermarkNVF, Constants::rPercent, Constants::gPercent, Constants::bPercent);
	const BufferType watermarkedMEgray = af::rgb2gray(watermarkME, Constants::rPercent, Constants::gPercent, Constants::bPercent);
	//warmup for arrayfire
	watermarkObj->detectWatermark(watermarkedNVFgray, NVF);
	watermarkObj->detectWatermark(watermarkedMEgray, ME);
#elif defined(_USE_EIGEN_)
	const BufferType watermarkedNVFgray(eigen_utils::eigenRgbToGray(watermarkNVF.getRGB(), Constants::rPercent, Constants::gPercent, Constants::bPercent));
	const BufferType watermarkedMEgray(eigen_utils::eigenRgbToGray(watermarkME.getRGB(), Constants::rPercent, Constants::gPercent, Constants::bPercent));
#endif

	float correlationNvf, correlationMe;
	//NVF and ME mask detection
	secs = Utils::executionTime([&]() { correlationNvf = watermarkObj->detectWatermark(watermarkedNVFgray, NVF); }, loops);
	cout << std::format("Calculation of the watermark correlation (NVF) of an image with {} rows and {} columns and parameters:\np = {}  PSNR(dB) = {}\n{}\n\n", rows, cols, p, psnr, Utils::formatExecutionTime(showFps, secs / loops));
	secs = Utils::executionTime([&]() { correlationMe = watermarkObj->detectWatermark(watermarkedMEgray, ME); }, loops);
	cout << std::format("Calculation of the watermark correlation (ME) of an image with {} rows and {} columns and parameters:\np = {}  PSNR(dB) = {}\n{}\n\n", rows, cols, p, psnr, Utils::formatExecutionTime(showFps, secs / loops));
	//print the correlation values
	cout << std::format("Correlation [NVF]: {:.16f}\nCorrelation [ME]: {:.16f}\n", correlationNvf, correlationMe);

	//save watermarked images to disk
	if (inir.GetBoolean("options", "save_watermarked_files_to_disk", false)) 
	{
		cout << "\nSaving watermarked files to disk...\n";
#pragma omp parallel sections
		{
#pragma omp section
			Utils::saveImage(imageFile, "W_NVF", watermarkNVF, alphaChannel);
#pragma omp section
			Utils::saveImage(imageFile, "W_ME", watermarkME, alphaChannel);
		}
		cout << "Successully saved to disk\n";
	}
	return EXIT_SUCCESS;
}

//embed watermark for a video or try to detect watermark in a video
int testForVideo(const INIReader& inir, const string& videoFile, const int p, const float psnr)
{
	const bool showFps = inir.GetBoolean("options", "execution_time_in_fps", false);
	const int watermarkInterval = inir.GetInteger("parameters_video", "watermark_interval", 30);

	//set ffmpeg log level
	av_log_set_level(AV_LOG_INFO);

	//load input video
	AVFormatContext* rawInputCtx = nullptr;
	Utils::checkError(avformat_open_input(&rawInputCtx, videoFile.c_str(), nullptr, nullptr) < 0, "ERROR: Failed to open input video file");
	AVFormatContextPtr inputFormatCtx(rawInputCtx, [](AVFormatContext* ctx) { if (ctx) { avformat_close_input(&ctx); } });
	avformat_find_stream_info(inputFormatCtx.get(), nullptr);
	av_dump_format(inputFormatCtx.get(), 0, videoFile.c_str(), 0);

	//find video stream and open video decoder
	const int videoStreamIndex = video_utils::findVideoStream(inputFormatCtx.get());
	Utils::checkError(videoStreamIndex == -1, "ERROR: No video stream found");
	const AVCodecContextPtr inputDecoderCtx(video_utils::openDecoder(inputFormatCtx->streams[videoStreamIndex]->codecpar), [](AVCodecContext* ctx) { avcodec_free_context(&ctx); });

	//initialize watermark functions class and host pinned memory for fast GPU<->CPU transfers, or simple Eigen memory for CPU implementation
	const int height = inputFormatCtx->streams[videoStreamIndex]->codecpar->height;
	const int width = inputFormatCtx->streams[videoStreamIndex]->codecpar->width;
	const auto watermarkObj = Utils::createWatermarkObject(height, width, inir.Get("paths", "watermark", ""), p, psnr);
	HostMemory<uint8_t> framePinned(width * height);

	//group common video data for both embedding and detection
	const VideoProcessingContext videoData(inputFormatCtx.get(), inputDecoderCtx.get(), videoStreamIndex, watermarkObj.get(), height, width, watermarkInterval, framePinned.get());

	//realtime watermarking of raw video
	const string makeWatermarkVideoPath = inir.Get("parameters_video", "encode_watermark_file_path", "");
	if (makeWatermarkVideoPath != "")
	{
		const string ffmpegOptions = inir.Get("parameters_video", "encode_options", "-c:v libx265 -preset fast -crf 23");
		//build the FFmpeg command
		std::ostringstream ffmpegCmd;
		ffmpegCmd << "ffmpeg -y -f rawvideo -pix_fmt yuv420p " << "-s " << width << "x" << height
			<< " -r " << video_utils::getFrameRate(inputFormatCtx.get(), videoStreamIndex) << " -i - -i \"" << videoFile << "\" " << ffmpegOptions
			<< " -c:s copy -c:a copy -map 1:s? -map 0:v -map 1:a? -max_interleave_delta 0 \"" << makeWatermarkVideoPath << "\"";
		cout << "\nFFmpeg encode command: " << ffmpegCmd.str() << "\n\n";

		//open FFmpeg process (with pipe) for writing
		FILEPtr ffmpegPipe(_popen(ffmpegCmd.str().c_str(), "wb"), _pclose);
		Utils::checkError(!ffmpegPipe.get(), "Error: Could not open FFmpeg pipe");

		BufferType inputFrame;
		GrayBuffer watermarkedFrame;
		//embed watermark on the video frames
		double secs = Utils::executionTime([&] { 
			video_utils::processFrames(videoData, [&](const AVFrame* frame, int& framesCount) { 
				video_utils::embedWatermark(videoData, inputFrame, watermarkedFrame, framesCount, frame, ffmpegPipe.get()); 
			}); 
		});
		cout << "\nWatermark embedding total execution time: " << Utils::formatExecutionTime(false, secs) << "\n";
	}

	//realtime watermarked video detection
	else if (inir.GetBoolean("parameters_video", "watermark_detection", false))
	{
		BufferType inputFrame;
		//detect watermark on the video frames
		int framesCount = 1;
		double secs = Utils::executionTime([&] { 
			framesCount = video_utils::processFrames(videoData, [&](const AVFrame* frame, int& framesCount) { 
				video_utils::detectWatermark(videoData, inputFrame, framesCount, frame); 
			}); 
		});
		cout << "\nWatermark detection total execution time: " << Utils::formatExecutionTime(false, secs) << "\n";
		cout << "\nWatermark detection average execution time per frame: " << Utils::formatExecutionTime(showFps, secs / framesCount) << "\n";
	}
	return EXIT_SUCCESS;
}

//creates a watermarked RGB BufferType
void makeRgbWatermark(const std::unique_ptr<WatermarkBase>& watermarkObj, const BufferType& image, const BufferType& rgbImage, BufferType& output, float& watermarkStrength, MASK_TYPE maskType)
{
#if defined(_USE_GPU_)
	output = watermarkObj->makeWatermark(image, rgbImage, watermarkStrength, maskType);
#elif defined(_USE_EIGEN_)
	output = std::move(watermarkObj->makeWatermark(image, rgbImage, watermarkStrength, maskType).getRGB());
#endif
}