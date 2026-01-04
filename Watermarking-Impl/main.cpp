#if defined(_USE_GPU_)
#include <arrayfire.h>
#elif defined(_USE_EIGEN_)
#include "cimg_init.h"
#include "eigen_utils.hpp"
#include <Eigen/Dense>
#include <omp.h>
#endif
#include "buffer.hpp"
#include "HostMemory.hpp"
#include "utils.hpp"
#include "ImageFileBuffer.hpp"
#include "INIReader.h"
#include "VideoProcessingContext.hpp"
#include "video_defines.hpp"
#include "video_utils.hpp"
#include "WatermarkBase.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <format>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <utility>

extern "C" {
#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
#include "libavutil/log.h"
}

using namespace video_utils;
#if defined(_USE_EIGEN_)
using namespace cimg_library;
using namespace Eigen;
#endif

using std::cout;
using std::string;

using FILEPtr = std::unique_ptr<FILE, decltype(&_pclose)>;

/*!
 *  \brief  Helper functions for testing the watermark algorithms
 *  \author Dimitris Karatzas
 */
int testForImage(const INIReader& inir, const int p, const float psnr);
int testForVideo(const INIReader& inir, const string& videoFile, const int p, const float psnr);
static inline string info(const string& str) { return "\033[38;5;208m" + str + "\033[0m"; }
static inline string err(const string& str) { return "\033[91m" + str + "\033[0m"; }
static inline string success(const string& str) { return "\033[92m" + str + "\033[0m"; }

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

		//initialize GPU specific backend data (OpenCL and CUDA)
#if defined(_USE_OPENCL_)
		try {
			af::setDevice(inir.GetInteger("options", "opencl_device", 0));
		}
		catch (const std::exception&) {
			cout << info("NOTE: Invalid OpenCL device specified, using default 0\n");
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
		//check valid parameter values
		Utils::checkError(p <= 1 || p % 2 != 1 || p > 9, "p parameter must be a positive odd number greater than or equal to 3 and less than or equal to 9");
		//initialize openmp
#pragma omp parallel
		{}
#else
		//TODO GPU: for p>3 we have problems with ME masking buffers
		Utils::checkError(p != 3, "For now, only p=3 is allowed");
#endif
		Utils::checkError(psnr <= 0, "PSNR must be a positive number");

		//test algorithms
		const string videoFile = inir.Get("paths", "video", "");
		const int code = videoFile != "" ? 
			testForVideo(inir, videoFile, p, psnr) : testForImage(inir, p, psnr);
		return code;
	}
	catch (const std::exception& ex) {
		cout << err(string("Fatal error: ") + ex.what() + "\n");
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
#if defined(_USE_EIGEN_)
	cout << info("\nUsing " + std::to_string(omp_get_max_threads()) + " parallel threads for Watermark calculations.\n");
	const int warmupLoops = 0;
#else
	const int warmupLoops = 1; //warmup for GPU kernels initialization
#endif
	cout << "Each test will be executed " << loops << " times. Average time will be shown below\n";

	ImageFileBuffer imgBuffer;
	//load image from disk into arrayfire (GPU), or CImg and copy from CImg object to Eigen arrays (CPU)
	double secs = Utils::executionTime([&] { Utils::loadImage(imgBuffer, imageFile); });
	auto& [rgbImage, image, alphaChannel, rows, cols, isRGB] = imgBuffer;
	cout << "Time to load image data from disk: " << secs << " seconds\n\n";
	Utils::checkError(cols > maxImageDims.first || rows > maxImageDims.second, "Image dimensions too high");

	float watermarkStrength = std::numeric_limits<float>::quiet_NaN(); //gpu release builds do not return strength to host for efficiency (always nan)
	//initialize watermark functions class, including parameters, ME and custom (NVF in this example) kernels
	const auto watermarkObj = Utils::createWatermarkObject(rows, cols, inir.Get("paths", "watermark", ""), p, psnr);
#if defined(_USE_GPU_)
	ImageBuffer watermarkNVF, watermarkME;
#elif defined(_USE_EIGEN_)
	ImageBuffer watermarkNVF = isRGB ? ImageBuffer(eigen_utils::makeEigenRGB(rows, cols)) : ImageBuffer(ArrayXXf(rows, cols));
	ImageBuffer watermarkME = isRGB ? ImageBuffer(eigen_utils::makeEigenRGB(rows, cols)) : ImageBuffer(ArrayXXf(rows, cols));
#endif

	//helper lambdas to run watermark embedding and detection with time measurement and output
	auto runMakeWatermark = [&](auto& outWatermark, auto method, const std::string& methodName) {
		double s = Utils::executionTime([&]() {
			watermarkObj->makeWatermark(image, rgbImage, outWatermark, watermarkStrength, method);
			//for GPU case make sure we include the final calculation times by syncing!
#if defined(_USE_GPU_)
			outWatermark.eval();
			af::sync();
#endif
		}, loops, warmupLoops);
		cout << std::format("Watermark strength (parameter a): {}\nCalculation of {} mask with {} rows and {} columns and parameters:\np = {}  PSNR(dB) = {}\n{}\n\n", watermarkStrength, methodName, rows, cols, p, psnr, Utils::formatExecutionTime(showFps, s / loops));
	};
	auto runDetectWatermark = [&](const auto& watermark, auto method, const std::string& methodName, float& outCorr) {
		const auto watermarkGray = isRGB ? ImageBuffer(Utils::rgb2gray(watermark)) : ImageBuffer(watermark);
		double s = Utils::executionTime([&]() { outCorr = watermarkObj->detectWatermark(watermarkGray, method); }, loops, warmupLoops);
		cout << std::format("Calculation of the watermark correlation ({}) of an image with {} rows and {} columns and parameters:\np = {}  PSNR(dB) = {}\n{}\n\n", methodName, rows, cols, p, psnr, Utils::formatExecutionTime(showFps, s / loops));
	};

	//embed watermark
	runMakeWatermark(watermarkNVF, NVF, "NVF");
	runMakeWatermark(watermarkME, ME, "ME");
	//detect watermark
	float correlationNvf, correlationMe;
	runDetectWatermark(watermarkNVF, NVF, "NVF", correlationNvf);
	runDetectWatermark(watermarkME, ME, "ME", correlationMe);

	//print the correlation values
	cout << std::format("Correlation [NVF]: {:.16f}\n", correlationNvf);
	cout << std::format("Correlation [ME]:  {:.16f}\n", correlationMe);

	//save watermarked images to disk
	if (inir.GetBoolean("options", "save_watermarked_files_to_disk", false)) 
	{
		cout << "\nSaving watermarked files to disk...\n";
		Utils::saveImage(imageFile, "W_NVF", watermarkNVF, alphaChannel);
		Utils::saveImage(imageFile, "W_ME", watermarkME, alphaChannel);
		cout << success("Successully saved to disk\n");
	}
	return EXIT_SUCCESS;
}

//embed watermark for a video or try to detect watermark in a video
int testForVideo(const INIReader& inir, const string& videoFile, const int p, const float psnr)
{
	const bool showFps = inir.GetBoolean("options", "execution_time_in_fps", false);
	const int watermarkInterval = std::max(1, static_cast<int>(inir.GetInteger("parameters_video", "watermark_interval", 1)));

	//set ffmpeg log level
	av_log_set_level(AV_LOG_INFO);

	//load input video
	AVFormatContext* rawInputCtx = nullptr;
	Utils::checkError(avformat_open_input(&rawInputCtx, videoFile.c_str(), nullptr, nullptr) < 0, "ERROR: Failed to open input video file");
	AVFormatContextPtr inputFormatCtx(rawInputCtx);
	avformat_find_stream_info(inputFormatCtx.get(), nullptr);

	//find video stream and open video decoder
	const int videoStreamIndex = findVideoStream(inputFormatCtx.get());
	Utils::checkError(videoStreamIndex == -1, "ERROR: No video stream found");
	const AVStream* videoStream = inputFormatCtx->streams[videoStreamIndex];

	bool useHwDecoder = false;
	const string hwCodec = inir.Get("parameters_video", "cuda_hw_decoder", "");
	const AVCodecContextPtr inputDecoderCtx = openDecoder(videoStream->codecpar, hwCodec, useHwDecoder);
	if (!hwCodec.empty() && !useHwDecoder && !inputDecoderCtx->hw_device_ctx)
		cout << info("WARNING: Hardware decoder '" + hwCodec + "' was requested, but not available. Using software decoder instead.\n");
	Utils::checkError(!inputDecoderCtx.get(), "ERROR: Could not open video decoder");

	//initialize watermark functions class and host pinned memory for fast GPU<->CPU transfers, or simple Eigen memory for CPU implementation
	const int height = videoStream->codecpar->height;
	const int width = videoStream->codecpar->width;
	const auto watermarkObj = Utils::createWatermarkObject(height, width, inir.Get("paths", "watermark", ""), p, psnr);
	//if CUDA HW decoder is used, allocate more pinned memory for YUV420 frames (3 planes: Y, U, V)
	HostMemory<uint8_t> framePinned(useHwDecoder ? width * height * 3 / 2 : width * height);
	//group common video data for both embedding and detection
	VideoProcessingContext videoData(inputFormatCtx.get(), inputDecoderCtx.get(), videoStreamIndex, videoStream, watermarkObj.get(), watermarkInterval, framePinned.get());

	//realtime watermarking of raw video
	const string makeWatermarkVideoPath = inir.Get("parameters_video", "encode_watermark_file_path", "");
	if (makeWatermarkVideoPath != "")
	{
		//for embedding we may need to apply a filter graph (for 10-bit or HDR content)
		const bool needsFilter = initFilterGraph(inputDecoderCtx.get(), videoStream, useHwDecoder, videoData.filterGraphContext);
#if defined(_USE_EIGEN_)
		//for video embedding only, set the number of openmp/eigen threads to physical cores
		eigen_utils::setThreadsToPhysicalCores();
		cout << info("\nUsing " + std::to_string(omp_get_max_threads()) + " parallel threads for Watermark calculations.\n");
#endif
		const string ffmpegOptions = inir.Get("parameters_video", "encode_options", "-c:v libx265 -preset fast -crf 23");
		//build the FFmpeg command
		std::ostringstream ffmpegCmd;
		ffmpegCmd << "ffmpeg -y -f rawvideo "<< getPixFmt(videoStream) << "-s " << width << "x" << height
			<< " -r " << getFrameRate(videoStream) << " -i - -i \"" << videoFile << "\" " << ffmpegOptions
			<< " -c:s copy -c:a copy -map 1:s? -map 0:v -map 1:a? -max_interleave_delta 0 " 
			<< getStreamRotation(videoStream) << getColorRange(videoStream) << "\"" << makeWatermarkVideoPath << "\"";
		cout << info("\nFFmpeg encode command: " + ffmpegCmd.str() + "\n\n");

		//open FFmpeg process (with pipe) for writing
		FILEPtr ffmpegPipe(_popen(ffmpegCmd.str().c_str(), "wb"), _pclose);
		Utils::checkError(!ffmpegPipe.get(), "Error: Could not open FFmpeg pipe");
		//embed watermark on the video frames
		double secs = Utils::executionTime([&] { videoDispatcher(videoData, useHwDecoder, VideoOp::EMBED, needsFilter, ffmpegPipe.get()); });
		cout << info("\n\nWatermark embedding total execution time: " + Utils::formatExecutionTime(false, secs) + "\n\n");
	}

	//realtime watermarked video detection
	else if (inir.GetBoolean("parameters_video", "watermark_detection", false))
	{
#if defined(_USE_EIGEN_)
		cout << info("\nUsing " + std::to_string(omp_get_max_threads()) + " parallel threads for Watermark calculations.\n");
#endif
		//detect watermark on the video frames
		int framesCount = 1;
		double secs = Utils::executionTime([&] { framesCount = videoDispatcher(videoData, useHwDecoder, VideoOp::DETECT); });
		cout << info("\nWatermark detection total execution time: " + Utils::formatExecutionTime(false, secs) + "\n");
		cout << info("\nWatermark detection average execution time per frame: " + Utils::formatExecutionTime(showFps, secs / framesCount) + "\n");
	}
	return EXIT_SUCCESS;
}