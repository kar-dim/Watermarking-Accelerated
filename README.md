# ICSD thesis / Efficient Image and Video Watermarking

![512](https://github.com/user-attachments/assets/6544f178-4f99-43ff-850c-9f40db478f35)


Code for my Diploma thesis at Information and Communication Systems Engineering (University of the Aegean, School of Engineering) with title "Efficient implementation of watermark and watermark detection algorithms for image and video using the graphics processing unit" [Link](https://hellanicus.lib.aegean.gr/handle/11610/19672). 
The original watermarking algorithms are described in this paper: [Link](https://www.icsd.aegean.gr/publication_files/637538981.pdf)

**NOTE**: This repository features a refactored and optimized version of the original implementation, with improved algorithms and execution times.
The deprecated original Thesis code is in the archived repository <a href="https://github.com/kar-dim/Watermarking-GPU/tree/old">old</a> branch. The optimized code for all implementations (CUDA, OpenCL and CPU Eigen) is in this repository  <a href="https://github.com/kar-dim/Watermarking-Accelerated">master</a> branch. The original Thesis code included only OpenCL and Eigen code, while here we also include a CUDA implementation.

# Overview

The aim of this project is to compare the performance (primarily execution speed) of watermarking algorithms when implemented on CPU versus GPU. This repository includes all the relevant implementations.

# Key Features

- Implementation of watermark embedding and detection algorithms for images and video.
- Comparative performance analysis between CPU and GPU implementations.

# Run the pre-built binaries

- Get the latest binaries [here](https://github.com/kar-dim/Watermarking-Accelerated/releases) for Eigen, OpenCL or CUDA platform. The binaries contain the sample application and the embedded CUDA/OpenCL/Eigen implementations of the watermarking algorithms.
- The watermark generation is based on Normal-distributed random values with zero mean and standard deviation of one. 
- The pre-built binaries come with a bundled archive named ```Watermarking-Generate_and_samples```, which includes:
    - Sample video and audio files.
    - Pre-generated watermark data (A bat file is included which generates the watermarks, with sizes exactly the same as the provided sample images.)
    - The ```Watermarking-Generate``` binary. This produces pseudo-random values. The archive already includes the sample watermarks, but one can generate a random watermark for any desired image size like this:  
```Watermarking-Generate.exe [rows] [cols] [seed] [fileName]```  then pass the provided watermark file path in the sample project configuration.
To use these samples, simply extract the archive (ideally) to the root directory of the binary you're using. By default, the binaries are configured to load video and image samples from the ```samples``` subdirectory relative to their location. If you'd like to change this behavior, you can do so by editing the ```settings.ini``` file (explained in a later section).

The sample application:
   - Embeds the watermark using the NVF and the proposed Prediction-Error mask for a video or image.
   - Detects the watermark using the proposed Prediction-Error based detector for a video or image.
   - Prints FPS/execution time for both operations, and both masks.
**NOTE**: For video operations, only the proposed mask is used, which is more optimal.
Needs to be parameterized from the corresponding ```settings.ini``` file. Here is a detailed explanation for each parameter:

| Parameter                         | Description                                                                                                                 |
|-----------------------------------|-----------------------------------------------------------------------------------------------------------------------------               |
| image                             | Path to the input image to embed and detect watermark. This will set the sample application to ```image mode```. |
| watermark                         | Path to the Random Matrix (watermark). This is produced by the ```Watermarking-Generate``` project. Watermark and Image sizes should match exactly. |
| save_watermarked_files_to_disk    | ```[true/false]```: Set to true to save the watermarked NVF and Prediction-Error files to disk.                                                |
| execution_time_in_fps             | ```[true/false]```: Set to true to display execution times in FPS. Else, it will display execution time in seconds.                            |
| p                                 | Window size for masking algorithms. Currently only ```p=3``` is allowed for ```OpenCL``` and ```CUDA``` implementations. ```Eigen``` implementation supports values of ```p=3,5,7``` and ```9```. |
| psnr                              | PSNR (Peak Signal-to-Noise Ratio). Higher values correspond to less watermark in the image, reducing noise, but making detection harder.   |
| loops_for_test                    | Loops the algorithms many times, simulating more work. A value of ```100~1000``` produces consistent execution times.                          |
| opencl_device                     | ```[OpenCL only / Number]```: Works only for OpenCL binary. If multiple OpenCL devices are found, then set this to the desired device. Set it to 0 if one device is found. |
| threads                           | ```[CPU only / Number]```: Maximum number of threads. Set to 0 to automatically find the maximum concurrent threads supported, or set them manually here.    |

**Video-only settings:**


| Parameter                         | Description                                                                                                                 |
|-----------------------------------|-----------------------------------------------------------------------------------------------------------------------------                |
| video                             | Path to the video file, if we want to embed or detect the watermark for a video. This will set the sample application to ```video mode``` and will read the video-only settings that are described in this section. |
| watermark_interval                | ```[Number]```: Embed or try to detect the watermark every X frames. If set to 1 when embedding, the watermark will be embedded for all frames, which degrades video quality.|
| encode_watermark_file_path        | Set this value to a file path, in order to embed watermark and save the watermarked file to disk.                                           |
| encode_options                    | These are FFmpeg options for encoding. Example: ```-c:v libx265 -preset fast -crf 23```  will pass these encoding options to FFmpeg.|
| watermark_detection               | ```[true/false]```: Set to true to try to detect the watermark of the "video" parameter. The detection occurs after ```watermark_interval``` frames. It is ignored when ```encode_watermark_file_path``` is set. |

# FFmpeg Command Used for Video Encoding

The following FFmpeg command is used to encode a new video while preserving the original input's metadata, subtitles, and audio tracks. It reads raw video frames from standard input (stdin) and copies audio/subtitles from the original input file as is. You can customize encoding settings (codec, CRF, etc) via the ```encode_options``` option as described above.
```
ffmpeg -y -f rawvideo -pix_fmt yuv420p -s <width>x<height>
  -r <frame_rate>
  -i -
  -i <input_video_file>
  <ffmpegOptions>
  -c:s copy -c:a copy
  -map 1:s? -map 0:v -map 1:a?
  -max_interleave_delta 0
  <output_file>
```

### Explanation:
- `-f rawvideo -pix_fmt yuv420p`: Specifies raw pixel format for input.
- `-s <width>x<height>`: Specifies frame size (extracted from the input).
- `-r <frame_rate>`: Frame rate of the video (extracted from the input).
- `-i -`: Accepts raw video from stdin.
- `-i <input_video_file>`: **USER SUPPLIED**: Original input file.
- `<ffmpegOptions>`: **USER SUPPLIED**: Encoding options (e.g., ```-c:v libx265 -preset fast -crf 23```).
- `-c:s copy -c:a copy`: Copies subtitle and audio streams without re-encoding.
- `-map 1:s? -map 0:v -map 1:a?`: Maps subtitles/audio from the original input, and video from stdin.
- `-max_interleave_delta 0`: Reduces potential interleaving delay issues.
- `<output_file>`: **USER SUPPLIED**: Output file path for the final video.

**NOTE:** Only Constant Frame Rate (CFR) works as expected for an input video. If the input video is Variable Frame Rate (VFR) there may be issues with audio/subtitles sync on the output file.


# How to Build

This project is built using **Visual Studio** and consists of a **solution with two projects**.

### Solution Configurations

The solution provides multiple build configurations, each targeting a specific backend:

| Configuration    | Backend     | Notes                                       |
|------------------|-------------|---------------------------------------------|
| `OPENCL_Release` | OpenCL      | Most recommended backend with very high performance. There is **no debug version** due to some known issues |
| `CUDA_Release`   | CUDA        | Recommended for systems with NVIDIA GPUs. Slightly faster than OpenCL backend    |
| `CUDA_Debug`     | CUDA        | Use for debugging CUDA-specific code        |
| `EIGEN_Release`  | Eigen       | Optimized CPU-based implementation (clang-cl toolset is used for maximum performance).         |
| `EIGEN_Debug`    | Eigen       | Use for debugging CPU implementation (clang-cl)      |

### Build Instructions

1. Open the `.sln` file in **Visual Studio 2022** (or a compatible version).
2. In the **Solution Configurations** dropdown (top toolbar), select your configuration (e.g. `CUDA_Release`).
3. Build the solution via **Build > Build Solution**.

**Note:** Both CUDA and OpenCL backends depend on **ArrayFire**, which in turn requires its own set of runtime dependencies.
If ArrayFire is properly installed, its `lib` directory (containing all required DLLs) is typically added to the system `PATH`, and everything should work out of the box.
However, since not all systems have ArrayFire installed, we include the necessary DLLs in the prebuilt binaries. These files are copied directly from `$(AF_PATH)/lib` for convenience (Post-Build event).
The same applies for CPU backend, where we copy the relevant libraries required by CImg (LibJPEG, LibPNG, zlib) and clang's OpenMP.
All backends require FFmpeg which is also copied (most libav* DLLs).


| Backend | Dependencies |
|---------|--------------|
| **CUDA**   | FreeImage.dll<br>afcuda.dll |
| **OpenCL** | FreeImage.dll<br>afopencl.dll<br>forge.dll<br>glfw3.dll<br>libiomp5md.dll<br>mkl_core.2.dll<br>mkl_def.2.dll<br>mkl_intel_thread.2.dll<br>mkl_rt.2.dll<br>mkl_tbb_thread.2.dll |
| **Eigen**  | zlib1.dll<br>libpng16.dll<br>jpeg62.dll<br>libomp.dll |


# Libraries Used

- [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page): A C++ template library for linear algebra.
- [ArrayFire](https://arrayfire.org): A C++ library for fast GPU computing.
- [FFmpeg](https://www.ffmpeg.org/): A complete, cross-platform solution to record, convert and stream audio and video.
- [CImg](https://cimg.eu/): A C++ library for image processing.
- [inih](https://github.com/jtilly/inih): A lightweight C++ library for parsing .ini configuration files.

# Additional Dependencies For Building

- OpenCL implementation: The [OpenCL Headers](https://github.com/KhronosGroup/OpenCL-Headers), [OpenCL C++ Bindings](https://github.com/KhronosGroup/OpenCL-CLHPP) and [OpenCL Library file](https://github.com/KhronosGroup/OpenCL-SDK) are already included and configured for this project.
- CUDA implementation: NVIDIA CUDA Toolkit is required for building.
- CPU Implementation: LibPNG and LibJPEG and zlib are also included, and are used internally by CImg for loading and saving of images.
- ArrayFire should be installed globally, with default installation options. Environment Variable "AF_PATH" will be defined automatically.
- FFmpeg must exist on system PATH (Pre-build binaries already include FFmpeg binaries and DLLs).
