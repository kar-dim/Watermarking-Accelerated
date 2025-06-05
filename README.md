# ICSD thesis Part 2 / GPU Watermarking

![512](https://github.com/user-attachments/assets/6544f178-4f99-43ff-850c-9f40db478f35)


Code for my Diploma thesis at Information and Communication Systems Engineering (University of the Aegean, School of Engineering) with title "Efficient implementation of watermark and watermark detection algorithms for image and video using the graphics processing unit" [Link](https://hellanicus.lib.aegean.gr/handle/11610/19672). 
The original watermarking algorithms are described in this paper: [Link](https://www.icsd.aegean.gr/publication_files/637538981.pdf)

**NOTE**: This repository features a refactored and optimized version of the original implementation, with improved algorithms and execution times.
The deprecated original Thesis code is in the <a href="https://github.com/kar-dim/Watermarking-GPU/tree/old">old</a> branch. The optimized code is in the default <a href="https://github.com/kar-dim/Watermarking-GPU/tree/master">master</a> branch. There is a newer implementation with the CUDA framework, which works only on NVIDIA GPUs, and is slightly faster on NVIDIA GPUs: <a href="https://github.com/kar-dim/Watermarking-GPU/tree/cuda">CUDA branch</a>

# Overview

The aim of this project is to compare the performance (primarily execution speed) of watermarking algorithms when implemented on CPU versus GPU. This repository focuses on the GPU implementation (Part 2). The CPU implementation can be found in the corresponding CPU repository (Part 1 / CPU usage [here](https://github.com/kar-dim/Watermarking-CPU) ).

# Key Features

- Implementation of watermark embedding and detection algorithms for images and video.
- Comparative performance analysis between CPU and GPU implementations.

# Run the pre-built binaries

- Get the latest binaries [here](https://github.com/kar-dim/Watermarking-GPU/releases) for OpenCL or CUDA platform. The binary contains the sample application and the embedded CUDA/OpenCL kernels. Before we can emded the watermark, we have to create it first.
- The watermark generation is based on Normal-distributed random values with zero mean and standard deviation of one. The ```CommonRandomMatrix``` produces pseudo-random values. A bat file is included to generate the watermarks, with sizes exactly the same as the provided sample images. Of course, one can generate a random watermark for any desired image size like this:  
```CommonRandomMatrix.exe [rows] [cols] [seed] [fileName]```  then pass the provided watermark file path in the sample project configuration.

The sample application:
   - Embeds the watermark using the NVF and the proposed Prediction-Error mask for a video or image.
   - Detects the watermark using the proposed Prediction-Error based detector for a video or image.
   - Prints FPS for both operations, and both masks (image only mode, for video ME masking is used only).
Needs to be parameterized from the corresponding ```settings.ini``` file. Here is a detailed explanation for each parameter:

| Parameter                         | Description                                                                                                                 |
|-----------------------------------|-----------------------------------------------------------------------------------------------------------------------------               |
| image                             | Path to the input image.                                                                                                                   |
| watermark                         | Path to the Random Matrix (watermark). This is produced by the CommonRandomMatrix project. Watermark and Image sizes should match exactly. |
| save_watermarked_files_to_disk    | \[true/false\]: Set to true to save the watermarked NVF and Prediction-Error files to disk.                                                |
| execution_time_in_fps             | \[true/false\]: Set to true to display execution times in FPS. Else, it will display execution time in seconds.                            |
| p                                 | Window size for masking algorithms. Currently only ```p=3``` is allowed.                                                                   |
| psnr                              | PSNR (Peak Signal-to-Noise Ratio). Higher values correspond to less watermark in the image, reducing noise, but making detection harder.   |
| loops_for_test                    | Loops the algorithms many times, simulating more work. A value of 1000 produces almost identical execution times.                          |
| opencl_device                     | [Number]: Works only for OpenCL binary. If multiple OpenCL devices are found, then set this to the desired device. Set it to 0 if one device is found. |

**Video-only settings:**


| Parameter                         | Description                                                                                                                 |
|-----------------------------------|-----------------------------------------------------------------------------------------------------------------------------                |
| video                             | Path to the video file, if we want to embed the watermark for a video, or try to detect its watermark.                                      |
| watermark_interval                | [Number]: Embed/Try to detect the watermark every X frames. If set to 1 then the watermark will be embedded for each frame, which degrades video quality.|
| encode_watermark_file_path        | Set this value to a file path, in order to embed watermark and save the watermarked file to disk.                                           |
| encode_options                    | These are ffmpeg options for encoding. Example: ```-c:v libx265 -preset fast -crf 23```  will pass these encoding options to ffmpeg.
| watermark_detection               | \[true/false\]: Set to true to try to detect the watermark of the "video" parameter. The detection occurs after "watermark_interval" frames.|


# Libraries Used

- [ArrayFire](https://arrayfire.org): A C++ library for fast GPU computing.
- [FFMpeg](https://www.ffmpeg.org/): A complete, cross-platform solution to record, convert and stream audio and video.
- [inih](https://github.com/jtilly/inih): A lightweight C++ library for parsing .ini configuration files.

# Additional Dependencies For Building

- OpenCL implementation: The [OpenCL Headers](https://github.com/KhronosGroup/OpenCL-Headers), [OpenCL C++ Bindings](https://github.com/KhronosGroup/OpenCL-CLHPP) and [OpenCL Library file](https://github.com/KhronosGroup/OpenCL-SDK) are already included and configured for this project.
- CUDA implementation: NVIDIA CUDA Toolkit.
- ArrayFire should be installed globally, with default installation options. Environment Variable "AF_PATH" will be defined automatically.
- FFMpeg must exist on system PATH (Pre-build binaries already include FFMpeg binaries and DLLs).
