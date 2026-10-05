# Third-party license inventory

This directory contains license texts, copyright notices and build information
for Watermarking's third-party dependencies. Components retain their own licenses.
See [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) for redistribution requirements.

## Component licenses and notices

| Component | Files | Description |
| --- | --- | --- |
| FFmpeg, including libx264 and libx265 | [GPL text](Common/FFMPEG_LICENSE.txt), [build information](Common/FFMPEG_BUILD.txt), [vendor documentation](Common/FFMPEG_VENDOR_README.txt) | The included Gyan FFmpeg build, commit `946fcce07b`, reports GPL version 3 or later. Build information includes its configuration and DLL hashes. |
| libpng | [Header notices](Common/LIBPNG_HEADER_NOTICES.txt) | Copyright and license notices from the vendored headers. |
| libtiff | [Header notices](Common/LIBTIFF_HEADER_NOTICES.txt) | Copyright and license notices from the vendored headers. |
| zlib-ng | [Header notice](Common/ZLIB_NG_HEADER_NOTICE.txt) | Copyright and license notice from the vendored headers. |
| Eigen | [License texts](Eigen/) | MPL, BSD, Apache and MINPACK texts for the vendored Eigen code. |
| OpenCL headers and C++ bindings | [Apache-2.0 text](OpenCL/OPENCL_LICENSE.txt), [header notices](OpenCL/OPENCL_HEADER_NOTICES.txt) | License and Khronos contributor copyrights. The runtime implementation is supplied by the system's GPU drivers. |
| Qt 6.12.0 | [LGPLv3 text](Qt/QT_LICENSE_LGPLv3.txt), [GPL/LGPL texts](Qt/QT_INSTALLED_LICENSES.txt) | License texts for the dynamically linked Qt libraries and plugins. |
| Qt dependencies | [Component notices](Qt/QT_REFERENCE_COMPONENT_NOTICES.txt), [SPDX metadata](Qt/SBOM/) | Reference copyrights, license expressions and source references for qtbase, qtimageformats and qtsvg. The metadata includes components outside this application's runtime bundle and does not establish exact binary/source versions. |
| CUDA 13.4 | [EULA](CUDA/CUDA_EULA.pdf), [toolkit license](CUDA/CUDA_TOOLKIT_LICENSE.txt) | NVIDIA toolkit terms and third-party notices. These terms do not establish compatibility with GPL dependencies. |
| Sofia Sans Semi Condensed | [SIL OFL 1.1](../Watermarking-UI/assets/fonts/OFL.txt) | Font copyright and license. The GUI deploys this notice as `licenses/SofiaSans/OFL.txt`. |

## Distribution

MSBuild copies this inventory, common notices and `THIRD_PARTY_NOTICES.md` for
all applications. GUI packages include Qt and font notices, each backend includes
its Eigen, OpenCL or CUDA license group.

This notice bundle is incomplete. Redistribution requires complete matching
dependency source and build information, plus applicable notices for FFmpeg's
embedded dependencies, Qt's third-party code and translations, CImg, TinyEXIF,
standalone libjpeg-turbo/libwebp, GoogleTest, LLVM OpenMP/clang-format, applicable
CUB/CCCL code and Microsoft redistributables. Exact binary revisions need to be
identified where unknown. Compatibility of the CUDA combination with GPL FFmpeg
remains unresolved.
