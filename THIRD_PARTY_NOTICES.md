# Third-party notices

Watermarking uses third-party code and runtime libraries. Each component retains
its own copyright and license. The application uses GNU GPL version 3, whose
full text is in `LICENSE`. The GPL permits redistribution and modification
under its conditions and provides no warranty.

The component license collection is indexed in [LICENSES/README.md](LICENSES/README.md).
This collection is incomplete. It does not supply complete corresponding source
or establish that the CUDA combination is GPL-compatible.

## FFmpeg

This software uses FFmpeg libraries licensed under GNU GPL version 3 or later,
including libx264 and libx265. The GPL text is in
`LICENSES/Common/FFMPEG_LICENSE.txt`. Build version, configuration and DLL hashes
are listed in `LICENSES/Common/FFMPEG_BUILD.txt`, vendor documentation is in
`LICENSES/Common/FFMPEG_VENDOR_README.txt`.
FFmpeg and its included libraries retain their original contributor copyrights.
All applicable component notices and exact corresponding source remain required.

## Qt

The GUI uses dynamically linked Qt 6.12.0 libraries and plugins under their
LGPL version 3 option. LGPL and GPL texts are in `LICENSES/Qt/` and `LICENSE`.
Qt is copyright The Qt Company Ltd. and other contributors. Reference component
copyrights and custom license texts are in `QT_REFERENCE_COMPONENT_NOTICES.txt`.
Qt's other third-party license texts, notices, translations and matching source
remain to be completed. The installed SPDX hashes do not match the DLLs, so its
source references are not independently verified for these exact binaries.

## Images, compression and metadata

This software is based in part on the work of the Independent JPEG Group.
Standalone libjpeg-turbo, libwebp and their component notices remain to be collected.
The known libpng, libtiff and zlib-ng header notices are in `LICENSES/Common/`.
The notices do not establish all implementation components or exact binary revisions.
CImg offers CeCILL-C or CeCILL-2.1. The CeCILL-2.1 text and contributor notices
for this GPL combination remain to be collected.
TinyEXIF is copyright (c) 2015-2025 Seacave and identifies the MIT license.
Its full permission/disclaimer remains to be collected.

## Compute backends

Eigen builds include the vendored Eigen license texts in `LICENSES/Eigen/`.
Eigen is primarily MPL-2.0 with file-specific permissive terms.
The exact LLVM OpenMP runtime license and notices still need collection.

OpenCL builds include Apache-2.0 and the current vendored Khronos header
copyrights in `LICENSES/OpenCL/`. The system/driver OpenCL implementation is
supplied separately by the user's driver provider.

CUDA builds include the NVIDIA CUDA 13.4 EULA and installed toolkit terms in
`LICENSES/CUDA/`. NVIDIA proprietary components retain their own terms and are
not relicensed under GPL. Redistribution permission for listed CUDA components
does not establish compatibility with third-party GPL code. Applicable CUB/CCCL
license texts and notices remain to be completed.

## Font, tools and Microsoft runtime

Sofia Sans Semi Condensed is copyright 2019 The Sofia Sans Project Authors and
is licensed under SIL OFL 1.1. The GUI includes the full notice in
`licenses/SofiaSans/OFL.txt`.

Microsoft CRT and OpenMP runtimes retain Microsoft's redistribution terms.
Applicable runtime notices/terms remain to be packaged.
GoogleTest and the separately vendored clang-format executable retain their
own licenses. Their exact notices remain required when those files are distributed.

## Source availability

Application source is available at
[Watermarking-Accelerated](https://github.com/kar-dim/Watermarking-Accelerated).
This does not supply the complete matching dependency source for the binaries.
A release-specific source package, with patches and build information, must be
provided with clear download directions before binary publication. An upstream
homepage or generic source link alone does not meet that requirement.
