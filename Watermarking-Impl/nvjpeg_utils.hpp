#pragma once
#include "CudaArray.hpp"
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <optional>
#include <vector>

/*!
 *  \brief  GPU JPEG decoding and encoding (nvJPEG) for the CUDA builds. The codecs are created per device on first use.
 *			The hardware decoder/encoder when the device has one, else the GPU (GPU assisted Huffman) backends.
 *  \author Dimitris Karatzas
 */
namespace nvjpeg_utils {
// decodes a JPEG file (1 or 3 components) into row-major planar 8-bit planes (the layout of a CImg image)
std::optional<CudaArray<uint8_t>> decode(const uint8_t* jpeg, size_t length, cudaStream_t stream);
// encodes row-major planar 8-bit planes (1 or 3 channels) with CImg's save_jpeg settings (quality 100, 4:2:0), empty on failure
std::vector<uint8_t> encode(const CudaArray<uint8_t>& planes, cudaStream_t stream);
} // namespace nvjpeg_utils
