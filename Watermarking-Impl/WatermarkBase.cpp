#include "buffer.hpp"
#include "half_float.hpp"
#include "simd.hpp"
#include "WatermarkBase.hpp"
#include "WatermarkCrypto.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <string>

WatermarkBuffer WatermarkBase::generateRandomMatrix(const std::string& watermarkPassword, WatermarkLoader loader) const {
    const int64_t numElements = static_cast<int64_t>(baseRows) * baseCols;
    // one ChaCha20 block (8 uint64) = 4 Box-Muller pairs = 8 values, the last block may be partial
    const int64_t fullBlocks = numElements / 8;
    const int64_t numBlocks = (numElements + 7) / 8;
    // precompute the base ChaCha20 state bytes from the given password
    const std::array<uint32_t, 16> baseState = WatermarkCrypto::computeBaseState(watermarkPassword);

    // the watermark is rounded to half precision: the GPU backends store it as half (half the memory traffic of the embedding and detection
    // kernels that read it) and every backend uses the same values
    const auto halfBits = std::make_unique_for_overwrite<uint16_t[]>(numElements);
    // stores the (up to) 8 values of a block, full blocks with the F16C conversion (every AVX2 CPU has it)
    const auto storeBlock = [&](const int64_t block, const float* values) {
        const int64_t first = block * 8;
        const int64_t count = std::min<int64_t>(8, numElements - first);
        if (count == 8) {
            _mm_storeu_si128(reinterpret_cast<__m128i*>(halfBits.get() + first), _mm256_cvtps_ph(_mm256_loadu_ps(values), _MM_FROUND_TO_NEAREST_INT));
            return;
        }
        for (int64_t i = 0; i < count; i++)
            halfBits[first + i] = HalfFloat::fromFloat(values[i]);
    };

    // full blocks in groups: 16 (AVX-512) or 8 ChaCha20 blocks at once, then the vectorized Box-Muller transform (4 or 2 blocks per call)
#if defined(__AVX512F__)
    constexpr int groupBlocks = 16;
#else
    constexpr int groupBlocks = 8;
#endif
    const int64_t numGroups = fullBlocks / groupBlocks;
#pragma omp parallel for schedule(static)
    for (int64_t group = 0; group < numGroups; group++) {
        const int64_t firstBlock = group * groupBlocks;
        std::array<float, groupBlocks * 8> values;
#if defined(__AVX512F__)
        const auto blocks = WatermarkCrypto::chacha20Blocks16(baseState, static_cast<uint64_t>(firstBlock));
        for (int quad = 0; quad < 4; quad++)
            WatermarkCrypto::generateBoxMullerNormalBlockQuad(blocks.data() + (quad * 4), values.data() + (quad * 32));
#else
        const auto blocks = WatermarkCrypto::chacha20Blocks8(baseState, static_cast<uint64_t>(firstBlock));
        for (int pair = 0; pair < 4; pair++)
            WatermarkCrypto::generateBoxMullerNormalBlockPair(blocks[pair * 2], blocks[(pair * 2) + 1], values.data() + (pair * 16));
#endif
        for (int b = 0; b < groupBlocks; b++)
            storeBlock(firstBlock + b, values.data() + (b * 8));
    }
    // the remaining full block pairs still use the vectorized Box-Muller transform: its results differ from the scalar std functions in
    // the last bits, so every full block pair must take the same path whatever the image size
    int64_t firstScalarBlock = numGroups * groupBlocks;
    for (; firstScalarBlock + 1 < fullBlocks; firstScalarBlock += 2) {
        std::array<float, 16> values;
        WatermarkCrypto::generateBoxMullerNormalBlockPair(
            WatermarkCrypto::chacha20Block(baseState, static_cast<uint64_t>(firstScalarBlock)), WatermarkCrypto::chacha20Block(baseState, static_cast<uint64_t>(firstScalarBlock + 1)), values.data());
        storeBlock(firstScalarBlock, values.data());
        storeBlock(firstScalarBlock + 1, values.data() + 8);
    }
    // scalar path: the odd last full block and the partial last block
#pragma omp parallel for schedule(static)
    for (int64_t block = firstScalarBlock; block < numBlocks; block++) {
        const std::array<uint64_t, 8> randomBits = WatermarkCrypto::chacha20Block(baseState, static_cast<uint64_t>(block));
        std::array<float, 8> values;
        for (int j = 0; j < 4; j++) {
            const auto [z0, z1] = WatermarkCrypto::generateBoxMullerNormalPair(randomBits[j * 2], randomBits[(j * 2) + 1]);
            values[j * 2] = z0;
            values[(j * 2) + 1] = z1;
        }
        storeBlock(block, values.data());
    }
    // load the watermark in the corresponding backend buffer (GPU array, Eigen Array etc)
    return loader(std::span<const uint16_t>(halfBits.get(), static_cast<size_t>(numElements)), baseRows, baseCols);
}
