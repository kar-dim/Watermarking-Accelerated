#include "buffer.hpp"
#include "WatermarkBase.hpp"
#include "WatermarkCrypto.hpp"
#include <array>
#include <cstdint>
#include <string>
#include <vector>

ImageBuffer WatermarkBase::generateRandomMatrix(const std::string& watermarkPassword, WatermarkLoader loader) const {
    const int64_t numElements = static_cast<int64_t>(baseRows) * baseCols;
    const int64_t numBlocks = numElements / 8;
    const int64_t remainder = numElements % 8;
    // precompute the base ChaCha20 state bytes from the given password
    const std::array<uint32_t, 16> baseState = WatermarkCrypto::computeBaseState(watermarkPassword);

    std::vector<float> randomNums(numElements);
    // main loop for generating randoms, process blocks of 8 because ChaCha20 outputs 8 64-bit values
#pragma omp parallel for schedule(static)
    for (int64_t block = 0; block < numBlocks; block++) {
        // generate random bytes
        const std::array<uint64_t, 8> randomBits = WatermarkCrypto::chacha20Block(baseState, static_cast<uint64_t>(block));
        const int64_t baseIdx = block * 8;
        for (int j = 0; j < 4; j++) {
            // generate random normal distributed values
            const auto [z0, z1] = WatermarkCrypto::generateBoxMullerNormalPair(randomBits[j * 2], randomBits[j * 2 + 1]);
            randomNums[baseIdx + j * 2] = z0;
            randomNums[baseIdx + j * 2 + 1] = z1;
        }
    }
    // tail handling (last block)
    if (remainder > 0) {
        // generate random bytes
        const std::array<uint64_t, 8> randomBits = WatermarkCrypto::chacha20Block(baseState, static_cast<uint64_t>(numBlocks));
        const int64_t baseIdx = numBlocks * 8;
        for (int j = 0; j < 4; j++) {
            const int64_t idx = baseIdx + j * 2;
            if (idx >= numElements)
                break;
            // generate random normal distributed values
            const auto [z0, z1] = WatermarkCrypto::generateBoxMullerNormalPair(randomBits[j * 2], randomBits[j * 2 + 1]);
            randomNums[idx] = z0;
            if (idx + 1 < numElements)
                randomNums[idx + 1] = z1;
        }
    }
    // load the random values in the corresponding backend buffer (GPU array, Eigen Array etc)
    return loader(randomNums, baseRows, baseCols);
}