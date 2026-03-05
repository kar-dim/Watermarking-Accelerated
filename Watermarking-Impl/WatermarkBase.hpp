#pragma once

#include "buffer.hpp"
#include "include/WatermarkTypes.hpp"
#include "WatermarkCrypto.hpp"
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

/*!
 *  \brief  Functions for watermark computation and detection, Base class.
 *  \author Dimitris Karatzas
 */
class WatermarkBase {
  protected:
    using WatermarkLoader = std::function<ImageBuffer(std::vector<float>, const unsigned int, const unsigned int)>;

    template <unsigned int ALIGNMENT>
    static constexpr unsigned int alignUp(const unsigned int x) {
        static_assert(ALIGNMENT > 0 && (ALIGNMENT & (ALIGNMENT - 1)) == 0, "ALIGNMENT must be a power of 2");
        return (x + (ALIGNMENT - 1)) & ~(ALIGNMENT - 1);
    }
    unsigned int baseRows, baseCols, totalPixels;
    ImageBuffer randomMatrix;
    float strengthFactor;

  public:
    WatermarkBase(const unsigned int rows, const unsigned int cols, const std::string& watermarkPassword, const float psnr, WatermarkLoader loader)
        : baseRows(rows), baseCols(cols), totalPixels(baseRows * baseCols), randomMatrix(generateRandomMatrix(watermarkPassword, loader)), strengthFactor(computeStrengthFactor(psnr)) {}

    // delete copy and move operations we don't wannt them
    WatermarkBase(const WatermarkBase&) = delete;
    WatermarkBase(WatermarkBase&&) = delete;
    WatermarkBase& operator=(const WatermarkBase&) = delete;
    WatermarkBase& operator=(WatermarkBase&&) = delete;

    virtual ~WatermarkBase() = default;

    // main watermark embedding method
    // it embeds the watermark computed from "inputGrayImage" (always grayscale, 2D)
    // into a new array "output" based on "inputImage" (RGB or grayscale always u8)
    virtual void makeWatermark(const ImageBuffer& inputGrayImage, const ImageBuffer& inputImage, ImageOutputBuffer& output, const MaskMethod maskType) = 0;

    // the main mask detector function
    virtual float detectWatermark(const ImageBuffer& inputImage, const MaskMethod maskType) = 0;

  private:
    static inline float computeStrengthFactor(const float psnr) { return 255.0f / std::sqrt(std::pow(10.0f, psnr / 10.0f)); }

    // helper method to generate the watermark based on the given seed, using a parallelized approach with OpenMP for very fast generation
    // it generates secure random values based on ChaCha20 and uses Box-Muller transform to conver to gaussian random
    ImageBuffer generateRandomMatrix(const std::string& watermarkPassword, WatermarkLoader loader) const {
        const int64_t numElements = static_cast<int64_t>(baseRows) * baseCols;
        std::vector<float> randomNums(numElements);
        // precompute the base ChaCha20 state bytes from the given password
        const std::array<uint32_t, 16> baseState = WatermarkCrypto::computeBaseState(watermarkPassword);
        // step by 8 because ChaCha20 generates 8 uint64_t per block
#pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < numElements; i += 8) {
            uint64_t blockCounter = static_cast<uint64_t>(i / 8);
            uint64_t randomBits[8];
            // generate 8 random numbers
            WatermarkCrypto::chacha20Block(baseState, blockCounter, randomBits);
            // process the 8 random integers into 4 Box-Muller pairs
            for (int64_t j = 0; j < 4; j++) {
                const int64_t idx = i + (j * 2);
                if (idx >= numElements)
                    break;
                // generate two floats in (0,1] and transform them based on Box-Muller (polar values)
                const auto [radius, theta] = WatermarkCrypto::generateBoxMullerPair(randomBits[j * 2], randomBits[j * 2 + 1]);
                // write values
                randomNums[idx] = radius * std::cos(theta);
                if (idx + 1 < numElements)
                    randomNums[idx + 1] = radius * std::sin(theta);
            }
        }
        // load the random values in the corresponding backend buffer (ArrayFire array, Eigen Array etc)
        return loader(randomNums, baseRows, baseCols);
    }
};