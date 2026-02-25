#pragma once

#include "buffer.hpp"
#include "include/WatermarkTypes.hpp"
#include <cmath>
#include <cstdint>
#include <functional>
#include <random>
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
    WatermarkBase(const unsigned int rows, const unsigned int cols, const uint32_t watermarkSeed, const float psnr, WatermarkLoader loader)
        : baseRows(rows), baseCols(cols), totalPixels(baseRows * baseCols), randomMatrix(generateRandomMatrix(watermarkSeed, loader)), strengthFactor(computeStrengthFactor(psnr)) {}

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
    ImageBuffer generateRandomMatrix(const uint32_t watermarkSeed, WatermarkLoader loader) const {
        constexpr int numPartitions = 64;
        const size_t numElements = static_cast<size_t>(baseRows) * baseCols;

        std::vector<float> randomNums(numElements);
        std::mt19937 masterGenerator(watermarkSeed);
        std::vector<unsigned int> partitionSeeds(numPartitions);

        // we have a deterministic starting seed for each thread
        for (int i = 0; i < numPartitions; i++)
            partitionSeeds[i] = masterGenerator();
        
        // generation in parallel
#pragma omp parallel for schedule(static)
        for (int p = 0; p < numPartitions; p++) {
            std::mt19937 localGenerator(partitionSeeds[p]);
            std::normal_distribution<float> distribution(0.0f, 1.0f);
            const auto start = p * numElements / numPartitions;
            const auto end = (p + 1) * numElements / numPartitions;
            for (auto i = start; i < end; i++)
                randomNums[i] = distribution(localGenerator);
        }
        return loader(randomNums, baseRows, baseCols);
    }
};