#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

/*!
 *  \brief  This is a helper project for my Thesis with title:
 *			EFFICIENT IMPLEMENTATION OF WATERMARKING ALGORITHMS AND
 *			WATERMARK DETECTION IN IMAGE AND VIDEO USING GPU.
 *  \author Dimitris Karatzas
 */
int main(int argc, char* argv[]) {
    constexpr int numPartitions = 64;
    constexpr int maxSize = 65536;

    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <rows> <cols> <seed> <output_file>\n";
        return EXIT_FAILURE;
    }

    // parse arguments
    const int rows = std::stoi(argv[1]);
    const int cols = std::stoi(argv[2]);
    const size_t numElements = static_cast<size_t>(rows) * cols;
    const unsigned int seed = std::stoul(argv[3]);
    const std::string filename = argv[4];
    if (rows <= 0 || cols <= 0 || rows > maxSize || cols > maxSize) {
        std::cerr << "Rows and columns must be positive integers less than or equal to " << maxSize << ".\n";
        return EXIT_FAILURE;
    }

    std::mt19937 masterGenerator(seed);
    std::vector<unsigned int> partitionSeeds(numPartitions);

    // generate a unique deterministic starting seed for each thread
    for (int i = 0; i < numPartitions; i++)
        partitionSeeds[i] = masterGenerator();

    // generate random numbers in parallel
    std::vector<float> randomNums(numElements);
#pragma omp parallel for schedule(static)
    for (int p = 0; p < numPartitions; p++) {
        std::mt19937 localGenerator(partitionSeeds[p]);
        // watermark is a Gaussian distribution with mean 0 and standard deviation 1
        std::normal_distribution<float> distribution(0.0f, 1.0f);

        // compute range for each thread
        const auto start = p * numElements / numPartitions;
        const auto end = (p + 1) * numElements / numPartitions;

        // generate random numbers for this thread
        for (auto i = start; i < end; i++)
            randomNums[i] = distribution(localGenerator);
    }

    // write the random numbers to the output file
    std::ofstream output(filename, std::ios::binary);
    if (!output) {
        std::cerr << "Error: Unable to open file " << filename << " for writing.\n";
        return EXIT_FAILURE;
    }
    output.write(reinterpret_cast<const char*>(randomNums.data()), randomNums.size() * sizeof(float));
    if (!output) {
        std::cerr << "Error: Failed to write data to " << filename << ".\n";
        return EXIT_FAILURE;
    }

    std::cout << "Successfully wrote " << rows * cols << " random floats to " << filename << ".\n";
    return EXIT_SUCCESS;
}