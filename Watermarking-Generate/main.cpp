#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <thread>
#include <omp.h>

/*!
 *  \brief  This is a helper project for my Thesis with title:
 *			EFFICIENT IMPLEMENTATION OF WATERMARKING ALGORITHMS AND
 *			WATERMARK DETECTION IN IMAGE AND VIDEO USING GPU, CUDA version
 *          It generates a random matrix of floats and writes it to a file.
 *  \author Dimitris Karatzas
 */
int main(int argc, char* argv[])
{
    if (argc != 5)
    {
        std::cerr << "Usage: " << argv[0] << " <rows> <cols> <seed> <output_file>\n";
        return EXIT_FAILURE;
    }
    // Parse arguments
    const int rows = std::stoi(argv[1]);
    const int cols = std::stoi(argv[2]);
    const unsigned int seed = std::stoul(argv[3]);
    const std::string filename = argv[4];
    if (rows <= 0 || cols <= 0 || rows >= 32768 || cols >= 32768)
    {
        std::cerr << "Rows and columns must be positive integers less than or equal to 32768.\n";
        return EXIT_FAILURE;
    }
    const int numElements = rows * cols;
    omp_set_num_threads(static_cast<int>(std::thread::hardware_concurrency()));

    // Generate random numbers in parallel
    std::vector<float> randomNums(numElements);
#pragma omp parallel
    {
        const int threadId = omp_get_thread_num();
        const int numThreads = omp_get_num_threads();
        std::mt19937 generator(seed);
        //watermark is a Gaussian distribution with mean 0 and standard deviation 1
        std::normal_distribution<float> distribution(0.0f, 1.0f);

        // Compute range for each thread
        const int threadElements = numElements / numThreads;
        const int start = threadId * threadElements;
        const int end = (threadId == numThreads - 1) ? numElements : start + threadElements;

        // Generate random numbers for this thread
        for (int i = start; i < end; i++)
            randomNums[i] = distribution(generator);
    }

    // Write the random numbers to the output file
    std::ofstream output(filename, std::ios::binary);
    if (!output)
    {
        std::cerr << "Error: Unable to open file " << filename << " for writing.\n";
        return EXIT_FAILURE;
    }
    output.write(reinterpret_cast<const char*>(randomNums.data()), randomNums.size() * sizeof(float));
    if (!output)
    {
        std::cerr << "Error: Failed to write data to " << filename << ".\n";
        return EXIT_FAILURE;
    }

    std::cout << "Successfully wrote " << rows * cols << " random floats to " << filename << ".\n";
    return EXIT_SUCCESS;
}