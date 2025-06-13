#include <cstdlib>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <random>
#include <string>
#include <thread>
#include <vector>

/*!
 *  \brief  This is a helper project for my Thesis with title:
 *			EFFICIENT IMPLEMENTATION OF WATERMARKING ALGORITHMS AND
 *			WATERMARK DETECTION IN IMAGE AND VIDEO USING GPU.
 *  \author Dimitris Karatzas
 */
int main(int argc, char* argv[])
{
    if (argc != 5)
    {
        std::cerr << "Usage: " << argv[0] << " <rows> <cols> <seed> <output_file>\n";
        return EXIT_FAILURE;
    }
	constexpr int maxSize = 65536;
    //parse arguments
    const int rows = std::stoi(argv[1]);
    const int cols = std::stoi(argv[2]);
    const unsigned int seed = std::stoul(argv[3]);
    const std::string filename = argv[4];
    if (rows <= 0 || cols <= 0 || rows > maxSize || cols > maxSize)
    {
        std::cerr << "Rows and columns must be positive integers less than or equal to " << maxSize <<".\n";
        return EXIT_FAILURE;
    }
    const int numElements = rows * cols;
    omp_set_num_threads(static_cast<int>(std::thread::hardware_concurrency()));

    //generate random numbers in parallel
    std::vector<float> randomNums(numElements);
#pragma omp parallel
    {
        const int threadId = omp_get_thread_num();
        const int numThreads = omp_get_num_threads();
        std::mt19937 generator(seed);
        //watermark is a Gaussian distribution with mean 0 and standard deviation 1
        std::normal_distribution<float> distribution(0.0f, 1.0f);

        //compute range for each thread
        const int threadElements = numElements / numThreads;
        const int start = threadId * threadElements;
        const int end = (threadId == numThreads - 1) ? numElements : start + threadElements;

        //generate random numbers for this thread
        for (int i = start; i < end; i++)
            randomNums[i] = distribution(generator);
    }

    //write the random numbers to the output file
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