#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

/*!
 *  \brief  CUDA runtime error checking helper
 *  \author Dimitris Karatzas
 */
namespace cuda_utils {
inline void cudaCheck(const cudaError_t error, const char* expression, const char* file, const int line) {
    if (error != cudaSuccess)
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(error) + " [" + expression + " @ " + file + ":" + std::to_string(line) + "]");
}
} // namespace cuda_utils

#define CUDA_CHECK(expression) ::cuda_utils::cudaCheck((expression), #expression, __FILE__, __LINE__)
