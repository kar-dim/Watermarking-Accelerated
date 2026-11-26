#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

/*!
 *  \brief  CUDA runtime error checking helper and Scoped Device management
 *  \author Dimitris Karatzas
 */
namespace cuda_utils {
class ScopedDevice {
  public:
    explicit ScopedDevice(int device) noexcept {
        if (device < 0)
            return;
        status = cudaGetDevice(&previous);
        if (status == cudaSuccess && previous != device) {
            status = cudaSetDevice(device);
            changed = status == cudaSuccess;
        }
    }
    ~ScopedDevice() {
        if (changed)
            cudaSetDevice(previous);
    }
    ScopedDevice(const ScopedDevice&) = delete;
    ScopedDevice& operator=(const ScopedDevice&) = delete;
    cudaError_t status = cudaSuccess;

  private:
    int previous = -1;
    bool changed = false;
};

class CudaError : public std::runtime_error {
  public:
    CudaError(const cudaError_t code, const std::string& message) : std::runtime_error(message), code(code) {}
    const cudaError_t code;
};
inline void cudaCheck(const cudaError_t error, const char* expression, const char* file, const int line) {
    if (error != cudaSuccess)
        throw CudaError(error, std::string("CUDA error: ") + cudaGetErrorString(error) + " [" + expression + " @ " + file + ":" + std::to_string(line) + "]");
}
} // namespace cuda_utils

#define CUDA_CHECK(expression) ::cuda_utils::cudaCheck((expression), #expression, __FILE__, __LINE__)
