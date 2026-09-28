#include "buffer.hpp"
#include "WatermarkBase.hpp"
#include "WatermarkCrypto.hpp"
#include <array>
#include <cstdint>
#include <string>
#if defined(_USE_CUDA_)
#include "cuda_utils.hpp"
#include "CudaStreamManager.hpp"
#elif defined(_USE_OPENCL_)
#include "OclQueueManager.hpp"
#include "opencl_utils.hpp"
#elif defined(_USE_EIGEN_)
#include "half_float.hpp"
#include <Eigen/Core>
#endif

WatermarkBuffer WatermarkBase::generateWatermark(const std::string& watermarkPassword, const int rows, const int cols) {
    const int64_t numElements = static_cast<int64_t>(rows) * cols;
    // the ChaCha20 start state (the SHA-256 key of the password), the block counter is set per block
    const std::array<uint32_t, 16> baseState = WatermarkCrypto::computeBaseState(watermarkPassword);
#if defined(_USE_CUDA_)
    // on the compute stream: the embedding and detection kernels of the watermark object run after it
    const cudaStream_t stream = CudaStreamManager::getInstance().getComputeStream();
    WatermarkBuffer watermark(rows, cols, stream);
    cuda_utils::launchGenerateWatermarkKernel(baseState, watermark.data(), numElements, stream);
    return watermark;
#elif defined(_USE_OPENCL_)
    // on the queue of the watermark object: its embedding and detection kernels run after it
    auto& queueManager = OclQueueManager::getInstance();
    WatermarkBuffer watermark(rows, cols, queueManager.getQueueRaw());
    cl_utils::launchGenerateWatermarkKernel(baseState, watermark.clBuffer(), numElements, queueManager.getQueue());
    return watermark;
#elif defined(_USE_EIGEN_)
    const auto halfBits = WatermarkCrypto::generateHalfWatermark(baseState, numElements);
    Eigen::ArrayXXf watermark(rows, cols);
    float* values = watermark.data();
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < numElements; i++)
        values[i] = HalfFloat::toFloat(halfBits[i]);
    return WatermarkBuffer(std::move(watermark));
#endif
}
