#include "buffer.hpp"
#include "WatermarkBase.hpp"
#include "WatermarkFactory.hpp"
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#if defined(_USE_OPENCL_)
#include "WatermarkOCL.hpp"
#elif defined(_USE_CUDA_)
#include "WatermarkCuda.cuh"
#elif defined(_USE_EIGEN_)
#include "WatermarkEigen.hpp"
#endif

using std::string;

/*!
 *  \brief  Factory implementation for creating backend-specific WatermarkBase instances
 *  \author Dimitris Karatzas
 */

std::unique_ptr<WatermarkBase> InternalUtils::createWatermarkObject(
    const unsigned int height, const unsigned int width, const string& watermarkPassword, const int p, const float psnr, std::unique_ptr<WatermarkBase> previous) {
    if (p != 3 && p != 5 && p != 7 && p != 9)
        throw std::invalid_argument("Unsupported value for p. Allowed p values: 3, 5, 7, 9");
    if (height < static_cast<unsigned int>(p) || width < static_cast<unsigned int>(p))
        throw std::invalid_argument("Image dimensions must each be at least p pixels");
    checkImageDimensions(height, width);
    const int rows = static_cast<int>(height);
    const int cols = static_cast<int>(width);
    WatermarkBuffer watermark = [&] {
        if (previous && previous->hasSize(rows, cols))
            return previous->releaseWatermark();
        previous.reset();
        return WatermarkBase::generateWatermark(watermarkPassword, rows, cols);
    }();
    previous.reset();
#if defined(_USE_OPENCL_)
    switch (p) {
    case 3: return std::make_unique<WatermarkOCL<3>>(rows, cols, std::move(watermark), psnr); break;
    case 5: return std::make_unique<WatermarkOCL<5>>(rows, cols, std::move(watermark), psnr); break;
    case 7: return std::make_unique<WatermarkOCL<7>>(rows, cols, std::move(watermark), psnr); break;
    case 9: return std::make_unique<WatermarkOCL<9>>(rows, cols, std::move(watermark), psnr); break;
#elif defined(_USE_CUDA_)
    switch (p) {
    case 3: return std::make_unique<WatermarkCuda<3>>(rows, cols, std::move(watermark), psnr); break;
    case 5: return std::make_unique<WatermarkCuda<5>>(rows, cols, std::move(watermark), psnr); break;
    case 7: return std::make_unique<WatermarkCuda<7>>(rows, cols, std::move(watermark), psnr); break;
    case 9: return std::make_unique<WatermarkCuda<9>>(rows, cols, std::move(watermark), psnr); break;
#elif defined(_USE_EIGEN_)
    switch (p) {
    case 3: return std::make_unique<WatermarkEigen<3>>(rows, cols, std::move(watermark), psnr); break;
    case 5: return std::make_unique<WatermarkEigen<5>>(rows, cols, std::move(watermark), psnr); break;
    case 7: return std::make_unique<WatermarkEigen<7>>(rows, cols, std::move(watermark), psnr); break;
    case 9: return std::make_unique<WatermarkEigen<9>>(rows, cols, std::move(watermark), psnr); break;
#endif
    default: throw std::invalid_argument("Unsupported value for p. Allowed p values: 3, 5, 7, 9");
    }
}
