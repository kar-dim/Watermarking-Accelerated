#pragma once

/*!
 * \brief Shared batch path identity and device failure checks
 * \author Dimitris Karatzas
 */

#include <exception>
#include <filesystem>
#include <string>
#include <windows.h>

#if defined(_USE_OPENCL_)
#include "opencl_init.h"
#elif defined(_USE_CUDA_)
#include "CudaCheck.hpp"
#endif

namespace WatermarkCore::BatchUtils {
inline std::string utf8(const std::filesystem::path& path) {
    const auto value = path.u8string();
    return {value.begin(), value.end()};
}

inline std::filesystem::path normalized(const std::filesystem::path& path) {
    const auto absolute = std::filesystem::absolute(path).lexically_normal();
    std::error_code error;
    const auto canonical = std::filesystem::weakly_canonical(absolute, error);
    return error ? absolute : canonical.lexically_normal();
}

struct PathLess {
    bool operator()(const std::filesystem::path& a, const std::filesystem::path& b) const {
        const auto& left = a.native();
        const auto& right = b.native();
        return CompareStringOrdinal(left.c_str(), static_cast<int>(left.size()), right.c_str(), static_cast<int>(right.size()), TRUE) == CSTR_LESS_THAN;
    }
};

inline bool lostDevice([[maybe_unused]] const std::exception& error) {
    // Retire a device only for driver failures, bad input file must not disable its GPU
#if defined(_USE_OPENCL_)
    if (const auto* value = dynamic_cast<const cl::Error*>(&error))
        return value->err() == CL_DEVICE_NOT_AVAILABLE || value->err() == CL_DEVICE_NOT_FOUND;
    if (const auto* value = dynamic_cast<const cl_utils::DriverError*>(&error))
        return value->code == CL_DEVICE_NOT_AVAILABLE || value->code == CL_DEVICE_NOT_FOUND;
#elif defined(_USE_CUDA_)
    if (const auto* value = dynamic_cast<const cuda_utils::CudaError*>(&error))
        return value->code == cudaErrorDevicesUnavailable || value->code == cudaErrorDeviceUninitialized || value->code == cudaErrorIllegalAddress || value->code == cudaErrorLaunchFailure ||
               value->code == cudaErrorLaunchTimeout;
#endif
    return false;
}
} // namespace WatermarkCore::BatchUtils
