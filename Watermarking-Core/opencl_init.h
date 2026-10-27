#pragma once
#define CL_HPP_MINIMUM_OPENCL_VERSION 200
#define CL_HPP_TARGET_OPENCL_VERSION 300
#define CL_HPP_ENABLE_EXCEPTIONS

/*!
 *  \brief  OpenCL target version configuration and C++ bindings header inclusion
 *  \author Dimitris Karatzas
 */

#include <CL/opencl.hpp>
#include <stdexcept>
#include <string>

namespace cl_utils {
class DriverError : public std::runtime_error {
  public:
    DriverError(const cl_int code, const std::string& operation) : std::runtime_error(operation + " failed with OpenCL error " + std::to_string(code)), code(code) {}
    const cl_int code;
};
} // namespace cl_utils
