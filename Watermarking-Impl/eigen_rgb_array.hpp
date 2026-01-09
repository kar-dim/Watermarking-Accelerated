#pragma once
#include "Eigen/Core"
#include <array>
#include <cstdint>
using EigenArrayRGB = std::array<Eigen::ArrayXXf, 3>;
using EigenArrayU8RGB = std::array<Eigen::Array<uint8_t, Eigen::Dynamic, Eigen::Dynamic>, 3>;