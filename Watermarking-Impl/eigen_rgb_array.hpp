#pragma once
#include <array>
#include <cstdint>
#include <Eigen/Dense>
using EigenArrayRGB = std::array<Eigen::ArrayXXf, 3>;
using EigenArrayU8RGB = std::array<Eigen::Array<uint8_t, Eigen::Dynamic, Eigen::Dynamic>, 3>;