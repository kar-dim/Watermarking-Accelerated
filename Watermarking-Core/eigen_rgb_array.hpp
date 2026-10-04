#pragma once
#include "Eigen/Core"
#include <array>
#include <cstdint>

/*!
 *  \brief  Type alias for 3-channel Eigen planar RGB byte arrays
 *  \author Dimitris Karatzas
 */
using EigenArrayU8RGB = std::array<Eigen::Array<uint8_t, Eigen::Dynamic, Eigen::Dynamic>, 3>;