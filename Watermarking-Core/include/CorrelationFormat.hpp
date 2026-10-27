#pragma once

#include <cmath>
#include <format>
#include <string>

/*!
 * \brief Shared display formatting for detection correlations
 * \author Dimitris Karatzas
 */

namespace WatermarkCore {
inline std::string formatCorrelationPercent(const double correlation) {
    const double percent = std::round(correlation * 1000.0) / 10.0;
    // Return text directly to avoid keeping a negative zero
    if (percent == 0.0)
        return "0.0%";
    return std::format("{:.1f}%", percent);
}
} // namespace WatermarkCore
