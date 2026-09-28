#pragma once
#include "Eigen/Core"
#include <utility>

/*!
 *  \brief  Holds a float image plane (luma, watermark) as an Eigen CPU array
 *  \author Dimitris Karatzas
 */
class ImageEigenBuffer {
  private:
    Eigen::ArrayXXf data;

  public:
    ImageEigenBuffer() = default;
    ImageEigenBuffer(const Eigen::ArrayXXf& gray) : data(gray) {}
    ImageEigenBuffer(Eigen::ArrayXXf&& gray) noexcept : data(std::move(gray)) {}
    ImageEigenBuffer& operator=(const Eigen::ArrayXXf& gray) {
        data = gray;
        return *this;
    }
    ImageEigenBuffer& operator=(Eigen::ArrayXXf&& gray) {
        data = std::move(gray);
        return *this;
    }

    Eigen::ArrayXXf& getGray() { return data; }

    const Eigen::ArrayXXf& getGray() const { return data; }
};
