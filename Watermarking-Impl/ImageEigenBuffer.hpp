#pragma once
#include "eigen_rgb_array.hpp"
#include <Eigen/Dense>
#include <utility>
#include <variant>

/*!
 *  \brief  Holds either an Eigen array or Eigen RGB array by using std::variant. CPU implementation.
 *  \author Dimitris Karatzas
 */
class ImageEigenBuffer 
{
private:
    std::variant<std::monostate, Eigen::ArrayXXf, EigenArrayRGB> data;

public:
    ImageEigenBuffer() = default;
    ImageEigenBuffer(const Eigen::ArrayXXf& gray) : data(gray) {}
    ImageEigenBuffer(const EigenArrayRGB& rgb) : data(rgb) {}
    ImageEigenBuffer(Eigen::ArrayXXf&& gray) noexcept : data(std::move(gray)) {}
    ImageEigenBuffer(EigenArrayRGB&& rgb)    noexcept : data(std::move(rgb)) {}
    ImageEigenBuffer& operator=(const Eigen::ArrayXXf& gray) { data = gray; return *this; }
    ImageEigenBuffer& operator=(const EigenArrayRGB& rgb) { data = rgb; return *this; }
    ImageEigenBuffer& operator=(Eigen::ArrayXXf&& gray) { data = std::move(gray); return *this; }
    ImageEigenBuffer& operator=(EigenArrayRGB&& rgb) { data = std::move(rgb); return *this; }

    //apply watermark
    void applyWatermark(const Eigen::ArrayXXf& uStrengthened, ImageEigenBuffer& output) const
    {
        if (isRGB()) 
        {
#pragma omp parallel for
            for (int channel = 0; channel < 3; channel++)
                output.getRGB()[channel] = (getRGB()[channel] + uStrengthened).cwiseMax(0).cwiseMin(255);
        }
        else
            output.getGray() = (getGray() + uStrengthened).cwiseMax(0).cwiseMin(255);
    }

    //assign input to output
    void assignTo(ImageEigenBuffer& output) const
    {
        if (isRGB())
            output.getRGB() = getRGB();
        else
            output.getGray() = getGray();
	}

    //helper methods to retrieve the actual data type
    bool isRGB() const { return std::holds_alternative<EigenArrayRGB>(data); }

    Eigen::ArrayXXf& getGray() { return std::get<Eigen::ArrayXXf>(data); }

    const Eigen::ArrayXXf& getGray() const { return std::get<Eigen::ArrayXXf>(data); }

    EigenArrayRGB& getRGB() { return std::get<EigenArrayRGB>(data); }

    const EigenArrayRGB& getRGB() const { return std::get<EigenArrayRGB>(data); }
};