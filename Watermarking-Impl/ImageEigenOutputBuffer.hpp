#pragma once
#include "eigen_rgb_array.hpp"
#include <Eigen/Dense>
#include <variant>

class ImageEigenOutputBuffer
{
private:
    using Gray8Buffer = Eigen::Array<uint8_t, Eigen::Dynamic, Eigen::Dynamic>;

    std::variant<std::monostate, Gray8Buffer, EigenArrayU8RGB> data;

public:
    ImageEigenOutputBuffer(const Gray8Buffer& gray) : data(gray) {}
	ImageEigenOutputBuffer(const EigenArrayU8RGB& rgb) : data(rgb) {}

    bool isRGB() const { return std::holds_alternative<EigenArrayU8RGB>(data); }
    const Gray8Buffer& getGray() const { return std::get<Gray8Buffer>(data); }
    const EigenArrayU8RGB& getRGB() const { return std::get<EigenArrayU8RGB>(data); }
    Gray8Buffer& getGray() { return std::get<Gray8Buffer>(data); }
    EigenArrayU8RGB& getRGB() { return std::get<EigenArrayU8RGB>(data); }
};