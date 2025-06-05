#pragma once
#include "eigen_rgb_array.hpp"
#include <Eigen/Dense>
#include <utility>
#include <variant>

class EigenImage {
public:
    std::variant<std::monostate, Eigen::ArrayXXf, EigenArrayRGB> data;

    EigenImage() = default;
    EigenImage(const Eigen::ArrayXXf& gray) = delete;
    EigenImage(Eigen::ArrayXXf&& gray)
        : data(std::move(gray)) 
    { }

    EigenImage(const EigenArrayRGB& rgb) = delete;
    EigenImage(EigenArrayRGB&& rgb)
        : data(std::move(rgb)) 
    { }

    bool isRGB() const 
    {
        return std::holds_alternative<EigenArrayRGB>(data);
    }

    Eigen::ArrayXXf& getGray() 
    {
        return std::get<Eigen::ArrayXXf>(data);
    }

    const Eigen::ArrayXXf& getGray() const 
    {
        return std::get<Eigen::ArrayXXf>(data);
    }

    EigenArrayRGB& getRGB() 
    {
        return std::get<EigenArrayRGB>(data);
    }

    const EigenArrayRGB& getRGB() const 
    {
        return std::get<EigenArrayRGB>(data);
    }
};