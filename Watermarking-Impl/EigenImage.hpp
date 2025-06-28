#pragma once
#include "eigen_rgb_array.hpp"
#include <Eigen/Dense>
#include <utility>
#include <variant>

/*!
 *  \brief  Holds either an Eigen array or Eigen RGB array by using std::variant. CPU implementation.
 *  \author Dimitris Karatzas
 */
class EigenImage 
{
private:
    std::variant<std::monostate, Eigen::ArrayXXf, EigenArrayRGB> data;

public:
    EigenImage() = default;

	//move constructors
    EigenImage(EigenImage&&) noexcept = default;
    EigenImage(Eigen::ArrayXXf&& gray) noexcept
        : data(std::move(gray))
    { }

    EigenImage(EigenArrayRGB&& rgb) noexcept
        : data(std::move(rgb))
    { }

	//move assignment operators
    EigenImage& operator=(EigenImage&& other) noexcept 
    {
        data = std::move(other.data);
        return *this;
    };

    EigenImage& operator=(Eigen::ArrayXXf&& gray) noexcept 
    {
        data = std::move(gray);
        return *this;
    };

    EigenImage& operator=(EigenArrayRGB&& rgb) noexcept 
    {
        data = std::move(rgb);
        return *this;
    };

    //copy constructors
    EigenImage(const EigenImage& other) : data(other.data)
    { };

    EigenImage(const Eigen::ArrayXXf& gray) : data(gray)
    { };

    EigenImage(const EigenArrayRGB& rgb) : data(rgb)
    { };

	//copy assignment operators
    EigenImage& operator=(const EigenImage& other) 
    {
        data = other.data;
        return *this;
    }
    EigenImage& operator=(const Eigen::ArrayXXf& gray) 
    {
        data = gray;
        return *this;
    };

    EigenImage& operator=(const EigenArrayRGB& rgb) 
    {
        data = rgb;
        return *this;
    };

    //helper methods to retrieve the actual data type
    bool isRGB() const { return std::holds_alternative<EigenArrayRGB>(data); }

    Eigen::ArrayXXf& getGray() { return std::get<Eigen::ArrayXXf>(data); }

    const Eigen::ArrayXXf& getGray() const { return std::get<Eigen::ArrayXXf>(data); }

    EigenArrayRGB& getRGB() { return std::get<EigenArrayRGB>(data); }

    const EigenArrayRGB& getRGB() const { return std::get<EigenArrayRGB>(data); }
};