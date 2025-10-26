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

	//move constructors
    ImageEigenBuffer(ImageEigenBuffer&&) noexcept = default;
    ImageEigenBuffer(Eigen::ArrayXXf&& gray) noexcept
        : data(std::move(gray))
    { }

    ImageEigenBuffer(EigenArrayRGB&& rgb) noexcept
        : data(std::move(rgb))
    { }

	//move assignment operators
    ImageEigenBuffer& operator=(ImageEigenBuffer&& other) noexcept 
    {
        data = std::move(other.data);
        return *this;
    };

    ImageEigenBuffer& operator=(Eigen::ArrayXXf&& gray) noexcept 
    {
        data = std::move(gray);
        return *this;
    };

    ImageEigenBuffer& operator=(EigenArrayRGB&& rgb) noexcept 
    {
        data = std::move(rgb);
        return *this;
    };

    //copy constructors
    ImageEigenBuffer(const ImageEigenBuffer& other) : data(other.data)
    { };

    ImageEigenBuffer(const Eigen::ArrayXXf& gray) : data(gray)
    { };

    ImageEigenBuffer(const EigenArrayRGB& rgb) : data(rgb)
    { };

	//copy assignment operators
    ImageEigenBuffer& operator=(const ImageEigenBuffer& other) 
    {
        data = other.data;
        return *this;
    }
    ImageEigenBuffer& operator=(const Eigen::ArrayXXf& gray) 
    {
        data = gray;
        return *this;
    };

    ImageEigenBuffer& operator=(const EigenArrayRGB& rgb) 
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