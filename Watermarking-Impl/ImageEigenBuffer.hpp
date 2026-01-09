#pragma once
#include "Eigen/Core"
#include "eigen_rgb_array.hpp"
#include "ImageEigenOutputBuffer.hpp"
#include <cstdint>
#include <variant>

/*!
 *  \brief  Holds either an Eigen array or Eigen RGB array by using std::variant. CPU implementation.
 *  \author Dimitris Karatzas
 */
class ImageEigenBuffer 
{
private:
    std::variant<std::monostate, Eigen::ArrayXXf, EigenArrayRGB> data;

	//helper method to process the output (either assign or apply watermark, always float to uint8)
    template<bool EMBED>
    void processOutput(ImageEigenOutputBuffer& output, const Eigen::ArrayXXf* uStrengthened = nullptr) const
    {
        auto finalize = [](const auto& expr) { return expr.round().cwiseMax(0).cwiseMin(255).template cast<uint8_t>(); };
        if (isRGB())
        {
            auto& rgbOutput = output.getRGB();
            const auto& rgbInput = getRGB();
#pragma omp parallel for
            for (int channel = 0; channel < 3; channel++)
            {
                if constexpr (EMBED)
                    rgbOutput[channel] = finalize(rgbInput[channel] + *uStrengthened);
                else
                    rgbOutput[channel] = finalize(rgbInput[channel]);
            }
        }
        else
        {
            if constexpr (EMBED)
                output.getGray() = finalize(getGray() + *uStrengthened);
            else
                output.getGray() = finalize(getGray());
        }
    }

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

	//apply watermark (float to uint8)
    void applyWatermark(const Eigen::ArrayXXf& uStrengthened, ImageEigenOutputBuffer& output) const
    {
        processOutput<true>(output, &uStrengthened);
    }

    //assign input to output (float to uint8)
    void assignTo(ImageEigenOutputBuffer& output) const
    {
        processOutput<false>(output, nullptr);
	}

    //helper methods to retrieve the actual data type
    bool isRGB() const { return std::holds_alternative<EigenArrayRGB>(data); }

    Eigen::ArrayXXf& getGray() { return std::get<Eigen::ArrayXXf>(data); }

    const Eigen::ArrayXXf& getGray() const { return std::get<Eigen::ArrayXXf>(data); }

    EigenArrayRGB& getRGB() { return std::get<EigenArrayRGB>(data); }

    const EigenArrayRGB& getRGB() const { return std::get<EigenArrayRGB>(data); }
};