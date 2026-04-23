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
class ImageEigenBuffer {
  private:
    std::variant<std::monostate, Eigen::ArrayXXf, EigenArrayRGB> data;

    // helper method to process the output (either assign or apply watermark, always float to uint8)
    // parallelized over columns using Eigen Map so each thread gets Eigen's vectorized op
    template <bool EMBED>
    void processOutput(ImageEigenOutputBuffer& output, const Eigen::ArrayXXf* uStrengthened = nullptr, const float scale = 0.0f) const {
        using FloatMap = Eigen::Map<const Eigen::ArrayXf>;
        using U8Map = Eigen::Map<Eigen::Array<uint8_t, Eigen::Dynamic, 1>>;
        auto finalize = [](const auto& expr) { return (expr + 0.5f).cwiseMax(0.0f).cwiseMin(255.0f).template cast<uint8_t>(); };

        if (isRGB()) {
            auto& rgbOutput = output.getRGB();
            const auto& rgbIn = getRGB();
            const int rows = static_cast<int>(rgbIn[0].rows());
            const int cols = static_cast<int>(rgbIn[0].cols());
#pragma omp parallel for schedule(static)
            for (int col = 0; col < cols; col++) {
                const int off = col * rows;
                FloatMap r0(rgbIn[0].data() + off, rows);
                FloatMap r1(rgbIn[1].data() + off, rows);
                FloatMap r2(rgbIn[2].data() + off, rows);
                U8Map o0(rgbOutput[0].data() + off, rows);
                U8Map o1(rgbOutput[1].data() + off, rows);
                U8Map o2(rgbOutput[2].data() + off, rows);
                if constexpr (EMBED) {
                    const auto u = FloatMap(uStrengthened->data() + off, rows) * scale;
                    o0 = finalize(r0 + u);
                    o1 = finalize(r1 + u);
                    o2 = finalize(r2 + u);
                } else {
                    o0 = finalize(r0);
                    o1 = finalize(r1);
                    o2 = finalize(r2);
                }
            }
        } else {
            const auto& grayIn = getGray();
            auto& grayOut = output.getGray();
            const int rows = static_cast<int>(grayIn.rows());
            const int cols = static_cast<int>(grayIn.cols());
#pragma omp parallel for schedule(static)
            for (int col = 0; col < cols; col++) {
                const int off = col * rows;
                U8Map dst(grayOut.data() + off, rows);
                FloatMap src(grayIn.data() + off, rows);
                if constexpr (EMBED)
                    dst = finalize(src + FloatMap(uStrengthened->data() + off, rows) * scale);
                else
                    dst = finalize(src);
            }
        }
    }

  public:
    ImageEigenBuffer() = default;
    ImageEigenBuffer(const Eigen::ArrayXXf& gray) : data(gray) {}
    ImageEigenBuffer(const EigenArrayRGB& rgb) : data(rgb) {}
    ImageEigenBuffer(Eigen::ArrayXXf&& gray) noexcept : data(std::move(gray)) {}
    ImageEigenBuffer(EigenArrayRGB&& rgb) noexcept : data(std::move(rgb)) {}
    ImageEigenBuffer& operator=(const Eigen::ArrayXXf& gray) {
        data = gray;
        return *this;
    }
    ImageEigenBuffer& operator=(const EigenArrayRGB& rgb) {
        data = rgb;
        return *this;
    }
    ImageEigenBuffer& operator=(Eigen::ArrayXXf&& gray) {
        data = std::move(gray);
        return *this;
    }
    ImageEigenBuffer& operator=(EigenArrayRGB&& rgb) {
        data = std::move(rgb);
        return *this;
    }

    // apply watermark (float to uint8), the unscaled u = mask*w is multiplied by scale (fused)
    void applyWatermark(const Eigen::ArrayXXf& uStrengthened, const float scale, ImageEigenOutputBuffer& output) const { processOutput<true>(output, &uStrengthened, scale); }

    // assign input to output (float to uint8)
    void assignTo(ImageEigenOutputBuffer& output) const { processOutput<false>(output, nullptr); }

    // helper methods to retrieve the actual data type
    bool isRGB() const { return std::holds_alternative<EigenArrayRGB>(data); }

    Eigen::ArrayXXf& getGray() { return std::get<Eigen::ArrayXXf>(data); }

    const Eigen::ArrayXXf& getGray() const { return std::get<Eigen::ArrayXXf>(data); }

    EigenArrayRGB& getRGB() { return std::get<EigenArrayRGB>(data); }

    const EigenArrayRGB& getRGB() const { return std::get<EigenArrayRGB>(data); }
};