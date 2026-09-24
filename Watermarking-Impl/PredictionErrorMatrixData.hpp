#pragma once
#include "Eigen/Cholesky"
#include "Eigen/Core"
#include <algorithm>
#include <vector>

// Aligned matrix to 64 bytes boundary (cache-line friendly),
// improves performance massively when used in parallel computations
template <typename T>
struct alignas(64) AlignedMatrix {
    T mat;
};

/*!
 *  \brief  Helper class holding Prediction Error Matrix Data and common operations,
 *			used by the Eigen implementation.
 *  \author Dimitris Karatzas
 */
template <int p>
class PredictionErrorMatrixData {
  private:
    static constexpr int localSize = (p * p) - 1;
    static constexpr int center = p / 2;
    using LocalVector = Eigen::Matrix<float, localSize, 1>;
    // double because float precision may not be enough especially for this combo: LOW threads, HIGH p, HIGH image size
    using AccumVector = Eigen::Matrix<double, localSize, 1>;
    using AccumMatrix = Eigen::Matrix<double, localSize, localSize>;

  public:
    LocalVector coefficients;
    AccumVector rx;
    AccumMatrix Rx;
    std::vector<AlignedMatrix<AccumMatrix>> RxAll;
    std::vector<AlignedMatrix<AccumVector>> rxAll;
    std::vector<AlignedMatrix<Eigen::MatrixXf>> neighborMatricesAll;
    std::vector<int> offsets;

  public:
    // initialize prediction error matrix data (allocate memory) for a given number of threads
    PredictionErrorMatrixData(const int numThreads, const int baseRows) : RxAll(numThreads), rxAll(numThreads) {
        offsets.reserve(localSize);
        for (int dj = 0; dj < p; dj++) {
            for (int di = 0; di < p; di++) {
                if (di == center && dj == center)
                    continue;
                // col-major offset
                offsets.push_back((dj - center) * baseRows + (di - center));
            }
        }
        // initialize the neighbor matrices only for large local sizes (p>=5), for small local sizes the dot product approach is faster
        if constexpr (p >= 5)
            neighborMatricesAll.resize(numThreads, AlignedMatrix<Eigen::MatrixXf>{.mat = Eigen::MatrixXf(std::max(baseRows - 2 * center, 0), localSize)});
    }

    // sets all Rx, rx matrices and vectors to zero
    void setZero() {
        Rx.setZero();
        rx.setZero();
        for (auto& RxMat : RxAll)
            RxMat.mat.setZero();
        for (auto& rxVec : rxAll)
            rxVec.mat.setZero();
    }

    // border pixels: rank 1 symmetric update (upper triangle) + rx accumulation
    void computePredictionErrorMatrices(const LocalVector& x_, const float pixelValue, const int index) {
        const AccumVector neighbors = x_.template cast<double>();
        RxAll[index].mat.template selfadjointView<Eigen::Upper>().rankUpdate(neighbors);
        rxAll[index].mat.noalias() += neighbors * static_cast<double>(pixelValue);
    }

    // reduce thread local matrices, then solve Rx * coefficients = rx with Cholesky
    bool computeCoefficients() {
        for (const auto& RxMat : RxAll)
            Rx += RxMat.mat;
        for (const auto& rxVec : rxAll)
            rx.noalias() += rxVec.mat;
        if (!Rx.allFinite() || !rx.allFinite())
            return false;
        // Cholesky reads upper triangle only
        Eigen::LLT<AccumMatrix, Eigen::Upper> llt(Rx);
        if (llt.info() != Eigen::Success)
            return false;
        coefficients = llt.solve(rx).template cast<float>();
        return true;
    }
};
