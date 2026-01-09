#pragma once
#include "Eigen/Cholesky"
#include "Eigen/Core"
#include <vector>

//Aligned matrix to 64 bytes boundary (cache-line friendly),
//improves performance massively when used in parallel computations
template<typename T>
struct alignas(64) AlignedMatrix { T mat; };

/*!
 *  \brief  Helper class holding Prediction Error Matrix Data and common operations,
 *			used by the Eigen implementation.
 *  \author Dimitris Karatzas
 */
template <int p>
class PredictionErrorMatrixData
{
private:
	static constexpr int localSize = (p * p) - 1;
	static constexpr int center = p / 2;
	using LocalVector = Eigen::Matrix<float, localSize, 1>;
	using LocalVectorDiag = Eigen::Matrix<float, localSize * (localSize + 1) / 2, 1>;
	using LocalMatrix = Eigen::Matrix<float, localSize, localSize>;
public:
	LocalVectorDiag RxVec;
	LocalVector coefficients, rx;
	LocalMatrix Rx;
	std::vector<AlignedMatrix<LocalVectorDiag>> RxVec_all;
	std::vector<AlignedMatrix<LocalVector>> rx_all;
	std::vector<int> offsets;

public:
	//initialize prediction error matrix data (allocate memory) for a given number of threads
	PredictionErrorMatrixData(const int numThreads, const int baseRows) : RxVec_all(numThreads), rx_all(numThreads)
	{
		offsets.reserve(localSize);
		for (int dj = 0; dj < p; dj++)
			for (int di = 0; di < p; di++)
			{
				if (di == center && dj == center)
					continue;
				//col-major offset
				offsets.push_back((dj - center) * baseRows + (di - center));
			}
	}

	//sets all Rx,rx matrices and vectors to zero
	void setZero()
	{
		RxVec.setZero();
		Rx.setZero();
		rx.setZero();
		for (auto& rxVec : RxVec_all) rxVec.mat.setZero();
		for (auto& rx : rx_all) rx.mat.setZero();
	}

	//computes the prediction error matrices for each thread
	void computePredictionErrorMatrices(const LocalVector& x_, const float pixelValue, const int index)
	{
		//calculate Rx optimized by using a vector representing the lower-triangular only instead of a matrix
		auto& currentRx = RxVec_all[index].mat;
		for (int i = 0, k = 0; i < localSize; i++)
			for (int j = 0; j <= i; j++, k++)
				currentRx(k) += x_(i) * x_(j);
		//calculate rx vector
		rx_all[index].mat.noalias() += x_ * pixelValue;
	}

	//calculates the coefficients by reducing (sum) the Rx/rx matrices calculated by each thread, and reconstructing the full Rx matrix
	bool computeCoefficients()
	{
		//reduction sums of Rx,rx of each thread
		for (const auto& RxVal : RxVec_all)
			RxVec.noalias() += RxVal.mat;
		for (const auto& rxVal : rx_all)
			rx.noalias() += rxVal.mat;
		if (!RxVec.allFinite())
			return false;
		
		//Reconstruct full Rx matrix from the vector
		for (int i = 0, k = 0; i < localSize; i++) 
		{
			for (int j = 0; j <= i; j++, k++) 
			{
				float value = RxVec(k);
				Rx(i, j) = value;
				Rx(j, i) = value;
			}
		}
		//solve the linear system Rx * coefficients = rx for coefficients
		Eigen::LLT<LocalMatrix> llt(Rx);
		if (llt.info() != Eigen::Success)
			return false;
		
		coefficients = llt.solve(rx);
		return true;
	}
};