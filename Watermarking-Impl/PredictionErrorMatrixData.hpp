#pragma once
#include <Eigen/Dense>
#include <vector>

template <int p>
class PredictionErrorMatrixData
{
private:
	static constexpr int localSize = (p * p) - 1;
	using LocalVector = Eigen::Matrix<float, localSize, 1>;
	using LocalVectorDiag = Eigen::Matrix<float, localSize * (localSize + 1) / 2, 1>;
	using LocalMatrix = Eigen::Matrix<float, localSize, localSize>;
	const int numThreads;
	LocalVectorDiag RxVec;
	LocalVector coefficients, rx;
	LocalMatrix Rx;
	std::vector<LocalVectorDiag> RxVec_all;
	std::vector<LocalVector> rx_all;

public:
	//initialize prediction error matrix data (allocate memory) for a given number of threads
	PredictionErrorMatrixData(const int numThreads) : numThreads(numThreads), RxVec_all(numThreads), rx_all(numThreads)
	{ }

	//sets all Rx,rx matrices and vectors to zero
	void setZero()
	{
		RxVec.setZero();
		Rx.setZero();
		rx.setZero();
		for (int i = 0; i < numThreads; i++)
		{
			RxVec_all[i].setZero();
			rx_all[i].setZero();
		}
	}

	//computes the prediction error matrices for each thread
	void computePredictionErrorMatrices(const LocalVector& x_, const float pixelValue, const int index)
	{
		//calculate Rx optimized by using a vector representing the lower-triangular only instead of a matrix
		auto& currentRx = RxVec_all[index];
		for (int i = 0, k = 0; i < localSize; i++)
			for (int j = 0; j <= i; j++, k++)
				currentRx(k) += x_(i) * x_(j);
		//calculate rx vector
		rx_all[index].noalias() += x_ * pixelValue;
	}

	//calculates the coefficients by reducing (sum) the Rx/rx matrices calculated by each thread, and reconstructing the full Rx matrix
	void computeCoefficients()
	{
		//reduction sums of Rx,rx of each thread
		for (int i = 0; i < numThreads; i++)
		{
			RxVec.noalias() += RxVec_all[i];
			rx.noalias() += rx_all[i];
		}
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
		coefficients = Rx.colPivHouseholderQr().solve(rx);
	}

	LocalVector getCoefficients() const { return coefficients; }
};