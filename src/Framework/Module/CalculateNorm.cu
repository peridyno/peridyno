#include "CalculateNorm.h"

namespace dyno 
{
	IMPLEMENT_TCLASS(CalculateNorm, TDataType)

	template <typename Real, typename Coord>
	__global__ void CN_CalculateNorm(
		DArray<Real> norm,
		DArray<Coord> vec)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= vec.size()) return;

		norm[tId] = vec[tId].norm();
	}

	template<typename TDataType>
	void CalculateNorm<TDataType>::compute()
	{
		auto& inData = this->inVec()->getData();

		int num = inData.size();

		if (this->outNorm()->isEmpty())
		{
			this->outNorm()->allocate();
		}

		auto& outData = this->outNorm()->getData();
		if (outData.size() != num)
		{
			outData.resize(num);
		}

		cuExecute(num,
			CN_CalculateNorm,
			outData,
			inData);
	}

	DEFINE_CLASS(CalculateNorm);
}