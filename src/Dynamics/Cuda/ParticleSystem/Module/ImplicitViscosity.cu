#include "ImplicitViscosity.h"
#include "Node.h"

namespace dyno
{
	template<typename TDataType>
	ImplicitViscosity<TDataType>::ImplicitViscosity()
		:ParticleApproximation<TDataType>()
	{
		this->varKernelType()->setCurrentKey(ParticleApproximation<TDataType>::EKernelType::KT_Smooth);
		this->varViscosity()->setValue(Real(0.05));
	}

	template<typename TDataType>
	ImplicitViscosity<TDataType>::~ImplicitViscosity()
	{
		mVelOld.clear();
		mVelBuf.clear();
	}

	template<typename Real, typename Coord, typename Kernel>
	__global__ void IV_ApplyViscosity(
		DArray<Coord> velNew,
		DArray<Coord> posArr,
		DArray<Coord> velOld,
		DArray<Coord> velBuf,
		DArrayList<int> neighbors,
		Real viscosity,
		Real smoothingLength,
		Real dt,
		Kernel weight,
		Real scale)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= velNew.size()) return;

		Coord dv_i(0);
		Coord pos_i = posArr[pId];
		Coord vel_i = velBuf[pId];
		Real totalWeight = 0.0f;

		List<int>& list_i = neighbors[pId];
		int nbSize = list_i.size();
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = list_i[ne];
			Real r = (pos_i - posArr[j]).norm();

			if (r > EPSILON)
			{
				Real w = weight(r, smoothingLength, scale);
				totalWeight += w;
				dv_i += w * velBuf[j];
			}
		}

		Real b = dt * viscosity / smoothingLength;
		b = totalWeight < EPSILON ? 0.0f : b;

		totalWeight = totalWeight < EPSILON ? 1.0f : totalWeight;

		dv_i /= totalWeight;

		velNew[pId] = velOld[pId] / (1.0f + b) + dv_i * b / (1.0f + b);
	}

	template<typename TDataType>
	void ImplicitViscosity<TDataType>::compute()
	{
		auto& poss = this->inPosition()->getData();
		auto& vels = this->inVelocity()->getData();
		auto& nbrIds = this->inNeighborIds()->getData();
		Real  h = this->inSmoothingLength()->getData();
		Real dt = this->inTimeStep()->getData();

		int num = vels.size();
		uint pDims = cudaGridSize(num, BLOCK_SIZE);

		mVelOld.resize(num);
		mVelBuf.resize(num);

		Real vis = this->varViscosity()->getData();

		int iterNum = this->varInterationNumber()->getData();

		mVelOld.assign(vels);
		for (int t = 0; t < iterNum; t++)
		{
			mVelBuf.assign(vels);
			cuZerothOrder(num, this->varKernelType()->getDataPtr()->currentKey(), this->mScalingFactor,
				IV_ApplyViscosity,
				vels,
				poss,
				mVelOld,
				mVelBuf,
				nbrIds,
				vis,
				h,
				dt);
		}
	}

	DEFINE_CLASS(ImplicitViscosity);
}