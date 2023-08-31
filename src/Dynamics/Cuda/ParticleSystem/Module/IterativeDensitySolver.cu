#include "IterativeDensitySolver.h"

#include "SummationDensity.h"

namespace dyno
{
//	IMPLEMENT_TCLASS(DensityPBD, TDataType)

	template<typename TDataType>
	IterativeDensitySolver<TDataType>::IterativeDensitySolver()
		: ParticleApproximation<TDataType>()
	{
		this->varIterationNumber()->setValue(3);
		this->varRestDensity()->setValue(Real(1000));

		mSummation = std::make_shared<SummationDensity<TDataType>>();

		this->inSmoothingLength()->connect(mSummation->inSmoothingLength());
		this->inSamplingDistance()->connect(mSummation->inSamplingDistance());
		this->inPosition()->connect(mSummation->inPosition());
		this->inNeighborIds()->connect(mSummation->inNeighborIds());

		mSummation->outDensity()->connect(this->outDensity());
	}

	template<typename TDataType>
	IterativeDensitySolver<TDataType>::~IterativeDensitySolver()
	{
		mLamda.clear();
		mDeltaPos.clear();
		mPositionOld.clear();
	}


	template <typename Real, typename Coord, typename Kernel>
	__global__ void K_ComputeLambdas(
		DArray<Real> lambdaArr,
		DArray<Real> rhoArr,
		DArray<Coord> posArr,
		DArrayList<int> neighbors,
		Real smoothingLength,
		Kernel gradient,
		Real scale)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.size()) return;

		Coord pos_i = posArr[pId];

		Real lamda_i = Real(0);
		Coord grad_ci(0);

		List<int>& list_i = neighbors[pId];
		int nbSize = list_i.size();
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = list_i[ne];
			Real r = (pos_i - posArr[j]).norm();

			if (r > EPSILON)
			{
				Coord g = gradient(r, smoothingLength, scale) * (pos_i - posArr[j]) * (1.0f / r);
				grad_ci += g;
				lamda_i += g.dot(g);
			}
		}

		lamda_i += grad_ci.dot(grad_ci);

		Real rho_i = rhoArr[pId];

		lamda_i = -(rho_i - 1000.0f) / (lamda_i + 0.1f);

		lambdaArr[pId] = lamda_i > 0.0f ? 0.0f : lamda_i;
	}

	template <typename Real, typename Coord, typename Kernel>
	__global__ void K_ComputeDisplacement(
		DArray<Coord> dPos,
		DArray<Real> lambdas,
		DArray<Coord> posArr,
		DArrayList<int> neighbors,
		Real smoothingLength,
		Real dt,
		Kernel gradient,
		Real scale)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.size()) return;

		Coord pos_i = posArr[pId];
		Real lamda_i = lambdas[pId];

		Coord dP_i(0);
		List<int>& list_i = neighbors[pId];
		int nbSize = list_i.size();
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = list_i[ne];
			Real r = (pos_i - posArr[j]).norm();
			if (r > EPSILON)
			{
				Coord dp_ij = 10.0f * (pos_i - posArr[j]) * (lamda_i + lambdas[j]) * gradient(r, smoothingLength, scale) * (1.0 / r);
				dP_i += dp_ij;

				atomicAdd(&dPos[pId][0], dp_ij[0]);
				atomicAdd(&dPos[j][0], -dp_ij[0]);

				if (Coord::dims() >= 2)
				{
					atomicAdd(&dPos[pId][1], dp_ij[1]);
					atomicAdd(&dPos[j][1], -dp_ij[1]);
				}

				if (Coord::dims() >= 3)
				{
					atomicAdd(&dPos[pId][2], dp_ij[2]);
					atomicAdd(&dPos[j][2], -dp_ij[2]);
				}
			}
		}
	}

	template <typename Real, typename Coord>
	__global__ void K_UpdatePosition(
		DArray<Coord> posArr,
		DArray<Coord> velArr,
		DArray<Coord> dPos,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.size()) return;

		posArr[pId] += dPos[pId];
	}

	template<typename TDataType>
	void IterativeDensitySolver<TDataType>::compute()
	{
		int num = this->inPosition()->size();

		if (mPositionOld.size() != this->inPosition()->size())
			mPositionOld.resize(this->inPosition()->size());

		mPositionOld.assign(this->inPosition()->getData());

		if (this->outDensity()->size() != this->inPosition()->size())
			this->outDensity()->resize(this->inPosition()->size());

		if (mDeltaPos.size() != this->inPosition()->size())
			mDeltaPos.resize(this->inPosition()->size());

		if (mLamda.size() != this->inPosition()->size())
			mLamda.resize(this->inPosition()->size());

		int it = 0;

		int itNum = this->varIterationNumber()->getData();
		while (it < itNum)
		{
			takeOneIteration();

			it++;
		}

		updateVelocity();
	}


	template<typename TDataType>
	void IterativeDensitySolver<TDataType>::takeOneIteration()
	{
		Real dt = this->inTimeStep()->getData();
		int num = this->inPosition()->size();

		mDeltaPos.reset();
		mSummation->varRestDensity()->setValue(this->varRestDensity()->getValue());
		mSummation->varKernelType()->setCurrentKey(this->varKernelType()->currentKey());
		mSummation->update();

		cuFirstOrder(num, this->varKernelType()->getDataPtr()->currentKey(), this->mScalingFactor,
			K_ComputeLambdas,
			mLamda,
			mSummation->outDensity()->getData(),
			this->inPosition()->getData(),
			this->inNeighborIds()->getData(),
			this->inSmoothingLength()->getValue());

		cuFirstOrder(num, this->varKernelType()->getDataPtr()->currentKey(), this->mScalingFactor,
			K_ComputeDisplacement,
			mDeltaPos,
			mLamda,
			this->inPosition()->getData(),
			this->inNeighborIds()->getData(),
			this->inSmoothingLength()->getData(),
			dt);

		cuExecute(num, K_UpdatePosition,
			this->inPosition()->getData(),
			this->inVelocity()->getData(),
			mDeltaPos,
			dt);
	}

	template <typename Real, typename Coord>
	__global__ void DP_UpdateVelocity(
		DArray<Coord> velArr,
		DArray<Coord> curPos,
		DArray<Coord> prePos,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= velArr.size()) return;

		velArr[pId] += (curPos[pId] - prePos[pId]) / dt;
	}

	template<typename TDataType>
	void IterativeDensitySolver<TDataType>::updateVelocity()
	{
		int num = this->inPosition()->size();

		Real dt = this->inTimeStep()->getData();

		cuExecute(num, DP_UpdateVelocity,
			this->inVelocity()->getData(),
			this->inPosition()->getData(),
			mPositionOld,
			dt);
	}

	DEFINE_CLASS(IterativeDensitySolver);
}