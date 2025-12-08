#include "IterativeFrictionSolver.h"

#include "SummationDensity.h"

#include "Algorithm/CudaRand.h"

namespace dyno
{
//	IMPLEMENT_TCLASS(DensityPBD, TDataType)

	template<typename TDataType>
	IterativeFrictionSolver<TDataType>::IterativeFrictionSolver()
		: ParticleApproximation<TDataType>()
	{
		this->varIterationNumber()->setValue(3);
		this->varRestDensity()->setValue(Real(1000));
	}

	template<typename TDataType>
	IterativeFrictionSolver<TDataType>::~IterativeFrictionSolver()
	{
		mPosBuf1.clear();
		mPosBuf0.clear();
		mPositionOld.clear();
	}

	template <typename Real, typename Coord, typename Kernel>
	__global__ void IFS_PreventParticleInterpenetration(
		DArray<Coord> nextPos,
		DArray<Coord> oldPos,
		DArrayList<int> neighbors,
		Real h,	//smoothing length
		Real d,	//sampling distance
		Kernel weight,
		Real scale)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= nextPos.size()) return;

		Coord p_i = oldPos[pId];

		Real V = d * d * d;

		Real lamda_i = Real(0);
		Coord grad_ci(0);

		List<int>& list_i = neighbors[pId];
		int nbSize = list_i.size();

		Coord deltaPos_i = Coord(0);

		RandNumber rGen(pId);

		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = list_i[ne];
			Coord p_j = oldPos[j];
			Coord p_ij = p_j - p_i;
			Real r = p_ij.norm();

			if (r < d)
			{
				Real w = weight(r, h, scale) * V;

				Real x = EPSILON * rGen.Generate();
				Real y = EPSILON * rGen.Generate();
				Real z = EPSILON * rGen.Generate();

				p_ij += Coord(x, y, z);

				Coord N = p_ij.normalize();

				Coord deltaPos_j = N * (d - r) * w;

				atomicAdd(&nextPos[j].x, deltaPos_j.x);
				atomicAdd(&nextPos[j].y, deltaPos_j.y);
				atomicAdd(&nextPos[j].z, deltaPos_j.z);

				deltaPos_i -= deltaPos_j;
			}
		}

		atomicAdd(&nextPos[pId].x, deltaPos_i.x);
		atomicAdd(&nextPos[pId].y, deltaPos_i.y);
		atomicAdd(&nextPos[pId].z, deltaPos_i.z);
	}

	template <typename Real, typename Coord, typename Kernel>
	__global__ void IFS_ApplyFriction(
		DArray<Coord> newPos,
		DArray<Coord> tmpPos,
		DArray<Coord> oldPos,
		DArrayList<int> neighbors,
		Real h,
		Real d,
		Real dt,
		Kernel weight,
		Real scale)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= newPos.size()) return;

		Coord oldPos_i = oldPos[pId];
		Coord tmpPos_i = tmpPos[pId];
		Coord dPos_i = tmpPos_i - oldPos_i;

		Real V = d * d * d;

		Coord deltaT_i = Coord(0);

		List<int>& list_i = neighbors[pId];
		int nbSize = list_i.size();
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = list_i[ne];

			Coord oldPos_j = oldPos[j];
			Coord tmpPos_j = tmpPos[j];
			Coord dPos_j = tmpPos_j - oldPos_j;

			Real r = (tmpPos_j - tmpPos_i).norm();

			Coord N_ij = tmpPos_j - tmpPos_i;
			N_ij.normalize();

			Real dN = dPos_j.dot(N_ij);
			Coord t = dPos_j - dN * N_ij;

			Coord deltaT_j = 1.5 * t * weight(r, h, scale) * V;

			atomicAdd(&newPos[j].x, -deltaT_j.x);
			atomicAdd(&newPos[j].y, -deltaT_j.y);
			atomicAdd(&newPos[j].z, -deltaT_j.z);

			deltaT_i += deltaT_j;
		}

		atomicAdd(&newPos[pId].x, deltaT_i.x);
		atomicAdd(&newPos[pId].y, deltaT_i.y);
		atomicAdd(&newPos[pId].z, deltaT_i.z);
	}

	template <typename Real, typename Coord>
	__global__ void IFS_UpdatePosition(
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
	void IterativeFrictionSolver<TDataType>::compute()
	{
		int num = this->inPosition()->size();

		if (mPositionOld.size() != this->inPosition()->size())
			mPositionOld.resize(this->inPosition()->size());

		mPositionOld.assign(this->inPosition()->getData());

		if (mPosBuf0.size() != this->inPosition()->size())
			mPosBuf0.resize(this->inPosition()->size());

		if (mPosBuf1.size() != this->inPosition()->size())
			mPosBuf1.resize(this->inPosition()->size());

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
	void IterativeFrictionSolver<TDataType>::takeOneIteration()
	{
		Real dt = this->inTimeStep()->getValue();
		int num = this->inPosition()->size();

		mPosBuf0.assign(this->inPosition()->getData());

		cuZerothOrder(num, this->varKernelType()->currentKey(), this->mScalingFactor,
			IFS_PreventParticleInterpenetration,
			this->inPosition()->getData(),
			mPosBuf0,
			this->inNeighborIds()->getData(),
			this->inSmoothingLength()->getValue(),
			this->inSamplingDistance()->getValue());

		mPosBuf1.assign(this->inPosition()->getData());

		cuZerothOrder(num, this->varKernelType()->currentKey(), this->mScalingFactor,
			IFS_ApplyFriction,
			this->inPosition()->getData(),
			mPosBuf1,
			mPosBuf0,
			this->inNeighborIds()->getData(),
			this->inSmoothingLength()->getValue(),
			this->inSamplingDistance()->getValue(),
			dt);
	}

	template <typename Real, typename Coord>
	__global__ void IFS_UpdateVelocity(
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
	void IterativeFrictionSolver<TDataType>::updateVelocity()
	{
		int num = this->inPosition()->size();

		Real dt = this->inTimeStep()->getData();

		cuExecute(num, IFS_UpdateVelocity,
			this->inVelocity()->getData(),
			this->inPosition()->getData(),
			mPositionOld,
			dt);
	}

	DEFINE_CLASS(IterativeFrictionSolver);
}