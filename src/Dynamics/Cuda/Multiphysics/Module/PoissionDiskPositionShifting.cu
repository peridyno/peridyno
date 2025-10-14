#include "PoissionDiskPositionShifting.h"

#include "ParticleSystem/Module/SummationDensity.h"

namespace dyno
{
	template<typename TDataType>
	PoissionDiskPositionShifting<TDataType>::PoissionDiskPositionShifting()
		: ParticleApproximation<TDataType>()
	{
		this->varIterationNumber()->setValue(50);
		this->varRestDensity()->setValue(Real(1000));

		mSummation = std::make_shared<SummationDensity<TDataType>>();

		this->inSmoothingLength()->connect(mSummation->inSmoothingLength());
		this->inSamplingDistance()->connect(mSummation->inSamplingDistance());
		this->inPosition()->connect(mSummation->inPosition());
		this->inNeighborIds()->connect(mSummation->inNeighborIds());

		mSummation->outDensity()->connect(this->outDensity());

		this->inVelocity()->tagOptional(true);
	}

	template<typename TDataType>
	PoissionDiskPositionShifting<TDataType>::~PoissionDiskPositionShifting()
	{
		mPosBuf.clear();
		mPosOld.clear();
	}

	template <typename Real, typename Coord, typename Kernel>
	__global__ void SSDS_OneJacobiStep(
		DArray<Coord> posNext,
		DArray<Coord> posBuf,
		DArray<Coord> posOld,
		DArray<Real> diagnals,
		DArray<Real> rho,
		DArrayList<int> neighbors,
		Real smoothingLength,
		Real samplingDistance,
		Real kappa,
		Real dt,
		Kernel gradient,
		Real scale)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posNext.size()) return;

		Real rho_0 = Real(1000);

		Real rho_i = rho[pId];
		rho_i = rho_i > rho_0 ? rho_i : rho_0;
		
		Real A = kappa * dt * dt / rho_0;
		Real C_plus = rho_i / rho_0;

		Real C_minus = Real(-1);

		Coord pos_i = posBuf[pId];

		List<int>& list_i = neighbors[pId];
		int nbSize = list_i.size();

		Real a_i = Real(1);
		Coord posAcc_i = posOld[pId];
		Coord deltaPos_i = Coord(0);
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = list_i[ne];
			Coord pos_j = posBuf[j];
			Real r = (pos_i - pos_j).norm();

			if (r > EPSILON)
			{
				Real a_ij = A * gradient(r, smoothingLength, scale) * (1.0f / r);

				posAcc_i += C_minus * a_ij * pos_j + C_plus * a_ij * (pos_j - pos_i);

				Coord posAcc_ji = C_minus * a_ij * (pos_i) + C_plus * a_ij * (pos_i - pos_j);
				Real a_ji = C_minus * a_ij;

				atomicAdd(&posNext[j][0], posAcc_ji[0]);
				atomicAdd(&posNext[j][1], posAcc_ji[1]);
				atomicAdd(&posNext[j][2], posAcc_ji[2]);
				atomicAdd(&diagnals[j], a_ji);

				a_i += C_minus * a_ij;
			}
		}

		atomicAdd(&posNext[pId][0], posAcc_i[0]);
		atomicAdd(&posNext[pId][1], posAcc_i[1]);
		atomicAdd(&posNext[pId][2], posAcc_i[2]);

		atomicAdd(&diagnals[pId], a_i);

	}

	template <typename Real, typename Coord>
	__global__ void SSDS_ComputeNewPos(
		DArray<Coord> posNext,
		DArray<Coord> posPre,
		DArray<Coord> posOld,
		DArray<Real> density,
		Real samplingDistance,
		Real dt,
		DArray<Real> diagnals)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posNext.size()) return;

		Coord grad = diagnals[pId] * posPre[pId] - posNext[pId];

		Coord posNext_i = posNext[pId] / diagnals[pId];

		Coord dx = posNext_i - posPre[pId];
		Coord ds = posPre[pId] - posOld[pId];

		const Real rho_0 = Real(1000);
		const Real m = rho_0 * samplingDistance * samplingDistance * samplingDistance;
		const Real C = m / (dt * dt);

		Real rho_i = density[pId];
		rho_i = rho_i > rho_0 ? rho_i : rho_0;
		Real lambda_i = rho_i / rho_0;
		Real square = C * dx.dot(grad);

		Real energy = 0.5 * C * ds.dot(ds) + 0.5 * (lambda_i - 1) * (lambda_i - 1);


		Real stepsize = Real(1);
		if (rho_i > rho_0)
		{
			Real eps = C * samplingDistance * samplingDistance * 0.01 * 0.01;
			stepsize = energy / (-square + eps);
			stepsize = stepsize > Real(1) ? Real(1) : stepsize;
		}

		posNext[pId] = posPre[pId] + stepsize * dx;
	}


	template <typename Real, typename Coord>
	__global__ void SSDS_PredictNewPos(
		DArray<Coord> posNext,
		DArray<Coord> posOld,
		DArray<Coord> velocity,
		Real dt
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posNext.size()) return;

		posNext[pId] = posOld[pId] + dt * velocity[pId];
	}

	template<typename TDataType>
	void PoissionDiskPositionShifting<TDataType>::compute()
	{

		updatePosition();

		//updateVelocity();
	}

	template<typename TDataType>
	void PoissionDiskPositionShifting<TDataType>::updatePosition()
	{
		int num = this->inPosition()->size();
		Real dt = this->inDelta()->getValue();
		auto& inPos = this->inPosition()->getData();

		if (mPosOld.size() != num)
			mPosOld.resize(num);

		if (mPosBuf.size() != num)
			mPosBuf.resize(num);

		if (mDiagnals.size() != num)
			mDiagnals.resize(num);

		if (this->inVelocity()->size() != num)
		{
			this->inVelocity()->allocate();
			this->inVelocity()->resize(num);
		}
		mPosOld.assign(inPos);


		int itNum = this->varIterationNumber()->getValue();
		int it = 0;
		while (it < itNum)
		{
			mSummation->varRestDensity()->setValue(this->varRestDensity()->getValue());
			mSummation->varKernelType()->setCurrentKey(this->varKernelType()->currentKey());
			mSummation->update();

			mPosBuf.assign(inPos);

			inPos.reset();

			mDiagnals.reset();


			cuFirstOrder(num, this->varKernelType()->getDataPtr()->currentKey(), this->mScalingFactor,
				SSDS_OneJacobiStep,
				inPos,
				mPosBuf,
				mPosOld,
				mDiagnals,
				mSummation->outDensity()->getData(),
				this->inNeighborIds()->getData(),
				this->inSmoothingLength()->getValue(),
				this->inSamplingDistance()->getValue(),
				this->varKappa()->getValue(),
				dt);

				cuExecute(num,
					SSDS_ComputeNewPos,
					inPos,
					mPosBuf,
					mPosOld,
					mSummation->outDensity()->getData(),
					this->inSamplingDistance()->getValue(),
					dt,
					mDiagnals);

			it++;
		}
	}

	//template <typename Real, typename Coord>
	//__global__ void SSDS_UpdateVelocity(
	//	DArray<Coord> velArr,
	//	DArray<Coord> curPos,
	//	DArray<Coord> prePos,
	//	Real dt)
	//{
	//	int pId = threadIdx.x + (blockIdx.x * blockDim.x);
	//	if (pId >= velArr.size()) return;

	//	velArr[pId] += (curPos[pId] - prePos[pId]) / dt;
	//}

	//template<typename TDataType>
	//void PoissionDiskPositionShifting<TDataType>::updateVelocity()
	//{
	//	int num = this->inPosition()->size();

	//	Real dt = this->inDelta()->getData();

	//	cuExecute(num, SSDS_UpdateVelocity,
	//		this->inVelocity()->getData(),
	//		this->inPosition()->getData(),
	//		mPosOld,
	//		dt);
	//}

	DEFINE_CLASS(PoissionDiskPositionShifting);
}