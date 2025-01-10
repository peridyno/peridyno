#include "SemiImplicitDensitySolver.h"

#include "SummationDensity.h"

namespace dyno
{
	template<typename TDataType>
	SemiImplicitDensitySolver<TDataType>::SemiImplicitDensitySolver()
		: ParticleApproximation<TDataType>()
	{
		this->varIterationNumber()->setValue(3);
		this->varRestDensity()->setValue(Real(1000));

		this->inAttribute()->tagOptional(true);

		mSummation = std::make_shared<SummationDensity<TDataType>>();

		this->inSmoothingLength()->connect(mSummation->inSmoothingLength());
		this->inSamplingDistance()->connect(mSummation->inSamplingDistance());
		this->inPosition()->connect(mSummation->inPosition());
		this->inNeighborIds()->connect(mSummation->inNeighborIds());

		mSummation->outDensity()->connect(this->outDensity());
	}

	template<typename TDataType>
	SemiImplicitDensitySolver<TDataType>::~SemiImplicitDensitySolver()
	{
		mPosBuf.clear();
		mPosOld.clear();
	}

#define CONSERVE_MOMETNUM
#define CASE_0
//#define CASE_1
//#define CASE_2

#define CHEBYSHEV_ACCELERATION
//#define ANDERSON_ACCELERATION

#ifdef CHEBYSHEV_ACCELERATION
	template<typename Real, typename Coord>
	__global__ void SSDS_Blend(
		DArray<Coord> pos,
		DArray<Coord> pos_k,
		DArray<Coord> pos_k_minus,
		Real omega)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= pos.size()) return;

		pos[pId] = omega * (pos_k[pId] - pos_k_minus[pId]) + pos_k_minus[pId];
	}
#endif

#ifdef ANDERSON_ACCELERATION
	template<typename Coord>
	__global__ void SSDS_CalculateG(
		DArray<Coord> g,
		DArray<Coord> x,
		DArray<Coord> fx)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= g.size()) return;

		g[pId] = fx[pId] - x[pId];
	}

	template<typename Real, typename Coord>
	__global__ void SSDS_CalculateLSM(
		DArray<Real> denom,
		DArray<Real> numer,
		DArray<Coord> g_k_minus,
		DArray<Coord> g_k)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= denom.size()) return;

		Coord dg = g_k[pId] - g_k_minus[pId];

		denom[pId] = dg.normSquared();
		numer[pId] = dg.dot(g_k[pId]);
	}

	template<typename Real, typename Coord>
	__global__ void SSDS_Blend(
		DArray<Coord> pos,
		DArray<Coord> pos_k,
		DArray<Coord> pos_k_minus,
		Real omega)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= pos.size()) return;

		pos[pId] = pos_k[pId] + omega * (pos_k_minus[pId] - pos_k[pId]);
	}
#endif


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
				
#ifdef CONSERVE_MOMETNUM
				posAcc_i += C_minus * a_ij * pos_j + C_plus * a_ij * (pos_j - pos_i);

				Coord posAcc_ji = C_minus * a_ij * (pos_i) + C_plus * a_ij * (pos_i - pos_j);
				Real a_ji = C_minus * a_ij;

				atomicAdd(&posNext[j][0], posAcc_ji[0]);
				atomicAdd(&posNext[j][1], posAcc_ji[1]);
				atomicAdd(&posNext[j][2], posAcc_ji[2]);
				atomicAdd(&diagnals[j], a_ji);
#else
				posAcc_i += C_minus * a_ij * pos_j + C_plus * a_ij * (pos_j - pos_i);
#endif // CONSERVE_MOMETNUM
				a_i += C_minus * a_ij;
			}
		}

#ifdef CONSERVE_MOMETNUM
		//To ensure momentum conservation
		atomicAdd(&posNext[pId][0], posAcc_i[0]);
		atomicAdd(&posNext[pId][1], posAcc_i[1]);
		atomicAdd(&posNext[pId][2], posAcc_i[2]);

		atomicAdd(&diagnals[pId], a_i);
#else
		posNext[pId] = posAcc_i / a_i;;
#endif // CONSERVE_MOMETNUM
	}

	template <typename Real, typename Coord>
	__global__ void SSDS_ComputeNewPos(
		DArray<Coord> posNext,
		DArray<Real> diagnals)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posNext.size()) return;

		posNext[pId] = posNext[pId] / diagnals[pId];
	}

	template <typename Real, typename Coord>
	__global__ void SSDS_ComputeNewPos(
		DArray<Coord> posNext,
		DArray<Coord> posOld,
		DArray<Attribute> attributes,
		DArray<Real> diagnals)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posNext.size()) return;

		posNext[pId] = attributes[pId].isDynamic() ? posNext[pId] / diagnals[pId] : posOld[pId];
	}

	template<typename TDataType>
	void SemiImplicitDensitySolver<TDataType>::compute()
	{
		updatePosition();

		updateVelocity();
	}


	template<typename TDataType>
	void SemiImplicitDensitySolver<TDataType>::updatePosition()
	{
		int num = this->inPosition()->size();
		Real dt = this->inTimeStep()->getValue();
		auto& inPos = this->inPosition()->getData();

		if (mPosOld.size() != num)
			mPosOld.resize(num);

		if (mPosBuf.size() != num)
			mPosBuf.resize(num);

		if (mDiagnals.size() != num)
			mDiagnals.resize(num);

		mPosOld.assign(inPos);

		int itNum = this->varIterationNumber()->getValue();
		int it = 0;
		while (it < itNum)
		{
			mSummation->varRestDensity()->setValue(this->varRestDensity()->getValue());
			mSummation->varKernelType()->setCurrentKey(this->varKernelType()->currentKey());
			mSummation->update();

			mPosBuf.assign(inPos);
#ifdef CONSERVE_MOMETNUM
			inPos.reset();
#endif
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

#ifdef CONSERVE_MOMETNUM
			if (this->inAttribute()->isEmpty())
			{
				cuExecute(num,
					SSDS_ComputeNewPos,
					inPos,
					mDiagnals);
			}
			else
			{
				cuExecute(num,
					SSDS_ComputeNewPos,
					inPos,
					mPosOld,
					this->inAttribute()->constData(),
					mDiagnals);
			}
			
#endif

			it++;
		}
	}

	template <typename Real, typename Coord>
	__global__ void SSDS_UpdateVelocity(
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
	void SemiImplicitDensitySolver<TDataType>::updateVelocity()
	{
		int num = this->inPosition()->size();

		Real dt = this->inTimeStep()->getData();

		cuExecute(num, SSDS_UpdateVelocity,
			this->inVelocity()->getData(),
			this->inPosition()->getData(),
			mPosOld,
			dt);
	}

	DEFINE_CLASS(SemiImplicitDensitySolver);
}