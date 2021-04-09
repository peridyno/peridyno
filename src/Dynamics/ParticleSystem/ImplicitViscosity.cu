#include "ImplicitViscosity.h"
#include "Framework/Node.h"

namespace dyno
{
	template<typename Real>
	__device__ Real VB_VisWeight(const Real r, const Real h)
	{
		Real q = r / h;
		if (q > 1.0f) return 0.0;
		else {
			const Real d = 1.0f - q;
			const Real RR = h*h;
			return 45.0f / (13.0f * (Real)M_PI * RR *h) *d;
		}
	}

	template<typename Real, typename Coord>
	__global__ void K_ApplyViscosity(
		DArray<Coord> velNew,
		DArray<Coord> posArr,
		DArrayList<int> neighbors,
		DArray<Coord> velOld,
		DArray<Coord> velArr,
		Real viscosity,
		Real smoothingLength,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.size()) return;

		Real r;
		Coord dv_i(0);
		Coord pos_i = posArr[pId];
		Coord vel_i = velArr[pId];
		Real totalWeight = 0.0f;

		List<int>& list_i = neighbors[pId];
		int nbSize = list_i.size();
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = list_i[ne];
			r = (pos_i - posArr[j]).norm();

			if (r > EPSILON)
			{
				Real weight = VB_VisWeight(r, smoothingLength);
				totalWeight += weight;
				dv_i += weight * velArr[j];
			}
		}

		Real b = dt*viscosity / smoothingLength;

		b = totalWeight < EPSILON ? 0.0f : b;

		totalWeight = totalWeight < EPSILON ? 1.0f : totalWeight;

		dv_i /= totalWeight;

		velNew[pId] = velOld[pId] / (1.0f + b) + dv_i*b / (1.0f + b);
	}

	template<typename Real, typename Coord>
	__global__ void VB_UpdateVelocity(
		DArray<Coord> velArr, 
		DArray<Coord> dVel)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= velArr.size()) return;

		velArr[pId] = dVel[pId];
	}

	template<typename TDataType>
	ImplicitViscosity<TDataType>::ImplicitViscosity()
		:ConstraintModule()
		, m_maxInteration(5)
	{
		m_viscosity.setValue(Real(0.05));
		m_smoothingLength.setValue(Real(0.011));
	}

	template<typename TDataType>
	ImplicitViscosity<TDataType>::~ImplicitViscosity()
	{
		m_velOld.clear();
		m_velBuf.clear();
	}

	template<typename TDataType>
	bool ImplicitViscosity<TDataType>::constrain()
	{
		auto& nbrIds = this->inNeighborIds()->getData();

		int num = m_position.getElementCount();
		if (num > 0)
		{
			uint pDims = cudaGridSize(num, BLOCK_SIZE);

			m_velOld.resize(num);
			m_velBuf.resize(num);

			Real vis = m_viscosity.getData();
			Real dt = getParent()->getDt();
			m_velOld.assign(m_velocity.getData());
			for (int t = 0; t < m_maxInteration; t++)
			{
				m_velBuf.assign(m_velocity.getData());
				cuExecute(num, 
					K_ApplyViscosity,
					m_velocity.getData(),
					m_position.getData(),
					nbrIds,
					m_velOld,
					m_velBuf,
					vis,
					m_smoothingLength.getData(),
					dt);
			}

			return true;
		}
	}

	template<typename TDataType>
	bool ImplicitViscosity<TDataType>::initializeImpl()
	{
		return true;
	}

	template<typename TDataType>
	void ImplicitViscosity<TDataType>::setIterationNumber(int n)
	{
		m_maxInteration = n;
	}

	template<typename TDataType>
	void ImplicitViscosity<TDataType>::setViscosity(Real mu)
	{
		m_viscosity.setValue(mu);
	}

	DEFINE_CLASS(ImplicitViscosity);
}