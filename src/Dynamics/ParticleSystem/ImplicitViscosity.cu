#include "ImplicitViscosity.h"
#include "Framework/Node.h"
#include "Topology/FieldNeighbor.h"

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
		NeighborList<int> neighbors,
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
		int nbSize = neighbors.getNeighborSize(pId);
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = neighbors.getElement(pId, ne);
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

		attachField(&m_viscosity, "viscosity", "The viscosity of the fluid!", false);
		attachField(&m_smoothingLength, "smoothing_length", "The smoothing length in SPH!", false);
		attachField(&m_position, "position", "Storing the particle positions!", false);
		attachField(&m_velocity, "velocity", "Storing the particle velocities!", false);
		attachField(&m_neighborhood, "neighborhood", "Storing neighboring particles' ids!", false);
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
		int num = m_position.getElementCount();
		if (num > 0)
		{
			uint pDims = cudaGridSize(num, BLOCK_SIZE);

			m_velOld.resize(num);
			m_velBuf.resize(num);

			Real vis = m_viscosity.getValue();
			Real dt = getParent()->getDt();
			m_velOld.assign(m_velocity.getValue());
			for (int t = 0; t < m_maxInteration; t++)
			{
				m_velBuf.assign(m_velocity.getValue());
				cuExecute(num, K_ApplyViscosity,
					m_velocity.getValue(),
					m_position.getValue(),
					m_neighborhood.getValue(),
					m_velOld,
					m_velBuf,
					vis,
					m_smoothingLength.getValue(),
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


}