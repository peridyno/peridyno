#include <cuda_runtime.h>
#include "SurfaceTension.h"
#include "Framework/MechanicalState.h"
#include "Topology/NeighborList.h"
#include "Kernel.h"

namespace dyno
{
	template<typename Real, typename Coord>
	__global__ void ST_ComputeSurfaceEnergy
	(
		DArray<Real> energyArr,
		DArray<Coord> posArr,
		NeighborList<int> neighbors,
		Real smoothingLength
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.size()) return;

		Real total_weight = Real(0);
		Coord dir_i(0);

		SmoothKernel<Real> kern;

		Coord pos_i = posArr[pId];
		int nbSize = neighbors.getNeighborSize(pId);
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = neighbors.getElement(pId, ne);
			Real r = (pos_i - posArr[j]).norm();

			if (r > EPSILON)
			{
				Real weight = -kern.Gradient(r, smoothingLength);
				total_weight += weight;
				dir_i += (posArr[j] - pos_i)*(weight / r);
			}
		}

		total_weight = total_weight < EPSILON ? 1.0f : total_weight;
		Real absDir = dir_i.norm() / total_weight;

		energyArr[pId] = absDir*absDir;
	}

	template<typename Real, typename Coord>
	__global__ void ST_ComputeSurfaceTension
	(
		DArray<Coord> velArr, 
		DArray<Real> energyArr, 
		DArray<Coord> posArr, 
		NeighborList<int> neighbors,
		Real smoothingLength,
		Real mass,
		Real restDensity,
		float dt
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.size()) return;

		Real Vref = mass / restDensity;

		float alpha = (float) 945.0f / (32.0f * (float)M_PI * smoothingLength * smoothingLength * smoothingLength);
		float ceof = 16000.0f * alpha;

		SmoothKernel<Real> kern;

		Coord F_i(0);
		Coord dv_pi(0);
		Coord pos_i = posArr[pId];
		int nbSize = neighbors.getNeighborSize(pId);
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = neighbors.getElement(pId, ne);
			float r = (pos_i - posArr[j]).norm();

			if (r > EPSILON)
			{
				Coord temp = Vref*Vref*kern.Gradient(r, smoothingLength)*(posArr[j] - pos_i) * (1.0f / r);
				Coord dv_ij = dt * ceof*1.0f*(energyArr[pId])*temp / mass;
				F_i += dv_ij;

// 				atomicAdd(&velArr[j].x, dv_ij.x);
// 				atomicAdd(&velArr[j].y, dv_ij.y);
// 				atomicAdd(&velArr[j].z, dv_ij.z);
			}
		}
		velArr[pId] -= F_i;

// 		atomicAdd(&velArr[pId].x, -F_i.x);
// 		atomicAdd(&velArr[pId].y, -F_i.y);
// 		atomicAdd(&velArr[pId].z, -F_i.z);
	}

	template<typename TDataType>
	SurfaceTension<TDataType>::SurfaceTension()
		: ForceModule()
		, m_posID(MechanicalState::position())
		, m_velID(MechanicalState::velocity())
		, m_neighborhoodID(MechanicalState::particle_neighbors())
		, m_intensity(Real(1))
		, m_soothingLength(Real(0.0125))
	{

	}

	template<typename TDataType>
	bool SurfaceTension<TDataType>::execute()
	{
// 		m_energy = DeviceBuffer<Real>::create(num);
// 
// 		DArray<Coord>* posArr = m_parent->GetNewPositionBuffer()->getDataPtr();
// 		DArray<Coord>* velArr = m_parent->GetNewVelocityBuffer()->getDataPtr();
// 		DArray<Attribute>* attArr = m_parent->GetAttributeBuffer()->getDataPtr();
// 		float dt = m_parent->getDt();
// 
// 		DArray<SPHNeighborList>* neighborArr = m_parent->GetNeighborBuffer()->getDataPtr();
// 
// 		DArray<Real>* energy = m_energy->getDataPtr();
// 
// 		Real mass = m_parent->GetParticleMass();
// 		Real smoothingLength = m_parent->GetSmoothingLength();
// 		Real restDensity = m_parent->GetRestDensity();
// 
// 		uint pDims = cudaGridSize(posArr->size(), BLOCK_SIZE);
// 		ST_ComputeSurfaceEnergy <Real, Coord> << < pDims, BLOCK_SIZE >> > (*energy, *posArr, *neighborArr, smoothingLength);
// 		ST_ComputeSurfaceTension <Real, Coord> << < pDims, BLOCK_SIZE >> > (*velArr, *energy, *posArr, *attArr, *neighborArr, smoothingLength, mass, restDensity, dt);

		return true;
	}

	template<typename TDataType>
	bool SurfaceTension<TDataType>::applyForce()
	{
		return true;
	}

	DEFINE_CLASS(SurfaceTension);
}