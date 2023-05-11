#include "SemiAnalyticalSummationDensity.h"

#include "IntersectionArea.h"

namespace dyno
{
	IMPLEMENT_TCLASS(SemiAnalyticalSummationDensity, TDataType)

	template<typename TDataType>
	SemiAnalyticalSummationDensity<TDataType>::SemiAnalyticalSummationDensity()
		: ParticleApproximation<TDataType>()
		, m_factor(Real(1))
	{
		this->varRestDensity()->setValue(Real(1000));//1000

		auto callback = std::make_shared<FCallBackFunc>(
			std::bind(&SemiAnalyticalSummationDensity<TDataType>::calculateParticleMass, this));

		this->varRestDensity()->attach(callback);
		this->inSamplingDistance()->attach(callback);
	}

	template<typename TDataType>
	void SemiAnalyticalSummationDensity<TDataType>::compute()
	{
		int p_num = this->inPosition()->getDataPtr()->size();
		int n_num = this->inNeighborIds()->getDataPtr()->size();
		int t_num = this->inTriangleInd()->getDataPtr()->size();
		int tn_num = this->inNeighborTriIds()->getDataPtr()->size();

		//printf("tn_num:  %d\n", tn_num);

		if (p_num != n_num ) {
			Log::sendMessage(Log::Error, "The input array sizes of DensitySummation are not compatible!");
			return;
		}

		if (this->outDensity()->size() != p_num) {
			this->outDensity()->resize(p_num);
		}

		compute(
			this->outDensity()->getData(),
			this->inPosition()->getData(),
			this->inTriangleInd()->getData(),
			this->inTriangleVer()->getData(),
			this->inNeighborIds()->getData(),
			this->inNeighborTriIds()->getData(),
			this->inSmoothingLength()->getData(),
			m_particle_mass,
			this->inSamplingDistance()->getData());
	}


	template<typename TDataType>
	void SemiAnalyticalSummationDensity<TDataType>::compute(DArray<Real>& rho)
	{
		compute(
			rho,
			this->inPosition()->getData(),
			this->inTriangleInd()->getData(),
			this->inTriangleVer()->getData(),
			this->inNeighborIds()->getData(),
			this->inNeighborTriIds()->getData(),
			this->inSmoothingLength()->getData(),
			m_particle_mass,
			this->inSamplingDistance()->getData());
	}

	template<typename Real, typename Coord, typename Kernel>
	__global__ void SD_ComputeDensity(
		DArray<Real> rhoArr,
		DArray<Coord> posArr,
		DArrayList<int> neighbors,
		Real smoothingLength,
		Real mass,
		Real sampling_distance,
		Kernel kernel,
		Real scale)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.size()) return;

		SpikyKernel<Real> kern;
		Real r;
		Real rho_i = Real(0);
		Real rho_tmp(0);
		Coord pos_i = posArr[pId];

		List<int>& list_i = neighbors[pId];

		int nbSize = list_i.size();
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = list_i[ne];
			r = (pos_i - posArr[j]).norm();
			rho_i += mass * kernel(r, smoothingLength, scale);
		}

		rhoArr[pId] = rho_i;
	}

	template<typename Real, typename Coord, typename Kernel>
	__global__ void SD_BoundaryIntegral(
		DArray<Real> rhoArr,
		DArray<Coord> posArr,
		DArray<TopologyModule::Triangle> Tri,
		DArray<Coord> positionTri,
		DArrayList<int> neighborsTri,
		Real smoothingLength,
		Real mass,
		Kernel kernel,
		Real scale)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= rhoArr.size()) return;

		Real rho_i = Real(0);
		Coord pos_i = posArr[pId];

		List<int>& nbrTriIds_i = neighborsTri[pId];
		int nbSizeTri = nbrTriIds_i.size();
		if (nbSizeTri > 0)
		{
			for (int ne = 0; ne < nbSizeTri; ne++)
			{
				int j = nbrTriIds_i[ne];


				Triangle3D t3d(positionTri[Tri[j][0]], positionTri[Tri[j][1]], positionTri[Tri[j][2]]);
				Plane3D PL(positionTri[Tri[j][0]], t3d.normal());
				Point3D p3d(pos_i);
				Point3D nearest_pt = p3d.project(PL);
				Real r = (nearest_pt.origin - pos_i).norm();

				float d = p3d.distance(PL);
				d = abs(d);
				if (smoothingLength - d > EPSILON&& smoothingLength* smoothingLength - d * d > EPSILON&& d > EPSILON)
				{

					Real a_ij =
						kernel(r, smoothingLength, scale)
						* 2.0 * (M_PI) * (1 - d / smoothingLength)
						* calculateIntersectionArea(p3d, t3d, smoothingLength)
						/ ((M_PI) * (smoothingLength * smoothingLength - d * d));
					rho_i += mass * a_ij;

				}
			}
			//printf("Boundary: %f \n", rho_i);
		}

		rhoArr[pId] += rho_i;
	}

	template<typename TDataType>
	void SemiAnalyticalSummationDensity<TDataType>::compute(
		DArray<Real>& rho, 
		DArray<Coord>& pos,
		DArray<TopologyModule::Triangle>& Tri,
		DArray<Coord>& positionTri,
		DArrayList<int>& neighbors,
		DArrayList<int>& neighborsTri,
		Real smoothingLength,
		Real mass,
		Real sampling_distance)
	{
		cuZerothOrder(rho.size(), this->varKernelType()->getDataPtr()->currentKey(), mScalingFactor,
			SD_ComputeDensity,
			rho,
			pos,
			neighbors,
			smoothingLength,
			mass,
			sampling_distance);

		if (neighborsTri.size() > 0)
		{
			cuIntegral(rho.size(), this->varKernelType()->getDataPtr()->currentKey(), mScalingFactor,
				SD_BoundaryIntegral,
				rho,
				pos,
				Tri,
				positionTri,
				neighborsTri,
				smoothingLength,
				mass);
		}
	}

	template<typename TDataType>
	void SemiAnalyticalSummationDensity<TDataType>::calculateParticleMass()
	{
		Real rho_0 = this->varRestDensity()->getData();
		Real d = this->inSamplingDistance()->getData();

		m_particle_mass = d*d*d*rho_0;
	}

	DEFINE_CLASS(SemiAnalyticalSummationDensity);
}