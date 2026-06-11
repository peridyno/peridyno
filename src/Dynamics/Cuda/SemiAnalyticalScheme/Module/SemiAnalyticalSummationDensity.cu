#include "SemiAnalyticalSummationDensity.h"

#include "IntersectionArea.h"

#include "Collision/Distance3D.h"

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
		auto ts = this->inTriangleSet()->constDataPtr();
		auto& triVertex = ts->getPoints();
		auto& triIndex = ts->triangleIndices();

		int p_num = this->inPosition()->getDataPtr()->size();
		int n_num = this->inNeighborIds()->getDataPtr()->size();
		int t_num = triIndex.size();
		int tn_num = this->inNeighborTriIds()->getDataPtr()->size();

		//printf("tn_num:  %d\n", tn_num);

		if (p_num != n_num ) {
			Log::sendMessage(Log::Error, "The input array sizes of DensitySummation are not compatible!");
			return;
		}

		if (this->outDensity()->size() != p_num) {
			this->outDensity()->resize(p_num);
			this->outBoundaryDensity()->resize(p_num);
		}

		compute(
			this->outDensity()->getData(),
			this->outBoundaryDensity()->getData(),
			this->inPosition()->getData(),
			triIndex,
			triVertex,
			this->inNeighborIds()->getData(),
			this->inNeighborTriIds()->getData(),
			this->inSmoothingLength()->getData(),
			m_particle_mass,
			this->varRestDensity()->getValue(),
			this->inSamplingDistance()->getData());
	}


	template<typename TDataType>
	void SemiAnalyticalSummationDensity<TDataType>::compute(
		DArray<Real>& rho,
		DArray<Real>& rhoBoundary)
	{
		auto ts = this->inTriangleSet()->constDataPtr();
		auto& triVertex = ts->getPoints();
		auto& triIndex = ts->triangleIndices();

		compute(
			rho,
			rhoBoundary,
			this->inPosition()->getData(),
			triIndex,
			triVertex,
			this->inNeighborIds()->getData(),
			this->inNeighborTriIds()->getData(),
			this->inSmoothingLength()->getData(),
			m_particle_mass,
			this->varRestDensity()->getValue(),
			this->inSamplingDistance()->getData());
	}

	template<typename Real, typename Coord, typename Kernel>
	__global__ void SASD_ComputeDensity(
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
	__global__ void SASD_BoundaryIntegral(
		DArray<Real> rhoInterior,
		DArray<Real> rhoBoundary,
		DArray<Coord> posArr,
		DArray<Topology::Triangle> triIndices,
		DArray<Coord> triVertices,
		DArrayList<int> triNeighbors,
		Real smoothingLength,
		Real rho_0,
		Kernel kernel,
		Real scale)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= rhoInterior.size()) return;

		Real weight_i = Real(0);
		Coord pos_i = posArr[pId];

		Real threshold = Real(0.1);

		List<int>& nbrTriIds_i = triNeighbors[pId];
		int nbSizeTri = nbrTriIds_i.size();

		bool onPositiveSide = true;

		if (nbSizeTri > 0)
		{
			for (int ne = 0; ne < nbSizeTri; ne++)
			{
				int j = nbrTriIds_i[ne];

				Triangle3D t3d(triVertices[triIndices[j][0]], triVertices[triIndices[j][1]], triVertices[triIndices[j][2]]);
				Plane3D plane(triVertices[triIndices[j][0]], t3d.normal());
				Point3D p3d(pos_i);
				Real tri_d = p3d.distance(t3d);
				Point3D nearest_pt = p3d.project(plane);

				Coord d_n = pos_i - nearest_pt.origin;

				Real r = d_n.norm();

				d_n = r > EPSILON ? d_n / r : t3d.normal();

				Real d = p3d.distance(plane);

				Real A_0 = ((M_PI) * (smoothingLength * smoothingLength - d * d));
				A_0 = A_0 < EPSILON ? EPSILON : A_0;
				Real omega_0 = 2 * M_PI * (1 - d / smoothingLength);
				Real omega = abs(d_n.dot(t3d.normal())) * calculateIntersectionArea(p3d, t3d, smoothingLength) * omega_0 / A_0;

				Real w_ij = kernel(r, smoothingLength, 1) * omega;
				weight_i += w_ij;
			}

			ProjectedPoint3D<Real> p3d;

			bool valid = calculateSignedDistance2TriangleSet(p3d, pos_i, triVertices, triIndices, nbrTriIds_i);

			onPositiveSide = p3d.signed_distance > 0 ? true : false;
		}

		//If on the negative side, calculate the compensate
		Real rho_i = onPositiveSide ? rho_0 * weight_i : rho_0 * (1 - weight_i);

		rhoBoundary[pId] = rho_i;
		rhoInterior[pId] += rho_i;
	}

	template<typename TDataType>
	void SemiAnalyticalSummationDensity<TDataType>::compute(
		DArray<Real>& rho, 
		DArray<Real>& rhoBoundary,
		DArray<Coord>& pos,
		DArray<Topology::Triangle>& Tri,
		DArray<Coord>& positionTri,
		DArrayList<int>& neighbors,
		DArrayList<int>& neighborsTri,
		Real smoothingLength,
		Real mass,
		Real restDensity,
		Real sampling_distance)
	{
		cuZerothOrder(rho.size(), this->varKernelType()->getDataPtr()->currentKey(), mScalingFactor,
			SASD_ComputeDensity,
			rho,
			pos,
			neighbors,
			smoothingLength,
			mass,
			sampling_distance);
		if (neighborsTri.size() > 0)
		{
			cuIntegral(rho.size(), this->varKernelType()->getDataPtr()->currentKey(), mScalingFactor,
				SASD_BoundaryIntegral,
				rho,
				rhoBoundary,
				pos,
				Tri,
				positionTri,
				neighborsTri,
				smoothingLength,
				restDensity);
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