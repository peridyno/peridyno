#pragma once
#include "ParticleSystem/Module/ParticleApproximation.h"

#include "Module/TopologyModule.h"

namespace dyno {
	/**
	 * @brief The standard summation density
	 * 
	 * @tparam TDataType 
	 */
	typedef typename TopologyModule::Triangle Triangle;

	template<typename TDataType>
	class SemiAnalyticalSummationDensity : public virtual ParticleApproximation<TDataType>
	{
		DECLARE_TCLASS(SemiAnalyticalSummationDensity, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		SemiAnalyticalSummationDensity();
		~SemiAnalyticalSummationDensity() override {};

		void compute() override;
	
	protected:
		void calculateParticleMass();

		void compute(DArray<Real>& rho);

	public:
		void compute(
			DArray<Real>& rho,
			DArray<Coord>& pos,
			DArray<TopologyModule::Triangle>& Tri,
			DArray<Coord>& positionTri,
			DArrayList<int>& neighbors,
			DArrayList<int>& neighborsTri,
			Real smoothingLength,
			Real mass,
			Real sampling_distance);

	public:
		DEF_VAR(Real, RestDensity, 1000, "Rest Density");

		DEF_ARRAY_IN(Coord, Position, DeviceType::GPU, "Particle position");
		DEF_ARRAYLIST_IN(int, NeighborIds, DeviceType::GPU, "Neighboring particles' ids");
		DEF_ARRAYLIST_IN(int, NeighborTriIds, DeviceType::GPU, "triangle neighbors");//
		DEF_ARRAY_IN(Triangle, TriangleInd, DeviceType::GPU, "triangle_index");
		DEF_ARRAY_IN(Coord, TriangleVer, DeviceType::GPU, "triangle_vertex");

		DEF_ARRAY_OUT(Real, Density, DeviceType::GPU, "Return the particle density");

	private:
		Real m_particle_mass;
		Real m_factor;
	};
}