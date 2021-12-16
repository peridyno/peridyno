#pragma once
#include "ParticleApproximation.h"

namespace dyno {
	/**
	 * @brief The standard summation density
	 * 
	 * @tparam TDataType 
	 */
	template<typename TDataType>
	class SummationDensity : public virtual ParticleApproximation<TDataType>
	{
		DECLARE_CLASS_1(SummationDensity, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		SummationDensity();
		~SummationDensity() override {};

		void compute() override;
	
	protected:
		void compute(DArray<Real>& rho);

	public:
		void compute(
			DArray<Real>& rho,
			DArray<Coord>& pos,
			DArrayList<int>& neighbors,
			Real smoothingLength,
			Real mass);

	public:
		DEF_VAR(Real, RestDensity, 1000, "Rest Density");
// 		DEF_VAR_IN(Real, SmoothingLength, "Indicating the smoothing length");
// 		DEF_VAR_IN(Real, SamplingDistance, "Indicating the initial sampling distance");

		///Define inputs
		/**
		 * @brief Particle positions
		 */
		DEF_ARRAY_IN(Coord, Position, DeviceType::GPU, "Particle position");

		/**
		 * @brief Neighboring particles
		 *
		 */
		DEF_ARRAYLIST_IN(int, NeighborIds, DeviceType::GPU, "Neighboring particles' ids");

		///Define outputs
		/**
		 * @brief Particle densities
		 */
		DEF_ARRAY_OUT(Real, Density, DeviceType::GPU, "Return the particle density");

	private:
		void calculateParticleMass();

		Real m_particle_mass;
		Real m_factor;
	};
}