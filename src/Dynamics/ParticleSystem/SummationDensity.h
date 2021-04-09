#pragma once
#include "Framework/ModuleCompute.h"

namespace dyno {
	/**
	 * @brief The standard summation density
	 * 
	 * @tparam TDataType 
	 */
	template<typename TDataType>
	class SummationDensity : public virtual ComputeModule
	{
		DECLARE_CLASS_1(SummationDensity, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		SummationDensity();
		~SummationDensity() override {};

		void compute() override;
	
	protected:
		void calculateScalingFactor();
		void calculateParticleMass();

		void compute(DArray<Real>& rho);

		void compute(
			DArray<Real>& rho,
			DArray<Coord>& pos,
			DArrayList<int>& neighbors,
			Real smoothingLength,
			Real mass);

	public:
		DEF_EMPTY_VAR(RestDensity, Real, "Rest Density");
		DEF_EMPTY_VAR(SmoothingLength, Real, "Indicating the smoothing length");
		DEF_EMPTY_VAR(SamplingDistance, Real, "Indicating the initial sampling distance");

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
		Real m_particle_mass;
		Real m_factor;
	};
}