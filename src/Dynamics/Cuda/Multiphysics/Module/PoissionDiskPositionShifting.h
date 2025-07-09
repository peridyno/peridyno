
#pragma once
#include "ParticleSystem/Module/ParticleApproximation.h"

namespace dyno {
	/*
	*@brief The iterative solver for obtaining the Poisson-disk distribution on the GPU, 
	*		which is derived from the SISPH/PBF method.
	* 
	*/


	template<typename TDataType> class SummationDensity;
	template<typename TDataType>
	class PoissionDiskPositionShifting : public ParticleApproximation<TDataType>
	{
		DECLARE_TCLASS(IterativeDensitySolver, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		PoissionDiskPositionShifting();
		~PoissionDiskPositionShifting() override;

	public:
		DEF_VAR_IN(Real, Delta, "");

		/**
		 * @brief Particle positions
		 */
		DEF_ARRAY_IN(Coord, Position, DeviceType::GPU, "Input particle position");

		/**
		 * @brief Particle velocities
		 */
		DEF_ARRAY_IN(Coord, Velocity, DeviceType::GPU, "Input particle velocity");

		/**
		 * @brief Neighboring particles' ids
		 *
		 */
		DEF_ARRAYLIST_IN(int, NeighborIds, DeviceType::GPU, "Neighboring particles' ids");

		/**
		 * @brief Final particle densities
		 */
		DEF_ARRAY_OUT(Real, Density, DeviceType::GPU, "Final particle density");

	public:
		DEF_VAR(int, IterationNumber, 5, "Iteration number of the PBD solver");

		DEF_VAR(Real, RestDensity, 1000, "Reference density");

		DEF_VAR(Real, Kappa, Real(60), "");

	protected:
		void compute() override;

	public:
		void updatePosition();

		//void updateVelocity();

	private:
		DArray<Coord> mPosBuf;
		DArray<Coord> mPosOld;
		//DArray<Real> mDivergence;
 		DArray<Real> mDiagnals;

	private:
		std::shared_ptr<SummationDensity<TDataType>> mSummation;
	};

	IMPLEMENT_TCLASS(PoissionDiskPositionShifting, TDataType)
}