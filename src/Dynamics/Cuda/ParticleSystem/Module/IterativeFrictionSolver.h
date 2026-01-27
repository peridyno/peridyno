#pragma once
#include "ParticleApproximation.h"

namespace dyno {

	template<typename TDataType> class SummationDensity;

	/*!
	*	\class	DensityPBD
	*	\brief	This class implements a position-based solver for incompressibility.
	*/
	template<typename TDataType>
	class IterativeFrictionSolver : public ParticleApproximation<TDataType>
	{
		DECLARE_TCLASS(IterativeDensitySolver, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		IterativeFrictionSolver();
		~IterativeFrictionSolver() override;

	public:
		DEF_VAR_IN(Real, TimeStep, "Time Step");

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

	public:
		DEF_VAR(int, IterationNumber, 5, "Iteration number of the PBD solver");

		DEF_VAR(Real, RestDensity, 1000, "Reference density");

	protected:
		void compute() override;

	public:
		void takeOneIteration();

		void updateVelocity();

	private:
		DArray<Coord> mPosBuf1;
		DArray<Coord> mPosBuf0;
		DArray<Coord> mPositionOld;
	};

	IMPLEMENT_TCLASS(IterativeFrictionSolver, TDataType)
}