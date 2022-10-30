#pragma once
#include "Module/ConstraintModule.h"
#include "Kernel.h"

namespace dyno {

	template<typename TDataType> class SummationDensity;

	/*!
	*	\class	DensityPBD
	*	\brief	This class implements a position-based solver for incompressibility.
	*/
	template<typename TDataType>
	class DensityPBD : public ConstraintModule
	{
		DECLARE_TCLASS(DensityPBD, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		DensityPBD();
		~DensityPBD() override;

		void constrain() override;

		void takeOneIteration();

		void updateVelocity();

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

		/**
		 * @brief Final particle densities
		 */
		DEF_ARRAY_OUT(Real, Density, DeviceType::GPU, "Final particle density");

	public:
		DEF_VAR(int, IterationNumber, 5, "Iteration number of the PBD solver");

		DEF_VAR(Real, RestDensity, 1000, "Reference density");

		DEF_VAR(Real, SamplingDistance, 0.005, "");

		DEF_VAR(Real, SmoothingLength, 0.01, "");

	private:
		SpikyKernel<Real> m_kernel;

		DArray<Real> m_lamda;
		DArray<Coord> m_deltaPos;
		DArray<Coord> m_position_old;

	private:
		std::shared_ptr<SummationDensity<TDataType>> m_summation;
	};

	IMPLEMENT_TCLASS(DensityPBD, TDataType)
}