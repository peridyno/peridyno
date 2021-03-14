#pragma once
#include "Array/Array.h"
#include "Framework/ModuleConstraint.h"
#include "Topology/FieldNeighbor.h"
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
		DECLARE_CLASS_1(DensityPBD, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		DensityPBD();
		~DensityPBD() override;

		bool constrain() override;

		void takeOneIteration();

		void updateVelocity();

	public:
		DeviceArrayField<Real> m_massInv; // mass^-1 as described in unified particle physics

	public:
		DEF_EMPTY_VAR(IterationNumber, int, "Iteration number of the PBD solver");

		DEF_EMPTY_VAR(RestDensity, Real, "Reference density");

		DEF_EMPTY_VAR(SamplingDistance, Real, "");

		DEF_EMPTY_VAR(SmoothingLength, Real, "");


		/**
		 * @brief Particle positions
		 */
		DEF_EMPTY_IN_ARRAY(Position, Coord, DeviceType::GPU, "Input particle position");

		/**
		 * @brief Particle velocities
		 */
		DEF_EMPTY_IN_ARRAY(Velocity, Coord, DeviceType::GPU, "Input particle velocity");


		/**
		 * @brief Neighboring particles' ids
		 *
		 */
		DEF_EMPTY_IN_NEIGHBOR_LIST(NeighborIndex, int, "Neighboring particles' ids");

		/**
		 * @brief New particle positions
		 */
		DEF_EMPTY_OUT_ARRAY(Position, Coord, DeviceType::GPU, "Output particle position");

		/**
		 * @brief New particle velocities
		 */
		DEF_EMPTY_OUT_ARRAY(Velocity, Coord, DeviceType::GPU, "Output particle velocity");

		/**
		 * @brief Final particle densities
		 */
		DEF_EMPTY_OUT_ARRAY(Density, Real, DeviceType::GPU, "Final particle density");

	private:
		SpikyKernel<Real> m_kernel;

		GArray<Real> m_lamda;
		GArray<Coord> m_deltaPos;
		GArray<Coord> m_position_old;

	private:
		std::shared_ptr<SummationDensity<TDataType>> m_summation;
	};



}