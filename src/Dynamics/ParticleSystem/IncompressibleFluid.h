#pragma once
#include "ParticleSystem.h"
#include "GhostParticles.h"

namespace dyno
{
	/*!
	*	\class	IncompressibleFluid
	*	\brief	Projection-based fluids.
	*
	*	This class implements an incompressible fluid solver.
	*
	*/
	template<typename TDataType>
	class IncompressibleFluid : public Node
	{
		DECLARE_CLASS_1(IncompressibleFluid, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		IncompressibleFluid();
		virtual ~IncompressibleFluid() {};

		/**
		 * @brief Particle position
		 */
		DEF_ARRAY_STATE(Coord, Position, DeviceType::GPU, "Particle position");

		/**
		 * @brief Particle velocity
		 */
		DEF_ARRAY_STATE(Coord, Velocity, DeviceType::GPU, "Particle velocity");

		/**
		 * @brief Particle force
		 */
		DEF_ARRAY_STATE(Coord, Force, DeviceType::GPU, "Force on each particle");

		/**
		 * @brief Particle force
		 */
		DEF_ARRAY_STATE(Attribute, Attribute, DeviceType::GPU, "Particle attribute");

		DEF_ARRAY_STATE(Coord, Normal, DeviceType::GPU, "Particle normal");
	public:
		DEF_NODE_PORT(ParticleSystem<TDataType>, FluidParticles, "Fluid particles");
		DEF_NODE_PORT(GhostParticles<TDataType>, BoundaryParticles, "Boundary particles");

	protected:
		void preUpdateStates() override;
		void postUpdateStates() override;
	};
}