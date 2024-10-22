#pragma once
#include "ParticleSystem.h"
#include "GhostParticles.h"

namespace dyno
{
	/*!
	*	\class	GhostFluid
	*	\brief	Ghost fluid method.
	*
	*	This class implements a fluid solver coupled with ghost boundary particles.
	*
	*/
	template<typename TDataType>
	class GhostFluid : public ParticleSystem<TDataType>
	{
		DECLARE_TCLASS(GhostFluid, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		GhostFluid();
		~GhostFluid() override {};

		/**
		 * @brief Particle position for both the fluid and solid
		 */
		DEF_ARRAY_STATE(Coord, PositionMerged, DeviceType::GPU, "Particle position");

		/**
		 * @brief Particle velocity
		 */
		DEF_ARRAY_STATE(Coord, VelocityMerged, DeviceType::GPU, "Particle velocity");

		/**
		 * @brief Particle force
		 */
		DEF_ARRAY_STATE(Attribute, AttributeMerged, DeviceType::GPU, "Particle attribute");

		DEF_ARRAY_STATE(Coord, NormalMerged, DeviceType::GPU, "Particle normal");


	public:
		DEF_NODE_PORT(ParticleSystem<TDataType>, FluidParticles, "Fluid particles");
		DEF_NODE_PORT(GhostParticles<TDataType>, BoundaryParticles, "Boundary particles");

	protected:
		void resetStates() override;

		void preUpdateStates() override;

		void postUpdateStates() override;
	};
}