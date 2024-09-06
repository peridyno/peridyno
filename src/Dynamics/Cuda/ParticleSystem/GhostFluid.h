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
		virtual ~GhostFluid() {};

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