#pragma once
#include "ParticleSystem.h"
#include "ParticleEmitter.h"

namespace dyno
{
	template<typename TDataType>
	class ParticleFluid : public ParticleSystem<TDataType>
	{
		DECLARE_TCLASS(ParticleFluid, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		ParticleFluid(std::string name = "default");
		virtual ~ParticleFluid();


		DEF_NODE_PORTS(ParticleEmitter<TDataType>, ParticleEmitter, "Particle Emitters");

		DEF_NODE_PORTS(ParticleSystem<TDataType>, InitialState, "Initial Fluid Particles");

	protected:
		void resetStates() override;

		void preUpdateStates();

		void loadParticleFromNode();
	};
}