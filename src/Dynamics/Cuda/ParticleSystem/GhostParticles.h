#pragma once
#include "ParticleSystem.h"
#include "Collision/Attribute.h"

namespace dyno 
{
	template<typename TDataType>
	class GhostParticles : public ParticleSystem<TDataType>
	{
		DECLARE_TCLASS(GhostParticles, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		GhostParticles();
		~GhostParticles() override;

	public:
		DEF_ARRAY_STATE(Coord, Normal, DeviceType::GPU, "Ghost particle normals");

		DEF_ARRAY_STATE(Attribute, Attribute, DeviceType::GPU, "Particle attributes");

	protected:
		void resetStates() override;
	};

	IMPLEMENT_TCLASS(GhostParticles, TDataType)
}
