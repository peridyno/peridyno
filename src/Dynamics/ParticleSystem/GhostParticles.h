#pragma once
#include "ParticleSystem.h"
#include "Attribute.h"

namespace dyno 
{
	template<typename TDataType>
	class GhostParticles : public ParticleSystem<TDataType>
	{
		DECLARE_CLASS_1(GhostParticles, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		GhostParticles();
		~GhostParticles() override;

		void loadPlane();

	public:
		DEF_ARRAY_STATE(Coord, Normal, DeviceType::GPU, "Ghost particle normals");

		DEF_ARRAY_STATE(Attribute, Attribute, DeviceType::GPU, "Particle attributes");

	protected:
		//void updateTopology() override;
	};
}
