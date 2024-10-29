#include "GhostParticles.h"

namespace dyno
{
	//IMPLEMENT_TCLASS(GhostParticles, TDataType)

	template<typename TDataType>
	GhostParticles<TDataType>::GhostParticles()
		: ParticleSystem<TDataType>()
	{
	}

	template<typename TDataType>
	GhostParticles<TDataType>::~GhostParticles()
	{
	}

	template<typename TDataType>
	void GhostParticles<TDataType>::resetStates()
	{
		ParticleSystem<TDataType>::resetStates();
	}

	DEFINE_CLASS(GhostParticles);
}