#include "GhostParticles.h"

namespace dyno
{
	//IMPLEMENT_CLASS_1(GhostParticles, TDataType)

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
		this->updateTopology();
	}

	DEFINE_CLASS(GhostParticles);
}