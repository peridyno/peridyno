#include "ParticleEmitter.h"

namespace dyno
{
	template<typename TDataType>
	ParticleEmitter<TDataType>::ParticleEmitter(std::string name)
		: Node(name)
	{
	}

	template<typename TDataType>
	ParticleEmitter<TDataType>::~ParticleEmitter()
	{
		
	}

	template<typename TDataType>
	void ParticleEmitter<TDataType>::generateParticles()
	{

	}

	template<typename TDataType>
	void ParticleEmitter<TDataType>::updateStates()
	{
		this->generateParticles();
	}


	DEFINE_CLASS(ParticleEmitter);
}