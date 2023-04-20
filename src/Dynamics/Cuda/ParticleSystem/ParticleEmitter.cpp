#include "ParticleEmitter.h"

namespace dyno
{
	template<typename TDataType>
	ParticleEmitter<TDataType>::ParticleEmitter()
		: ParametricModel<TDataType>()
	{
	}

	template<typename TDataType>
	ParticleEmitter<TDataType>::~ParticleEmitter()
	{
		mPosition.clear();
		mVelocity.clear();
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

	template<typename TDataType>
	std::string ParticleEmitter<TDataType>::getNodeType()
	{
		return "Particle Emitters";
	}

	DEFINE_CLASS(ParticleEmitter);
}