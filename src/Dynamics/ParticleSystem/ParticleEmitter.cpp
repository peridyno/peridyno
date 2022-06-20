#include "ParticleEmitter.h"

namespace dyno
{
	template<typename TDataType>
	ParticleEmitter<TDataType>::ParticleEmitter(std::string name)
		: Node(name)
	{
		this->varScale()->setValue(Vec3f(1, 1, 1));
		this->varScale()->setMin(0.01);
		this->varScale()->setMax(100.0f);
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