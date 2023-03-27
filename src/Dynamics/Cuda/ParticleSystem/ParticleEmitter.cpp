#include "ParticleEmitter.h"

namespace dyno
{
	template<typename TDataType>
	ParticleEmitter<TDataType>::ParticleEmitter(std::string name)
		: Node(name)
	{
// 		this->varScale()->setValue(Vec3f(1, 1, 1));
// 		this->varScale()->setMin(0.01);
// 		this->varScale()->setMax(100.0f);
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