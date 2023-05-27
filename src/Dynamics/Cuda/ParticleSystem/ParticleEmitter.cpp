#include "ParticleEmitter.h"

namespace dyno
{
	template<typename TDataType>
	ParticleEmitter<TDataType>::ParticleEmitter()
		: ParametricModel<TDataType>()
	{
		this->varVelocityMagnitude()->setRange(Real(0), Real(10));
		this->varSamplingDistance()->setRange(Real(0.001), Real(1.0));

		this->allowExported(true);
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