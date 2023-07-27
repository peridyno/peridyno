#include "ParticleEmitter.h"

namespace dyno
{
	ParticleEmitter::ParticleEmitter()
		: ParametricModel<DataType3f>()
	{
		this->varVelocityMagnitude()->setRange(0.0f, 10.0f);
		this->varSamplingDistance()->setRange(0.001f, 1.0f);

		this->allowExported(true);
	}

	ParticleEmitter::~ParticleEmitter()
	{
		mPosition.clear();
		mVelocity.clear();
	}

	void ParticleEmitter::generateParticles()
	{

	}

	void ParticleEmitter::updateStates()
	{
		this->generateParticles();
	}

	std::string ParticleEmitter::getNodeType()
	{
		return "Particle Emitters";
	}
}