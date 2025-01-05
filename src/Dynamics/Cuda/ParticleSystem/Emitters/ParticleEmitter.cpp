#include "ParticleEmitter.h"

namespace dyno
{
	template<typename TDataType>
	ParticleEmitter<TDataType>::ParticleEmitter()
		: ParametricModel<TDataType>()
	{
		this->setForceUpdate(true);
		this->setAutoHidden(false);

		this->varVelocityMagnitude()->setRange(Real(0), Real(10));
		this->varSamplingDistance()->setRange(Real(0.001), Real(1.0));
		this->varSpacing()->setRange(Real(0), Real(2));
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
	void ParticleEmitter<TDataType>::resetStates()
	{
		ParametricModel<TDataType>::resetStates();

		mTimeInterval = 0;
	}

	template<typename TDataType>
	void ParticleEmitter<TDataType>::updateStates()
	{
		mPosition.clear();
		mVelocity.clear();

		Real d = (this->stateElapsedTime()->getValue() - mTimeInterval) * this->varVelocityMagnitude()->getValue();

		if (d > this->varSamplingDistance()->getValue() * this->varSpacing()->getValue())
		{
			this->generateParticles();

			mTimeInterval = this->stateElapsedTime()->getValue();
		}
	}

	template<typename TDataType>
	std::string ParticleEmitter<TDataType>::getNodeType()
	{
		return "Particle Emitters";
	}

	DEFINE_CLASS(ParticleEmitter);
}