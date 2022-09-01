#include "MakeParticleSystem.h"

namespace dyno
{
	template<typename TDataType>
	MakeParticleSystem<TDataType>::MakeParticleSystem()
		: ParticleSystem<TDataType>()
	{
	}

	template<typename TDataType>
	MakeParticleSystem<TDataType>::~MakeParticleSystem()
	{
	}

	template<typename TDataType>
	void MakeParticleSystem<TDataType>::resetStates()
	{
		auto& inTopo = this->inPoints()->getData();

		Coord vel = this->varInitialVelocity()->getData();

		std::vector<Coord> hostVel;
		hostVel.assign(inTopo.getPoints().size(), vel);

		auto pts = std::make_shared<PointSet<TDataType>>();
		pts->copyFrom(inTopo);

		this->statePointSet()->setDataPtr(pts);
		this->statePosition()->assign(pts->getPoints());
		this->stateVelocity()->assign(hostVel);

		this->stateForce()->resize(inTopo.getPoints().size());
		this->stateForce()->reset();

		hostVel.clear();
	}

	DEFINE_CLASS(MakeParticleSystem);
}