#include "ParticleSystem.h"

namespace dyno
{
	IMPLEMENT_CLASS(ParticleSystem)

	ParticleSystem::ParticleSystem()
		: Node()
	{
		auto ptSet = std::make_shared<PointSet>();
		this->statePointSet()->setDataPtr(ptSet);
	}

	ParticleSystem::~ParticleSystem()
	{
	}

	std::string ParticleSystem::getNodeType()
	{
		return "ParticleSystem";
	}

	void ParticleSystem::resetStates()
	{
		auto ptSet = this->statePointSet()->getDataPtr();
		if (ptSet == nullptr) return;

		auto pts = ptSet->getPoints();

		if (pts.size() > 0)
		{
			this->statePosition()->resize(pts.size());
			this->stateVelocity()->resize(pts.size());
			this->stateForce()->resize(pts.size());

			this->statePosition()->getData().assign(pts);
			this->stateVelocity()->getDataPtr()->reset();
		}

		Node::resetStates();
	}
}