#include "ParticleSystem.h"

#include "Topology/PointSet.h"

namespace dyno
{
	template<typename TDataType>
	ParticleSystem<TDataType>::ParticleSystem()
		: Node()
	{
		auto ptSet = std::make_shared<PointSet<TDataType>>();
		this->statePointSet()->setDataPtr(ptSet);
	}

	template<typename TDataType>
	ParticleSystem<TDataType>::~ParticleSystem()
	{
	}

	template<typename TDataType>
	std::string ParticleSystem<TDataType>::getNodeType()
	{
		return "Particle Fluids";
	}

	template<typename TDataType>
	void ParticleSystem<TDataType>::resetStates()
	{
		auto ptSet = this->statePointSet()->getDataPtr();
		if (ptSet != nullptr) {
			if (!this->statePosition()->isEmpty())
			{
				ptSet->setPoints(this->statePosition()->constData());
			}
			else
				ptSet->clear();
		}

		Node::resetStates();
	}

	template<typename TDataType>
	void ParticleSystem<TDataType>::postUpdateStates()
	{
		if (!this->statePosition()->isEmpty())
		{
			auto ptSet = this->statePointSet()->getDataPtr();
			int num = this->statePosition()->size();
			auto& pts = ptSet->getPoints();
			if (num != pts.size())
			{
				pts.resize(num);
			}

			pts.assign(this->statePosition()->getData());
		}
		else
		{
			auto points = this->statePointSet()->getDataPtr();
			points->clear();
		}
	}

	DEFINE_CLASS(ParticleSystem);
}