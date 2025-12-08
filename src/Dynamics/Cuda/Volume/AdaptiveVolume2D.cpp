#include "AdaptiveVolume2D.h"

namespace dyno
{
	template<typename TDataType>
	AdaptiveVolume2D<TDataType>::AdaptiveVolume2D()
		: Node()
	{
		this->varDx()->setRange(0.0001, 1.0);
	}

	template<typename TDataType>
	AdaptiveVolume2D<TDataType>::~AdaptiveVolume2D()
	{
	}

	template<typename TDataType>
	void AdaptiveVolume2D<TDataType>::resetStates()
	{
		printf("AdaptiveVolume2D resetStates \n");
		if (this->stateAGridSet()->isEmpty() == false)
		{
			auto m_AGrid = this->stateAGridSet()->getDataPtr();
			m_AGrid->update();
		}
	}

	template<typename TDataType>
	void AdaptiveVolume2D<TDataType>::updateStates()
	{
		Node::updateStates();

		if (this->stateAGridSet()->isEmpty() == false)
		{
			auto m_AGrid = this->stateAGridSet()->getDataPtr();
			m_AGrid->update();
		}
	}

	DEFINE_CLASS(AdaptiveVolume2D);
}