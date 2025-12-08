#include "AdaptiveGridGenerator.h"
#include "Algorithm/CudaRand.h"

namespace dyno
{
	template<typename TDataType>
	AdaptiveGridGenerator<TDataType>::AdaptiveGridGenerator()
		: ComputeModule()
	{
	}

	template<typename TDataType>
	AdaptiveGridGenerator<TDataType>::~AdaptiveGridGenerator()
	{
	}

	template<typename TDataType>
	void AdaptiveGridGenerator<TDataType>::compute()
	{
		if (this->inAGridSet()->isEmpty() == false)
		{
			auto m_AGrid = this->inAGridSet()->getDataPtr();
			m_AGrid->setNeighborType(this->varNeighMode()->currentKey());
			m_AGrid->update();
		}
	}

	DEFINE_CLASS(AdaptiveGridGenerator);
}