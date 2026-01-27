#include "AdaptiveVolume.h"

namespace dyno
{
	template<typename TDataType>
	AdaptiveVolume<TDataType>::AdaptiveVolume()
		: Node()
	{
		this->varDx()->setRange(0.001, 1.0);
	}

	template<typename TDataType>
	AdaptiveVolume<TDataType>::~AdaptiveVolume()
	{
	}

	DEFINE_CLASS(AdaptiveVolume);
}