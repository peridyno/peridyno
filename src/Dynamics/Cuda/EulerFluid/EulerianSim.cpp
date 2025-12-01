#include "EulerianSim.h"

namespace dyno
{
	template<typename TDataType>
	EulerianSim<TDataType>::EulerianSim()
		: Node()
	{
	}

	template<typename TDataType>
	EulerianSim<TDataType>::~EulerianSim()
	{
	}

	DEFINE_CLASS(EulerianSim);
}