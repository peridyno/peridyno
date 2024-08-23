#include "OceanBase.h"

namespace dyno
{
	template<typename TDataType>
	OceanBase<TDataType>::OceanBase()
		: Node()
	{
	}

	template<typename TDataType>
	OceanBase<TDataType>::~OceanBase()
	{
	}

	template<typename TDataType>
	bool OceanBase<TDataType>::validateInputs()
	{
		return this->getOceanPatch() != nullptr;
	}

	DEFINE_CLASS(OceanBase);
}

