#include "VolumeMinus.h"

namespace dyno
{
	IMPLEMENT_CLASS_1(VolumeMinus, TDataType)

	template<typename TDataType>
	VolumeMinus<TDataType>::VolumeMinus()
		: Volume()
	{
	}

	template<typename TDataType>
	VolumeMinus<TDataType>::~VolumeMinus()
	{
	}
}