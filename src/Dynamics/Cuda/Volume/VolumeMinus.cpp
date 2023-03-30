#include "VolumeMinus.h"

namespace dyno
{
	IMPLEMENT_TCLASS(VolumeMinus, TDataType)

	template<typename TDataType>
	VolumeMinus<TDataType>::VolumeMinus()
		//: Volume()
	{
	}

	template<typename TDataType>
	VolumeMinus<TDataType>::~VolumeMinus()
	{
	}

	DEFINE_CLASS(VolumeMinus);
}