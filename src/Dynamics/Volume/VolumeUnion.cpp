#include "VolumeUnion.h"

namespace dyno
{
	IMPLEMENT_CLASS_1(VolumeUnion, TDataType)

	template<typename TDataType>
	VolumeUnion<TDataType>::VolumeUnion()
		: Volume()
	{
	}

	template<typename TDataType>
	VolumeUnion<TDataType>::~VolumeUnion()
	{
	}
}