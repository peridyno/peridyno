#include "VolumeIntersect.h"

namespace dyno
{
	IMPLEMENT_CLASS_1(VolumeIntersect, TDataType)

	template<typename TDataType>
	VolumeIntersect<TDataType>::VolumeIntersect()
		: Volume()
	{
	}

	template<typename TDataType>
	VolumeIntersect<TDataType>::~VolumeIntersect()
	{
	}
}