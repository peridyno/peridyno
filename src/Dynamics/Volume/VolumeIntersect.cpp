#include "VolumeIntersect.h"

namespace dyno
{
	IMPLEMENT_TCLASS(VolumeIntersect, TDataType)

	template<typename TDataType>
	VolumeIntersect<TDataType>::VolumeIntersect()
		//: Volume()
	{
	}

	template<typename TDataType>
	VolumeIntersect<TDataType>::~VolumeIntersect()
	{
	}

	DEFINE_CLASS(VolumeIntersect);
}