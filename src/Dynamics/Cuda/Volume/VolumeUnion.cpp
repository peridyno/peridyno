#include "VolumeUnion.h"

namespace dyno
{
	IMPLEMENT_TCLASS(VolumeUnion, TDataType)

	template<typename TDataType>
	VolumeUnion<TDataType>::VolumeUnion()
		//: Volume()
	{
	}

	template<typename TDataType>
	VolumeUnion<TDataType>::~VolumeUnion()
	{
	}

	DEFINE_CLASS(VolumeUnion);
}