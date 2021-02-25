#pragma once
#include "Volume.h"

namespace dyno {
	template<typename TDataType>
	class VolumeUnion : public Volume<TDataType>
	{
		DECLARE_CLASS_1(VolumeUnion, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		VolumeUnion();
		~VolumeUnion() override;


	public:
	};


#ifdef PRECISION_FLOAT
template class VolumeUnion<DataType3f>;
#else
template class VolumeUnion<DataType3d>;
#endif

}
