#pragma once
#include "Volume.h"

namespace dyno {
	template<typename TDataType>
	class VolumeIntersect : public Volume<TDataType>
	{
		DECLARE_CLASS_1(VolumeIntersect, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		VolumeIntersect();
		~VolumeIntersect() override;

	public:
	};
}
