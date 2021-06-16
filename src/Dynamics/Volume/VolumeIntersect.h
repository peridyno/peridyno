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

		void updateVolume() override {};

	public:
		DEF_PORT_IN(DistanceField3D<TDataType>, A, "");
		DEF_PORT_IN(DistanceField3D<TDataType>, B, "");

	private:
	};
}
