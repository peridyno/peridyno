#pragma once
#include "Volume.h"

namespace dyno {
	template<typename TDataType>
	class VolumeIntersect : public Volume<TDataType>
	{
		DECLARE_TCLASS(VolumeIntersect, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		VolumeIntersect();
		~VolumeIntersect() override;

		//void updateVolume() override {};

	public:
		DEF_NODE_PORT(Volume<TDataType>, A, "");
		DEF_NODE_PORT(Volume<TDataType>, B, "");

	private:
	};
}
