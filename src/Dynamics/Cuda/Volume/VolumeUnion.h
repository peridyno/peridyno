#pragma once
#include "Volume.h"

namespace dyno {
	template<typename TDataType>
	class VolumeUnion : public Volume<TDataType>
	{
		DECLARE_TCLASS(VolumeUnion, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		VolumeUnion();
		~VolumeUnion() override;

		//void updateVolume() override {};

	public:
		DEF_NODE_PORT(Volume<TDataType>, A, "");
		DEF_NODE_PORT(Volume<TDataType>, B, "");
	};
}
