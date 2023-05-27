#pragma once
#include "Volume.h"

namespace dyno {
	template<typename TDataType>
	class VolumeMinus : public Volume<TDataType>
	{
		DECLARE_TCLASS(VolumeMinus, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		VolumeMinus();
		~VolumeMinus() override;

		//void updateVolume() override {};

	public:
		DEF_NODE_PORT(Volume<TDataType>, A, "");
		DEF_NODE_PORT(Volume<TDataType>, B, "");

	private:

	};
}
