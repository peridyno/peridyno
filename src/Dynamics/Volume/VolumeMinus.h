#pragma once
#include "Volume.h"

namespace dyno {
	template<typename TDataType>
	class VolumeMinus : public Volume<TDataType>
	{
		DECLARE_CLASS_1(VolumeMinus, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		VolumeMinus();
		~VolumeMinus() override;


	public:
	};


#ifdef PRECISION_FLOAT
template class VolumeMinus<DataType3f>;
#else
template class VolumeMinus<DataType3d>;
#endif

}
