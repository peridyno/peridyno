#pragma once
#include "Framework/Node.h"

namespace dyno {

	template<typename TDataType>
	class Volume : public Node
	{
		DECLARE_CLASS_1(Volume, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		Volume();
		~Volume() override;
	public:
	};


#ifdef PRECISION_FLOAT
template class Volume<DataType3f>;
#else
template class Volume<DataType3d>;
#endif

}
