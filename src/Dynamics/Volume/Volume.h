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

	DEFINE_CLASS(Volume);
}
