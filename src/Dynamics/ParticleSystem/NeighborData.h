#pragma once
#include "Platform.h"

namespace dyno
{
	template<typename TDataType>
	class TPair
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		DYN_FUNC TPair() {};
		DYN_FUNC TPair(int id, Coord p)
		{
			index = id;
			pos = p;
		}

		int index;
		Coord pos;
		Real weight = Real(1.0);
	};

}