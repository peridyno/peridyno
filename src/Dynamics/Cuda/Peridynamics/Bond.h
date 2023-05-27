#pragma once
#include "Platform.h"

namespace dyno
{
	/**
	 * @brief Definition of a bond in Peridynamics
	 */
	template<typename TDataType>
	class TBond
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		DYN_FUNC TBond() {};
		DYN_FUNC TBond(int id, Coord p)
		{
			index = id;
			pos = p;
		}

		int index;
		Coord pos;
		Real weight = Real(1.0);
	};

}