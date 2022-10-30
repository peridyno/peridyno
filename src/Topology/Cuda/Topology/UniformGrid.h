#pragma once
#include "Array/Array3D.h"

namespace dyno
{
	template<typename TDataType>
	class UniformGrid3D
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		UniformGrid3D();
		~UniformGrid3D();

	private:
		DArray3D<Coord> m_coords;
	};
}


