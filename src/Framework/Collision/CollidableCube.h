#pragma once
#include "Array/Array.h"
#include "Framework/CollidableObject.h"

namespace dyno
{
	template<typename TDataType>
	class CollidableCube : public CollidableObject
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		CollidableCube();
		virtual ~CollidableCube();

	private:
		Coord m_length;
		Coord m_center;
	};



}
