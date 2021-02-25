#pragma once
#include "Array/Array.h"
#include "Framework/CollidableObject.h"

namespace dyno
{
	template<typename TDataType>
	class CollidableSpheres : public CollidableObject
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		CollidableSpheres();
		virtual ~CollidableSpheres();

		void setRadius(Real radius) { m_radius = radius; }
		void setCenters(DeviceArray<Coord>& centers);

	private:
		Real m_radius;
		DeviceArray<Coord> m_centers;
	};



}
