#pragma once
#include "Array/Array.h"
#include "Module/CollidableObject.h"

namespace dyno
{
	class TopologyMapping;

	template<typename TDataType>
	class CollidablePoints : public CollidableObject
	{
		DECLARE_TCLASS(CollidablePoints, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Rigid Rigid;
		typedef typename TDataType::Matrix Matrix;

		CollidablePoints();
		virtual ~CollidablePoints();

		void setRadius(Real radius) { m_radius = radius; }
		void setRadii(DArray<Coord>& radii);

		void setPositions(DArray<Coord>& centers);
		void setVelocities(DArray<Coord>& vel);

		DArray<Coord>& getPositions() { return m_positions; }
		DArray<Coord>& getVelocities() { return m_velocities; }

		void updateCollidableObject() override;
		void updateMechanicalState() override;

	private:
		std::shared_ptr<TopologyMapping> m_mapping;

		bool m_bUniformRaidus;
		Real m_radius;
		
		DArray<Coord> m_positions;
		DArray<Coord> m_velocities;
	};
}
