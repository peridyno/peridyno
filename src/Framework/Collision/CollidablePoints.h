#pragma once
#include "Array/Array.h"
#include "Framework/CollidableObject.h"
#include "Framework/FieldArray.h"

namespace dyno
{
	class TopologyMapping;

	template<typename TDataType>
	class CollidablePoints : public CollidableObject
	{
		DECLARE_CLASS_1(CollidablePoints, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Rigid Rigid;
		typedef typename TDataType::Matrix Matrix;

		CollidablePoints();
		virtual ~CollidablePoints();

		void setRadius(Real radius) { m_radius = radius; }
		void setRadii(DeviceArray<Coord>& radii);

		void setPositions(DeviceArray<Coord>& centers);
		void setVelocities(DeviceArray<Coord>& vel);

		DeviceArray<Coord>& getPositions() { return m_positions; }
		DeviceArray<Coord>& getVelocities() { return m_velocities; }

		bool initializeImpl() override;

		void updateCollidableObject() override;
		void updateMechanicalState() override;

	private:
		std::shared_ptr<TopologyMapping> m_mapping;

		bool m_bUniformRaidus;
		Real m_radius;
		
		DeviceArray<Coord> m_positions;
		DeviceArray<Coord> m_velocities;
	};


#ifdef PRECISION_FLOAT
	template class CollidablePoints<DataType3f>;
#else
	template class CollidablePoints<DataType3d>;
#endif
}
