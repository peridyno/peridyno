#pragma once
#include "Module/TopologyMapping.h"

#include "Collision/CollisionData.h"
#include "Topology/EdgeSet.h"

#include "Primitive/Primitive3D.h"

namespace dyno
{
	template<typename TDataType>
	class BoundingBoxToEdgeSet : public TopologyMapping
	{
		DECLARE_TCLASS(BoundingBoxToEdgeSet, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename ::dyno::TAlignedBox3D<Real> AABB;

		BoundingBoxToEdgeSet();

	protected:
		bool apply() override;

	public:
		DEF_ARRAY_IN(AABB, AABB, DeviceType::GPU, "");

		DEF_INSTANCE_OUT(EdgeSet<TDataType>, EdgeSet, "");
	};
}