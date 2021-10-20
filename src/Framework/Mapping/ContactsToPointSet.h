#pragma once
#include "Module/TopologyMapping.h"

#include "Collision/CollisionData.h"
#include "Topology/PointSet.h"

namespace dyno
{
	template<typename TDataType>
	class ContactsToPointSet : public TopologyMapping
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		ContactsToPointSet();

	protected:
		bool apply() override;

	public:
		DEF_ARRAY_IN(TContactPair<Real>, Contacts, DeviceType::GPU, "");

		DEF_INSTANCE_OUT(PointSet<TDataType>, PointSet, "");
	};
}