#pragma once
#include "Module/TopologyMapping.h"

#include "Collision/CollisionData.h"
#include "Topology/EdgeSet.h"

namespace dyno
{
	template<typename TDataType>
	class ContactsToEdgeSet : public TopologyMapping
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		ContactsToEdgeSet();

	protected:
		bool apply() override;

	public:
		DEF_VAR(Real, Scale, 1.0f, "A parameter to scale the normal magnitude");

		DEF_ARRAY_IN(TContactPair<Real>, Contacts, DeviceType::GPU, "");

		DEF_INSTANCE_OUT(EdgeSet<TDataType>, EdgeSet, "");
	};
}