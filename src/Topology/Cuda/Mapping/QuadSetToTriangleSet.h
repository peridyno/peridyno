#pragma once
#include "Module/TopologyMapping.h"

#include "Topology/QuadSet.h"
#include "Topology/TriangleSet.h"

namespace dyno
{
	template<typename TDataType>
	class QuadSetToTriangleSet : public TopologyMapping
	{
		DECLARE_TCLASS(QuadSetToTriangleSet, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		QuadSetToTriangleSet();

	public:
		DEF_INSTANCE_IN(QuadSet<TDataType>, QuadSet, "");

		DEF_INSTANCE_OUT(TriangleSet<TDataType>, TriangleSet, "");

	protected:
		bool apply() override;
	};
}