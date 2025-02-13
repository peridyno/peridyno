#pragma once
#include "Module/TopologyMapping.h"


#include "Topology/TriangleSet.h"
#include "Topology/HeightField.h"

namespace dyno
{
	template<typename TDataType>
	class ApplyBumpMap2TriangleSet : public TopologyMapping
	{
		DECLARE_TCLASS(ApplyBumpMap2TriangleSet, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename ::dyno::Vector<Real, 3> Coord3D;

		ApplyBumpMap2TriangleSet();

	public:
		DEF_INSTANCE_IN(TriangleSet<TDataType>, TriangleSet, "");

		DEF_INSTANCE_IN(HeightField<TDataType>, HeightField, "");

		DEF_INSTANCE_OUT(TriangleSet<TDataType>, TriangleSet, "");

	protected:
		bool apply() override;
	};
}