#pragma once
#include "Module/TopologyMapping.h"

#include "Topology/HeightField.h"
#include "Topology/TriangleSet.h"

namespace dyno
{
	template<typename TDataType>
	class HeightFieldToTriangleSet : public TopologyMapping
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		HeightFieldToTriangleSet();

	protected:
		bool apply() override;

	public:
		DEF_INSTANCE_IN(HeightField<TDataType>, HeightField, "");
		DEF_INSTANCE_OUT(TriangleSet<TDataType>, TriangleSet, "");

		DEF_VAR(Real, Scale, Real(1), "");
		DEF_VAR(Coord, Translation, Coord(0), "");

	private:
		TriangleSet<TDataType> mStandardSphere;
		TriangleSet<TDataType> mStandardCapsule;
	};
}