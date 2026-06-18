#pragma once
#include "Module/TopologyMapping.h"

#include "Topology/AdaptiveGridSet.h"

#include "Topology/EdgeSet.h"

namespace dyno
{
	template<typename TDataType>
	class VolumeToGridCell : public TopologyMapping
	{
		DECLARE_TCLASS(VolumeToGridCell, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		VolumeToGridCell();
		~VolumeToGridCell() override;

	public:
		DEF_VAR(Real, IsoValue, Real(0), "Iso value");

		DEF_INSTANCE_IN(AdaptiveGridSet<TDataType>, Volume, "");

 		DEF_INSTANCE_OUT(EdgeSet<TDataType>, GridCell, "");

	protected:
		bool apply() override;
	};

	IMPLEMENT_TCLASS(VolumeToGridCell, TDataType)
}