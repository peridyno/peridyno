#pragma once
#include "Module/TopologyMapping.h"

#include "Volume/VoxelOctree.h"

#include "Topology/TriangleSet.h"

namespace dyno
{
	template<typename TDataType>
	class AdaptiveVolumeToTriangleSet : public TopologyMapping
	{
		DECLARE_TCLASS(AdaptiveVolumeToTriangleSet, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		AdaptiveVolumeToTriangleSet();
		~AdaptiveVolumeToTriangleSet() override;

	public:
		DEF_VAR(Real, IsoValue, Real(0), "Iso value");

		DEF_INSTANCE_IO(VoxelOctree<TDataType>, Volume, "");

 		DEF_INSTANCE_OUT(TriangleSet<TDataType>, TriangleSet, "");

	protected:
		bool apply() override;
	};

	IMPLEMENT_TCLASS(AdaptiveVolumeToTriangleSet, TDataType)
}