#pragma once
#include "Module/TopologyMapping.h"

#include "Volume/VoxelOctree.h"

#include "Topology/TriangleSet.h"

namespace dyno
{
	template<typename TDataType>
	class VolumeToTriangleSet : public TopologyMapping
	{
		DECLARE_TCLASS(VolumeToTriangleSet, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		VolumeToTriangleSet();
		~VolumeToTriangleSet() override;

	public:
		DEF_VAR(Real, IsoValue, Real(0), "Iso value");

		DEF_INSTANCE_IO(VoxelOctree<TDataType>, Volume, "");

 		DEF_INSTANCE_OUT(TriangleSet<TDataType>, TriangleSet, "");

	protected:
		bool apply() override;
	};

	IMPLEMENT_TCLASS(VolumeToTriangleSet, TDataType)
}